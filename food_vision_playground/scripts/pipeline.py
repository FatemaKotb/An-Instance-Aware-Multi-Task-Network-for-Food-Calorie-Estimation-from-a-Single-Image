from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch

from scripts.types import InstanceOutputs, PipelineOutput
from scripts.fusion_stub import FusionStub


# ---------- Utilities ----------

def _mask_pool_feature(feat: torch.Tensor, mask_hw: np.ndarray) -> torch.Tensor:
    """
    feat: [1,C,Hf,Wf]
    mask_hw: [H,W] bool (image resolution)
    Returns: pooled vector [C] (CPU tensor)
    """
    assert feat.ndim == 4 and feat.shape[0] == 1
    _, C, Hf, Wf = feat.shape

    mask = torch.from_numpy(mask_hw.astype(np.float32))[None, None]  # [1,1,H,W]
    mask = torch.nn.functional.interpolate(mask, size=(Hf, Wf), mode="nearest")  # [1,1,Hf,Wf]
    mask = mask.to(feat.device)

    # avoid empty masks
    denom = mask.sum().clamp(min=1.0)
    pooled = (feat * mask).sum(dim=(2, 3)).squeeze(0) / denom.squeeze()  # [C]
    return pooled.detach().cpu()

def _mask_pool_depth(depth_hw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float]:
    vals = depth_hw[mask_hw]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.median(vals)), float(np.mean(vals))


# ---------- Main pipeline runner ----------

@torch.inference_mode()
def run_pipeline(
    img_rgb_uint8: np.ndarray,
    backbone_model,
    backbone_transform,
    seg_model,
    seg_preprocess,
    depth_model,
    depth_fn,   # callable(depth_model, img_rgb_uint8, device)-> np.ndarray [H,W]
    device: str = "cuda",
    seg_score_thresh: float = 0.5,
    fusion=None,
) -> PipelineOutput:
    """
    backbone_model/backbone_transform: from your unified efficientnet loader (mode='backbone')
    seg_model/seg_preprocess: from your torchvision Mask R-CNN loader
    depth_model + depth_fn: any depth model wrapper that returns a depth map [H,W]
    fusion: FusionStub or your future fusion module
    """
    if fusion is None:
        fusion = FusionStub()

    H, W = img_rgb_uint8.shape[:2]

    # 1) Backbone features (multi-scale)
    from PIL import Image
    img_pil = Image.fromarray(img_rgb_uint8)
    x_back = backbone_transform(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    feats = backbone_model(x_back)  # list of [1,C,Hf,Wf]

    # 2) Segmentation instances (torchvision-style)
    x_seg = seg_preprocess(img_pil).to(device)  # [C,H,W]
    out = seg_model([x_seg])[0]
    scores = out["scores"].detach().cpu().numpy()
    keep = scores >= seg_score_thresh

    boxes = out["boxes"].detach().cpu().numpy()[keep]           # [N,4]
    cls_ids = out["labels"].detach().cpu().numpy()[keep]        # [N]
    scrs = scores[keep]                                         # [N]
    masks = out["masks"].detach().cpu().numpy()[keep, 0] >= 0.5 # [N,H,W] bool

    # 3) Depth map
    depth_hw = depth_fn(depth_model, img_rgb_uint8, device=device)  # [H,W] float32

    # 4) Build per-instance inputs + 5) Stub fusion
    instance_outputs: List[InstanceOutputs] = []
    for i in range(masks.shape[0]):
        m = masks[i]
        area_px = int(m.sum())

        pooled_feats = [_mask_pool_feature(f, m) for f in feats]  # list of [C_i] on CPU
        d_med, d_mean = _mask_pool_depth(depth_hw, m)

        emb = fusion(pooled_feats, d_med, d_mean)  # [D] CPU tensor

        instance_outputs.append(
            InstanceOutputs(
                mask=m,
                box_xyxy=boxes[i],
                score=float(scrs[i]),
                class_id=int(cls_ids[i]),
                embedding=emb,
                area_px=area_px,
                depth_median=d_med,
                depth_mean=d_mean,
            )
        )

    return PipelineOutput(
        instance_outputs=instance_outputs,
        depth_map=depth_hw.astype(np.float32),
        backbone_features=feats,
    )
