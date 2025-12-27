from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch

from scripts.dtypes import InstanceOutputs, PipelineOutput
from scripts.fusion_stub import FusionStub


class Pipeline:
    """
    Inference pipeline that wires:
      - backbone features (multi-scale)
      - segmentation instances (masks/boxes/scores)
      - depth map
      - per-instance pooling
      - fusion block (stub now, real later)

    This class is designed so you can replace fusion later without changing anything else.
    """

    def __init__(
        self,
        backbone_block,
        seg_block,
        depth_block,
        device: str = "cuda",
        seg_score_thresh: float = 0.5,
        fusion: Optional[object] = None,
    ):
        self.backbone_block = backbone_block
        self.seg_block = seg_block
        self.depth_block = depth_block
        self.device = device
        self.seg_score_thresh = seg_score_thresh

        # ---------- Stub fusion module (replace later) ----------
        if fusion is None:
            fusion = FusionStub()
        self.fusion = fusion

    def _mask_pool_feature(self, feat: torch.Tensor, mask_hw: np.ndarray) -> torch.Tensor:
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

    def _mask_pool_depth(self, depth_hw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float]:
        vals = depth_hw[mask_hw]
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.median(vals)), float(np.mean(vals))

    @torch.inference_mode()
    def run(self, img_rgb_uint8: np.ndarray) -> PipelineOutput:
        """
        backbone_model/backbone_transform: from your unified efficientnet loader (mode='backbone')
        seg_model/seg_preprocess: from your torchvision Mask R-CNN loader
        depth_model + depth_fn: any depth model wrapper that returns a depth map [H,W]
        fusion: FusionStub or your future fusion module
        """
        H, W = img_rgb_uint8.shape[:2]

        # 1) Backbone features (multi-scale)
        back_out = self.backbone_block(img_rgb_uint8)
        feats = back_out.features  # list of [1,C,Hf,Wf]

        # 2) Segmentation instances (torchvision-style)
        seg_out = self.seg_block(img_rgb_uint8, score_thresh=self.seg_score_thresh)
        boxes = seg_out.boxes_xyxy
        cls_ids = seg_out.class_ids
        scrs = seg_out.scores
        masks = seg_out.masks

        # 3) Depth map
        depth_hw = self.depth_block(img_rgb_uint8).depth  # [H,W] float32

        # 4) Build per-instance inputs + 5) Stub fusion
        instance_outputs: List[InstanceOutputs] = []
        for i in range(masks.shape[0]):
            m = masks[i]
            area_px = int(m.sum())

            pooled_feats = [self._mask_pool_feature(f, m) for f in feats]  # list of [C_i] on CPU
            d_med, d_mean = self._mask_pool_depth(depth_hw, m)

            emb = self.fusion(pooled_feats, d_med, d_mean)  # [D] CPU tensor

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

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray) -> PipelineOutput:
        return self.run(img_rgb_uint8)
