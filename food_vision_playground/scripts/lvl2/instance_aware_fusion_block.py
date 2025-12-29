from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.ops import roi_align

from scripts.dtypes import FusionOutput


def _resize_to_roi(arr_hw: np.ndarray, box_xyxy: np.ndarray, out_hw: int = 7) -> np.ndarray:
    """Crop arr (H,W) by box and resize to (out_hw,out_hw)."""
    H, W = arr_hw.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = int(max(0, min(W - 1, round(x1))))
    x2 = int(max(0, min(W, round(x2))))
    y1 = int(max(0, min(H - 1, round(y1))))
    y2 = int(max(0, min(H, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_hw, out_hw), dtype=arr_hw.dtype)

    crop = arr_hw[y1:y2, x1:x2]
    pil = Image.fromarray(crop)
    pil = pil.resize((out_hw, out_hw), resample=Image.NEAREST)
    return np.asarray(pil, dtype=arr_hw.dtype)


def _depth_stats_from_mask(depth_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Layout: [mean, std, mad, q10, q25, q50, q75, q90]."""
    vals = depth_hw[mask_hw.astype(bool)].astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((8,), dtype=np.float32)

    q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90]).astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    mad = float(np.median(np.abs(vals - q50)))
    return np.array([mean, std, mad, q10, q25, q50, q75, q90], dtype=np.float32)


class InstanceAwareFusionBlock:
    """Prepares ROI tensors needed by InstanceAwareFusionHead (classifier).

    Produces:
      - roi_feat: [C,7,7] via ROIAlign on last backbone feature map
      - mask_roi: [1,7,7] resized binary mask inside bbox
      - depth_roi:[1,7,7] resized depth inside bbox masked by mask_roi
      - depth_stats: [8] stats over full-res masked depth

    Also produces legacy fusion fields so the pipeline stays compatible.
    """

    def __init__(self, *, out_hw: int = 7, sampling_ratio: int = 2):
        self.out_hw = int(out_hw)
        self.sampling_ratio = int(sampling_ratio)

    def __call__(
        self,
        pooled_feats: List[torch.Tensor],
        depth_median: float,
        depth_mean: float,
        *,
        img_rgb_uint8: np.ndarray,
        box_xyxy: np.ndarray,
        mask_hw: np.ndarray,
        depth_hw: np.ndarray,
        backbone_last_feat: torch.Tensor,
    ) -> FusionOutput:
        masked_features = [p.detach().cpu().float() for p in pooled_feats]
        global_features = torch.cat([p.float() for p in masked_features], dim=0)
        instance_depth = torch.tensor([depth_median, depth_mean], dtype=torch.float32)

        H_img, W_img = int(img_rgb_uint8.shape[0]), int(img_rgb_uint8.shape[1])
        _, _, _, Wf = backbone_last_feat.shape
        spatial_scale = float(Wf) / float(W_img)

        b = box_xyxy.astype(np.float32)
        rois = torch.tensor([[0, b[0], b[1], b[2], b[3]]], device=backbone_last_feat.device, dtype=torch.float32)
        roi_feat = roi_align(
            input=backbone_last_feat,
            boxes=rois,
            output_size=(self.out_hw, self.out_hw),
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        ).squeeze(0)  # [C,7,7]

        mask_u8 = mask_hw.astype(np.uint8)
        mask_roi = _resize_to_roi(mask_u8, b, out_hw=self.out_hw).astype(np.float32)
        mask_roi = (mask_roi > 0.5).astype(np.float32)[None, ...]  # [1,7,7]

        depth_roi = _resize_to_roi(depth_hw.astype(np.float32), b, out_hw=self.out_hw).astype(np.float32)
        depth_roi = (depth_roi * (mask_roi[0] > 0.5)).astype(np.float32)[None, ...]  # [1,7,7]

        d_stats = _depth_stats_from_mask(depth_hw.astype(np.float32), mask_hw.astype(bool))
        d_stats[0] = float(depth_mean)     # mean
        d_stats[5] = float(depth_median)   # median (q50)

        return FusionOutput(
            masked_features=masked_features,
            global_features=global_features,
            instance_depth=instance_depth,
            roi_feat=roi_feat.detach().cpu().to(torch.float16),
            mask_roi=torch.from_numpy(mask_roi),
            depth_roi=torch.from_numpy(depth_roi),
            depth_stats=torch.from_numpy(d_stats.astype(np.float32)),
        )
