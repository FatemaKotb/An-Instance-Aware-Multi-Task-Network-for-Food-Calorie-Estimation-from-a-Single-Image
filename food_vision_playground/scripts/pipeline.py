from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch

from scripts.dtypes import (
    InstanceOutputs,
    PipelineOutput,
    FusionOutput,
    PredictionOutput,
    PhysicsOutput,
)
from scripts.level4_physics.physics_head import PhysicsHead


class Pipeline:
    """
    Full inference pipeline (Level 4, physics-based calorie estimation).

    Steps:
    1) Backbone features (EfficientNet)
    2) Instance segmentation (Mask-RCNN)
    3) Depth estimation (ZoeDepth)
    4) Instance-aware fusion (semantic + geometric)
    5) Prediction head (food class + portion)
    6) Physics head (area/volume/calories)
    """

    def __init__(
        self,
        backbone_block,
        seg_block,
        depth_block,
        fusion_block,
        prediction_head,
        physics_head: PhysicsHead,
        device: str = "cuda",
        seg_score_thresh: float = 0.5,
    ):
        self.backbone_block = backbone_block
        self.seg_block = seg_block
        self.depth_block = depth_block
        self.fusion_block = fusion_block
        self.prediction_head = prediction_head
        self.physics_head = physics_head
        self.device = device
        self.seg_score_thresh = seg_score_thresh

    # ---------- Utilities ----------

    def _mask_pool_feature(self, feat: torch.Tensor, mask_hw: np.ndarray) -> torch.Tensor:
        """
        Pool backbone feature tensor using mask.
        feat: [1,C,Hf,Wf]
        mask_hw: [H,W] boolean
        Returns: pooled feature vector [C] on CPU
        """
        assert feat.ndim == 4 and feat.shape[0] == 1
        _, C, Hf, Wf = feat.shape

        mask = torch.from_numpy(mask_hw.astype(np.float32))[None, None]  # [1,1,H,W]
        mask = torch.nn.functional.interpolate(mask, size=(Hf, Wf), mode="nearest")
        mask = mask.to(feat.device)

        denom = mask.sum().clamp(min=1.0)
        pooled = (feat * mask).sum(dim=(2, 3)).squeeze(0) / denom.squeeze()  # [C]
        return pooled.detach().cpu()

    def _mask_pool_depth(self, depth_hw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float]:
        """
        Compute per-instance depth statistics (median + mean)
        """
        vals = depth_hw[mask_hw]
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.median(vals)), float(np.mean(vals))

    # ---------- Main pipeline runner ----------

    @torch.inference_mode()
    def run(self, img_rgb_uint8: np.ndarray) -> PipelineOutput:
        """
        Runs the full pipeline and returns structured output.
        """
        # 1) Backbone features
        back_out = self.backbone_block(img_rgb_uint8)
        feats = back_out.features  # list of [1,C,Hf,Wf]

        # 2) Segmentation
        seg_out = self.seg_block(img_rgb_uint8, score_thresh=self.seg_score_thresh)
        boxes = seg_out.boxes_xyxy
        seg_class_ids = seg_out.class_ids
        scrs = seg_out.scores
        masks = seg_out.masks

        # 3) Depth map
        depth_hw = self.depth_block(img_rgb_uint8).depth  # [H,W]

        # 4) Per-instance processing
        instance_outputs: List[InstanceOutputs] = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            area_px = int(mask.sum())

            # Feature pooling
            pooled_feats = [self._mask_pool_feature(f, mask) for f in feats]

            # Depth stats
            d_med, d_mean = self._mask_pool_depth(depth_hw, mask)

            # Fusion
            fusion_out: FusionOutput = self.fusion_block(pooled_feats, d_med, d_mean)

            # Prediction
            pred: PredictionOutput = self.prediction_head(
                img_rgb_uint8=img_rgb_uint8,
                box_xyxy=boxes[i],
                mask_hw=mask,
                fusion=fusion_out,
            )

            # Physics-based calorie estimation
            phys: PhysicsOutput = self.physics_head(
                mask_hw=mask,
                depth_median=d_med,
                prediction=pred,
            )

            instance_outputs.append(
                InstanceOutputs(
                    mask=mask,
                    box_xyxy=boxes[i],
                    score=float(scrs[i]),
                    seg_class_id=int(seg_class_ids[i]),
                    area_px=area_px,
                    depth_median=d_med,
                    depth_mean=d_mean,
                    fusion=fusion_out,
                    prediction=pred,
                    physics=phys,
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
