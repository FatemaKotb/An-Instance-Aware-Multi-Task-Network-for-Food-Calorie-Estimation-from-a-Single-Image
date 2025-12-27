from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch

from scripts.dtypes import InstanceOutputs, PipelineOutput
from scripts.fusion_stub import FusionStub
from scripts.prediction_head_stub import PredictionHeadStub
from scripts.physics_head_stub import PhysicsHeadStub


class Pipeline:
    """
    Inference pipeline that wires:
      - backbone features (multi-scale)
      - segmentation instances (masks/boxes/scores)
      - depth map
      - per-instance pooling
      - fusion module -> (F_i, f_i, v_i)
      - prediction head -> food class + portion
      - physics head -> area/volume/calories

    This class is designed so you can replace FusionStub / PredictionHeadStub / PhysicsHeadStub
    later without changing anything else.
    """

    def __init__(
        self,
        backbone_block,
        seg_block,
        depth_block,
        device: str = "cuda",
        seg_score_thresh: float = 0.5,
        fusion: Optional[object] = None,
        prediction_head: Optional[object] = None,
        physics_head: Optional[object] = None,
    ):
        self.backbone_block = backbone_block
        self.seg_block = seg_block
        self.depth_block = depth_block
        self.device = device
        self.seg_score_thresh = seg_score_thresh

        # ---------- Fusion module (replace later) ----------
        if fusion is None:
            fusion = FusionStub()
        self.fusion = fusion

        # ---------- Prediction head (stub) ----------
        if prediction_head is None:
            prediction_head = PredictionHeadStub(num_food_classes=101)
        self.prediction_head = prediction_head

        # ---------- Physics head (stub) ----------
        if physics_head is None:
            physics_head = PhysicsHeadStub(default_kcal_per_volume=1.0)
        self.physics_head = physics_head

    # ---------- Utilities ----------

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

    # ---------- Main pipeline runner ----------

    @torch.inference_mode()
    def run(self, img_rgb_uint8: np.ndarray) -> PipelineOutput:
        """Run the full pipeline on a single RGB uint8 image."""

        # 1) Backbone features (multi-scale)
        back_out = self.backbone_block(img_rgb_uint8)
        feats = back_out.features  # list of [1,C,Hf,Wf]

        # 2) Segmentation instances
        seg_out = self.seg_block(img_rgb_uint8, score_thresh=self.seg_score_thresh)
        boxes = seg_out.boxes_xyxy
        seg_class_ids = seg_out.class_ids
        scrs = seg_out.scores
        masks = seg_out.masks

        # 3) Depth map
        depth_hw = self.depth_block(img_rgb_uint8).depth  # [H,W] float32

        # 4) Per-instance pooling -> 5) fusion -> 6) prediction head -> 7) physics head
        instance_outputs: List[InstanceOutputs] = []
        for i in range(masks.shape[0]):
            m = masks[i]
            area_px = int(m.sum())

            # F_i input: multi-scale masked pooled vectors
            pooled_feats = [self._mask_pool_feature(f, m) for f in feats]  # list of [C_i] on CPU

            # v_i input: compact depth stats
            d_med, d_mean = self._mask_pool_depth(depth_hw, m)

            # Fusion output: (F_i, f_i, v_i)
            fusion_out = self.fusion(pooled_feats, d_med, d_mean)

            # Prediction head: (F_i, f_i, v_i) -> (food logits, portion)
            pred = self.prediction_head(fusion_out)

            # Physics head: (mask, depth stats, predicted class) -> (area, volume, calories)
            phys = self.physics_head(mask_hw=m, depth_median=d_med, prediction=pred)

            instance_outputs.append(
                InstanceOutputs(
                    mask=m,
                    box_xyxy=boxes[i],
                    score=float(scrs[i]),
                    seg_class_id=int(seg_class_ids[i]),
                    fusion=fusion_out,
                    prediction=pred,
                    physics=phys,
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
