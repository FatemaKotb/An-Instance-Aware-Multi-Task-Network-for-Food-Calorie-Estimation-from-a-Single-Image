from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch

from scripts.dtypes import InstanceOutputs, PipelineOutput


class Pipeline:
    def __init__(
        self,
        backbone_block,
        seg_block,
        depth_block,
        fusion_block,
        prediction_head,
        physics_head,
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

    def _mask_pool_feature(self, feat: torch.Tensor, mask_hw: np.ndarray) -> torch.Tensor:
        assert feat.ndim == 4 and feat.shape[0] == 1
        mask = torch.from_numpy(mask_hw.astype(np.float32))[None, None]  # [1,1,H,W]
        mask = torch.nn.functional.interpolate(mask, size=feat.shape[-2:], mode="nearest")
        mask = mask.to(feat.device)
        denom = mask.sum().clamp(min=1.0)
        pooled = (feat * mask).sum(dim=(2, 3)).squeeze(0) / denom.squeeze()
        return pooled.detach().cpu()

    def _mask_pool_depth(self, depth_hw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float]:
        vals = depth_hw[mask_hw]
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.median(vals)), float(np.mean(vals))

    @torch.inference_mode()
    def run(self, img_rgb_uint8: np.ndarray) -> PipelineOutput:
        back_out = self.backbone_block(img_rgb_uint8)
        feats = back_out.features

        seg_out = self.seg_block(img_rgb_uint8, score_thresh=self.seg_score_thresh)
        boxes = seg_out.boxes_xyxy
        seg_class_ids = seg_out.class_ids
        scrs = seg_out.scores
        masks = seg_out.masks

        depth_hw = self.depth_block(img_rgb_uint8).depth

        instance_outputs: List[InstanceOutputs] = []
        for i in range(masks.shape[0]):
            m = masks[i]
            area_px = int(m.sum())

            pooled_feats = [self._mask_pool_feature(f, m) for f in feats]
            d_med, d_mean = self._mask_pool_depth(depth_hw, m)

            # Fusion block may optionally need full context (ROI tensors, etc.).
            # FusionStub accepts and ignores these extras.
            fusion_out = self.fusion_block(
                pooled_feats,
                d_med,
                d_mean,
                img_rgb_uint8=img_rgb_uint8,
                box_xyxy=boxes[i],
                mask_hw=m,
                depth_hw=depth_hw,
                backbone_last_feat=feats[-1],
            )

            pred = self.prediction_head(img_rgb_uint8=img_rgb_uint8, box_xyxy=boxes[i], mask_hw=m, fusion=fusion_out)
            phys = self.physics_head(mask_hw=m, depth_median=d_med, prediction=pred)

            instance_outputs.append(
                InstanceOutputs(
                    mask=m,
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
