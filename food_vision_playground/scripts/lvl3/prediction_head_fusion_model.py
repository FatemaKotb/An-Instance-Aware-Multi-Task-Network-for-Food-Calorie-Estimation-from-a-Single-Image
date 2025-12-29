from __future__ import annotations

from typing import List, Tuple, Optional
import torch

from scripts.dtypes import PredictionOutput, FusionOutput
from scripts.lvl2.instance_aware_fusion_head import InstanceAwareFusionHead


class PredictionHeadFusionModel:
    """Uses a trained InstanceAwareFusionHead checkpoint for classification.

    Requires FusionOutput.roi_feat/mask_roi/depth_roi/depth_stats (from InstanceAwareFusionBlock).
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.in_channels = int(ckpt["in_channels"])
        self.num_classes = int(ckpt["num_classes"])
        self.subset_meta = ckpt.get("subset_meta", {})
        self.class_names: Optional[List[str]] = self.subset_meta.get("chosen_class_names")

        self.model = InstanceAwareFusionHead(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, *, img_rgb_uint8, box_xyxy, mask_hw, fusion: FusionOutput) -> PredictionOutput:
        if fusion.roi_feat is None or fusion.mask_roi is None or fusion.depth_roi is None or fusion.depth_stats is None:
            raise RuntimeError(
                "FusionOutput is missing ROI tensors. "
                "Use InstanceAwareFusionBlock when running PredictionHeadFusionModel."
            )

        roi_feat = fusion.roi_feat.unsqueeze(0).to(self.device, non_blocking=True).to(torch.float32)        # [1,C,7,7]
        mask_roi = fusion.mask_roi.unsqueeze(0).to(self.device, non_blocking=True).to(torch.float32)        # [1,1,7,7]
        depth_roi = fusion.depth_roi.unsqueeze(0).to(self.device, non_blocking=True).to(torch.float32)      # [1,1,7,7]
        depth_stats = fusion.depth_stats.unsqueeze(0).to(self.device, non_blocking=True).to(torch.float32)  # [1,8]

        logits = self.model(roi_feat, mask_roi, depth_roi, depth_stats).squeeze(0)  # [K]
        probs = torch.softmax(logits, dim=0)

        topk = min(5, probs.numel())
        top_probs, top_idx = torch.topk(probs, k=topk, dim=0)

        top5_foods: List[Tuple[str, float]] = []
        for i in range(topk):
            cls_id = int(top_idx[i].item())
            name = self.class_names[cls_id] if self.class_names and cls_id < len(self.class_names) else str(cls_id)
            top5_foods.append((name, float(top_probs[i].item())))

        pred_id = int(top_idx[0].item())
        pred_name = self.class_names[pred_id] if self.class_names and pred_id < len(self.class_names) else str(pred_id)
        pred_conf = float(top_probs[0].item())

        return PredictionOutput(
            food_logits=logits.detach().cpu(),
            food_class_id=pred_id,
            food_class_name=pred_name,
            food_conf=pred_conf,
            top5_foods=top5_foods,
            portion=1.0,
        )
