from __future__ import annotations

from typing import Optional

from scripts.dtypes import FusionOutput, PredictionOutput


class PredictionHeadHybridCLIP:
    """
    Hybrid prediction head:

    - Always runs BOTH:
        (A) Food-101 / pretrained CLIP head (general)
        (B) Fine-tuned Egyptian linear head (specialist)

    - Decision rule:
        Return Food-101 prediction UNLESS Egyptian confidence >= egypt_conf_thresh,
        in which case return Egyptian prediction.

    - Traceability:
        We attach extra attributes to the returned PredictionOutput:
          - pred.food101_top5_foods
          - pred.food101_food_class_name
          - pred.food101_food_conf
          - pred.egypt_top5_foods
          - pred.egypt_food_class_name
          - pred.egypt_food_conf
          - pred.hybrid_selected ("food101" or "egypt")

    This avoids changing scripts/dtypes.py while still allowing logging.
    """

    def __init__(
        self,
        food101_head,
        egypt_head,
        egypt_conf_thresh: float = 0.35,
    ):
        self.food101_head = food101_head
        self.egypt_head = egypt_head
        self.egypt_conf_thresh = float(egypt_conf_thresh)

    def __call__(
        self,
        img_rgb_uint8,
        box_xyxy,
        mask_hw,
        fusion: Optional[FusionOutput] = None,
    ) -> PredictionOutput:
        # Run both heads
        pred_food101: PredictionOutput = self.food101_head(
            img_rgb_uint8=img_rgb_uint8,
            box_xyxy=box_xyxy,
            mask_hw=mask_hw,
            fusion=fusion,
        )

        pred_egypt: PredictionOutput = self.egypt_head(
            img_rgb_uint8=img_rgb_uint8,
            box_xyxy=box_xyxy,
            mask_hw=mask_hw,
            fusion=fusion,
        )

        egypt_conf = float(getattr(pred_egypt, "food_conf", 0.0) or 0.0)

        # Choose output
        if egypt_conf >= self.egypt_conf_thresh:
            chosen = pred_egypt
            chosen.hybrid_selected = "egypt"
        else:
            chosen = pred_food101
            chosen.hybrid_selected = "food101"

        # Attach “other head” predictions for logging/analysis
        # (Dataclasses here are not slots, so dynamic attrs are fine.)
        chosen.food101_top5_foods = getattr(pred_food101, "top5_foods", [])
        chosen.food101_food_class_name = getattr(pred_food101, "food_class_name", "unknown")
        chosen.food101_food_conf = float(getattr(pred_food101, "food_conf", 0.0) or 0.0)

        chosen.egypt_top5_foods = getattr(pred_egypt, "top5_foods", [])
        chosen.egypt_food_class_name = getattr(pred_egypt, "food_class_name", "unknown")
        chosen.egypt_food_conf = float(getattr(pred_egypt, "food_conf", 0.0) or 0.0)

        chosen.egypt_conf_thresh = self.egypt_conf_thresh

        return chosen
