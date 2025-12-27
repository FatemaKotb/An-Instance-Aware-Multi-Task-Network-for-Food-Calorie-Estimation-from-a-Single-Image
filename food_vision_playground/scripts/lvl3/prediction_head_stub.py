from __future__ import annotations

from typing import Optional

import torch

from scripts.dtypes import FusionOutput, PredictionOutput


class PredictionHeadStub:
    """
    Prediction Head (stub): Classification & Portion Estimation.

    Expected I/O (matches the diagram):
      Inputs:
        - FusionOutput.global_features  -> f_i (semantic instance descriptor)
        - FusionOutput.instance_depth   -> v_i (depth descriptor)
        - (optionally) FusionOutput.masked_features -> F_i

      Outputs:
        - food_logits: [K] (or food_class_id + confidence)
        - portion: float (or portion bins)

    This stub returns:
      - zeros logits ("unknown")
      - NaN portion

    Replace later with a learned head.
    """

    def __init__(self, num_food_classes: int = 101):
        self.num_food_classes = int(num_food_classes)

    @torch.inference_mode()
    def __call__(self, fusion: FusionOutput) -> PredictionOutput:
        food_logits = torch.zeros(self.num_food_classes, dtype=torch.float32)

        # With no trained classifier, we can't pick a meaningful food id.
        food_class_id: Optional[int] = None
        food_conf: Optional[float] = None

        # Portion is also undefined until you decide a target representation.
        portion: float = float('nan')

        return PredictionOutput(
            food_logits=food_logits,
            food_class_id=food_class_id,
            food_conf=food_conf,
            portion=portion,
        )
