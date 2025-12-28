from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from scripts.dtypes import PhysicsOutput, PredictionOutput


class PhysicsHeadStub:
    """
    Physics-Based Calorie Estimation (stub).

    Expected I/O (matches the diagram):
      Inputs:
        - mask -> Area (A_i)
        - depth stats -> Volume proxy (V_i)
        - predicted food class -> density / kcal per volume lookup

      Outputs:
        - area_px (A_i proxy)
        - volume (V_i proxy)
        - calories (C_i proxy)

    This stub uses:
      A_i = sum(mask)
      V_i = A_i * depth_median
      C_i = V_i * kcal_per_volume

    Replace later with calibrated geometry + proper food densities.
    """

    def __init__(
        self,
        kcal_per_volume_by_class: Optional[Dict[int, float]] = None,
        default_kcal_per_volume: float = 1.0,
    ):
        self.kcal_per_volume_by_class = kcal_per_volume_by_class or {}
        self.default_kcal_per_volume = float(default_kcal_per_volume)

    def __call__(
        self,
        mask_hw: np.ndarray,
        depth_median: float,
        prediction: PredictionOutput,
    ) -> PhysicsOutput:
        area_px = int(mask_hw.astype(bool).sum())

        if np.isnan(depth_median):
            volume = float("nan")
        else:
            volume = float(area_px) * float(depth_median)

        # With locked contract, stub uses -1 for "unknown"
        class_id = int(prediction.food_class_id)
        kcal_per_volume = self.kcal_per_volume_by_class.get(class_id, self.default_kcal_per_volume)

        calories = float("nan") if np.isnan(volume) else float(volume) * float(kcal_per_volume)

        return PhysicsOutput(
            area_px=area_px,
            volume=volume,
            calories=calories,
            kcal_per_volume_used=float(kcal_per_volume),
        )
