from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from scripts.dtypes import PhysicsOutput, PredictionOutput

# Minimal demo tables (very rough typical values)
# density: g / cm^3
FOOD_DENSITY = {
    "pizza": 0.6,
    "salad": 0.3,
    "burger": 0.7,
    "rice": 0.8,
}

# energy: kcal / g
FOOD_KCAL_PER_G = {
    "pizza": 2.66,
    "salad": 0.15,
    "burger": 2.95,
    "rice": 1.30,
}


class PhysicsHeadStub:
    """
    Physics-Based Calorie Estimation (stub).

    Expected I/O (matches the diagram):
      Inputs:
        - mask -> Area (A_i) proxy
        - depth stats -> Volume proxy (V_i)
        - predicted food label -> kcal per volume lookup (derived)

      Outputs:
        - area_px (A_i proxy)
        - volume (V_i proxy)
        - calories (C_i proxy)

    This stub uses:
      A_i = sum(mask)
      V_i = A_i * depth_median
      kcal_per_volume(food) = (kcal_per_g(food) * density(food))
      C_i = V_i * kcal_per_volume(food)

    Replace later with calibrated geometry + proper food densities.
    """

    def __init__(
        self,
        density_by_name: Optional[Dict[str, float]] = None,
        kcal_per_g_by_name: Optional[Dict[str, float]] = None,
        default_density: float = 1.0,
        default_kcal_per_g: float = 1.0,
    ):
        self.density_by_name = density_by_name or dict(FOOD_DENSITY)
        self.kcal_per_g_by_name = kcal_per_g_by_name or dict(FOOD_KCAL_PER_G)
        self.default_density = float(default_density)
        self.default_kcal_per_g = float(default_kcal_per_g)

    def __call__(
        self,
        mask_hw: np.ndarray,
        depth_median: float,
        prediction: PredictionOutput,
    ) -> PhysicsOutput:
        area_px = int(mask_hw.astype(bool).sum())

        # Volume proxy (still not metric): pixels * relative depth
        if np.isnan(depth_median):
            volume = float("nan")
        else:
            volume = float(area_px) * float(depth_median)

        # Use predicted class name if available; otherwise fall back to defaults
        name = (prediction.food_class_name or "unknown").strip().lower()

        density = float(self.density_by_name.get(name, self.default_density))
        kcal_per_g = float(self.kcal_per_g_by_name.get(name, self.default_kcal_per_g))

        # Derived kcal per "volume unit" (here: per cm^3 in theory, but volume is a proxy in this pipeline)
        kcal_per_volume = kcal_per_g * density

        calories = float("nan") if np.isnan(volume) else float(volume) * float(kcal_per_volume)

        return PhysicsOutput(
            area_px=area_px,
            volume=volume,
            calories=calories,
            kcal_per_volume_used=float(kcal_per_volume),
        )