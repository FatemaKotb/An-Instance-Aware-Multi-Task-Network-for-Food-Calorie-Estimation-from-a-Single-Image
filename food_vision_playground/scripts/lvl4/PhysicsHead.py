from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from scripts.dtypes import PhysicsOutput, PredictionOutput


# ============================================================
# FOOD PHYSICAL PROPERTIES (Level 4)
# ------------------------------------------------------------
# Units:
#   density          -> g / cm^3
#   energy           -> kcal / g
#
# NOTE:
# These are approximate literature / USDA-style averages.
# They are proxies and MUST be calibrated later.
# ============================================================

FOOD_DENSITY: Dict[str, float] = {
    # Core foods
    "pizza": 0.60,
    "salad": 0.30,
    "burger": 0.70,
    "rice": 0.80,
    "pasta": 0.75,
    "bread": 0.55,
    "chicken": 1.05,
    "steak": 1.10,
    "fries": 0.65,

    # Generic fallbacks for dataset classes
    "noodles": 0.78,
    "soup": 0.45,
    "fish": 1.00,
    "egg": 1.03,
    "cheese": 1.10,
    "sandwich": 0.60,
    "cake": 0.50,
    "dessert": 0.55,
    "fruit": 0.65,
    "vegetable": 0.40,

    # Catch-all
    "unknown": 1.00,
}

FOOD_KCAL_PER_G: Dict[str, float] = {
    # Core foods
    "pizza": 2.66,
    "salad": 0.15,
    "burger": 2.95,
    "rice": 1.30,
    "pasta": 1.31,
    "bread": 2.50,
    "chicken": 2.39,
    "steak": 2.71,
    "fries": 3.12,

    # Generic fallbacks
    "noodles": 1.38,
    "soup": 0.40,
    "fish": 2.06,
    "egg": 1.55,
    "cheese": 4.02,
    "sandwich": 2.50,
    "cake": 3.80,
    "dessert": 3.50,
    "fruit": 0.52,
    "vegetable": 0.25,

    # Catch-all
    "unknown": 1.00,
}


# ============================================================
# PHYSICS-BASED CALORIE ESTIMATION HEAD (LEVEL 4)
# ============================================================

class PhysicsHead:
    """
    Level-4: Physics-Based Calorie Estimation Head

    Inputs:
      - instance mask (H x W)
      - depth median for instance
      - semantic prediction (class name)

    Outputs:
      - area_px: pixel area of food instance
      - volume: relative volume proxy
      - calories: estimated kcal
      - kcal_per_volume_used

    Notes:
      * Volume is NOT metric yet
      * Calibration will come later
      * This is instance-aware by design
    """

    def __init__(
        self,
        density_by_name: Optional[Dict[str, float]] = None,
        kcal_per_g_by_name: Optional[Dict[str, float]] = None,
        default_density: float = 1.0,
        default_kcal_per_g: float = 1.0,
    ):
        self.density_by_name = density_by_name or FOOD_DENSITY
        self.kcal_per_g_by_name = kcal_per_g_by_name or FOOD_KCAL_PER_G
        self.default_density = float(default_density)
        self.default_kcal_per_g = float(default_kcal_per_g)

    # --------------------------------------------------------
    # Forward call
    # --------------------------------------------------------
    def __call__(
        self,
        mask_hw: np.ndarray,
        depth_median: float,
        prediction: PredictionOutput,
    ) -> PhysicsOutput:

        # 1. Area proxy (pixel count)
        area_px = int(np.count_nonzero(mask_hw))

        # 2. Volume proxy
        if area_px == 0 or np.isnan(depth_median):
            volume = float("nan")
        else:
            volume = float(area_px) * float(depth_median)

        # 3. Resolve food name
        food_name = (
            prediction.food_class_name.lower().strip()
            if prediction.food_class_name
            else "unknown"
        )

        # 4. Lookup physical parameters
        density = float(
            self.density_by_name.get(food_name, self.default_density)
        )
        kcal_per_g = float(
            self.kcal_per_g_by_name.get(food_name, self.default_kcal_per_g)
        )

        # 5. Energy per volume unit
        kcal_per_volume = kcal_per_g * density

        # 6. Calories
        calories = (
            float("nan")
            if np.isnan(volume)
            else volume * kcal_per_volume
        )

        return PhysicsOutput(
            area_px=area_px,
            volume=volume,
            calories=calories,
            kcal_per_volume_used=kcal_per_volume,
        )
