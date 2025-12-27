from __future__ import annotations
from typing import Dict, Optional
import numpy as np


# ---------- Physics block (simple placeholder) ----------

class PhysicsCalorieEstimator:
    """
    Minimal physics estimator:
      area_px -> area_norm (optional)
      thickness proxy -> median depth (relative)
      calories = area_px * depth_median * density[class_id]

    For now, density table can be dummy (e.g., all ones) until you have food classes.
    """
    def __init__(self, density_by_class: Optional[Dict[int, float]] = None, default_density: float = 1.0):
        self.density_by_class = density_by_class or {}
        self.default_density = float(default_density)

    def estimate(self, area_px: int, depth_median: float, class_id: int) -> float:
        rho = self.density_by_class.get(int(class_id), self.default_density)
        if np.isnan(depth_median):
            return float("nan")
        return float(area_px) * float(depth_median) * float(rho)
