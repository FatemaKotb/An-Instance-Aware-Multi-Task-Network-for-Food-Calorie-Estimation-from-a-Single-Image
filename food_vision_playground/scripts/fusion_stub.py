from __future__ import annotations
from typing import List
import torch


# ---------- Stub fusion module (replace later) ----------

class FusionStub:
    """
    Drop-in placeholder for Instance-Aware Fusion.
    Currently: concatenates pooled features + 2 depth stats into one embedding.
    Replace later with your learnable fusion network.
    """
    def __init__(self):
        pass

    def __call__(self, pooled_feats: List[torch.Tensor], depth_median: float, depth_mean: float) -> torch.Tensor:
        # concat all pooled feature scales into a single vector
        f = torch.cat([p.float() for p in pooled_feats], dim=0)  # [sum(C_i)]
        d = torch.tensor([depth_median, depth_mean], dtype=torch.float32)
        return torch.cat([f, d], dim=0)  # [D]
