from __future__ import annotations

from typing import List

import torch

from scripts.dtypes import FusionOutput


# ---------- Stub fusion module (replace later) ----------

class FusionStub:
    """
    Drop-in placeholder for Instance-Aware Fusion.

    In the paper diagram, the fusion module produces three outputs:
      - Masked features (F_i): per-instance masked/pool features per scale
      - Global pooled features (f_i): a single descriptor vector for the instance
      - Instance depth (v_i): compact depth descriptor (e.g., median/mean)

    This stub keeps those three signals separate so downstream heads can consume them
    with the expected input/output structure.

    Replace later with your learnable fusion network.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        pooled_feats: List[torch.Tensor],
        depth_median: float,
        depth_mean: float,
    ) -> FusionOutput:
        # F_i: masked features per scale (already pooled by the pipeline)
        masked_features = [p.detach().cpu().float() for p in pooled_feats]

        # f_i: global pooled descriptor (simple concat of all scales)
        global_features = torch.cat([p.float() for p in masked_features], dim=0)  # [sum(C_i)]

        # v_i: instance depth descriptor (2 stats)
        instance_depth = torch.tensor([depth_median, depth_mean], dtype=torch.float32)

        return FusionOutput(
            masked_features=masked_features,
            global_features=global_features,
            instance_depth=instance_depth,
        )
