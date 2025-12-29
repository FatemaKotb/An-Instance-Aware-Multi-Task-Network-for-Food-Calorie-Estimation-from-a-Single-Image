from __future__ import annotations

from typing import List
import torch

from scripts.dtypes import FusionOutput


class FusionStub:
    def __init__(self):
        pass

    def __call__(
        self,
        pooled_feats: List[torch.Tensor],
        depth_median: float,
        depth_mean: float,
        **_: object,
    ) -> FusionOutput:
        masked_features = [p.detach().cpu().float() for p in pooled_feats]
        global_features = torch.cat([p.float() for p in masked_features], dim=0)
        instance_depth = torch.tensor([depth_median, depth_mean], dtype=torch.float32)

        return FusionOutput(
            masked_features=masked_features,
            global_features=global_features,
            instance_depth=instance_depth,
        )
