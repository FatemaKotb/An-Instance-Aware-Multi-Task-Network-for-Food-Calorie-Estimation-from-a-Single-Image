from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch


@dataclass
class InstanceInputs:
    """What your future fusion module will consume."""
    mask: np.ndarray                 # [H,W] bool
    box_xyxy: np.ndarray             # [4]
    score: float
    class_id: int

    # pooled backbone semantics (per scale)
    pooled_feats: List[torch.Tensor] # each [C_i] on CPU

    # pooled depth descriptor(s)
    depth_median: float
    depth_mean: float

@dataclass
class InstanceOutputs:
    """What downstream heads / physics will consume."""
    mask: np.ndarray                 # [H,W] bool
    box_xyxy: np.ndarray             # [4]
    score: float
    class_id: int

    # embedding after (stub) fusion: [D] torch tensor on CPU
    embedding: torch.Tensor

    # geometry proxies
    area_px: int
    depth_median: float
    depth_mean: float

@dataclass
class PipelineOutput:
    instance_outputs: List[InstanceOutputs]
    depth_map: np.ndarray            # [H,W] float32
    backbone_features: List[torch.Tensor]  # raw feature maps [1,C,H,W] on GPU/CPU
