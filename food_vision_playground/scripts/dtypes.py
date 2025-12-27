from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

Mode = Literal["cls", "backbone"]


@dataclass
class ClassificationResult:
    topk_indices: List[int]
    topk_probs: List[float]
    logits: Optional[torch.Tensor] = None  # [B, num_classes] if return_logits=True


@dataclass
class BackboneOutput:
    # Multi-scale feature maps: each tensor is [B, C_i, H_i, W_i]
    features: List[torch.Tensor]


@dataclass
class InstanceSegmentationResult:
    boxes_xyxy: np.ndarray     # [N,4]
    scores: np.ndarray         # [N]
    class_ids: np.ndarray      # [N]
    masks: np.ndarray          # [N,H,W] bool


@dataclass
class DepthResult:
    depth: np.ndarray  # [H,W] float32


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


@dataclass
class DeviceConfig:
    device: str = "cuda"  # or "cpu"
