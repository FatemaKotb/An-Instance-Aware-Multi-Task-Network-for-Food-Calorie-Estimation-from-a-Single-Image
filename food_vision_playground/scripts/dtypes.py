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


# ---------- Fusion outputs (match the diagram) ----------

@dataclass
class FusionOutput:
    """Outputs of the Instance-Aware Fusion Module.

    - masked_features (F_i): pooled semantics per scale (list of [C_s])
    - global_features (f_i): a single global descriptor ([D])
    - instance_depth (v_i): depth descriptor (e.g., [depth_median, depth_mean])
    """
    masked_features: List[torch.Tensor]  # each [C_s] on CPU
    global_features: torch.Tensor        # [D] on CPU
    instance_depth: torch.Tensor         # [Dv] on CPU (typically [2])


# ---------- Prediction head outputs (green block) ----------

@dataclass
class PredictionOutput:
    """Outputs of the Prediction Head.

    - food_logits: [K] logits for K food classes
    - food_class_id: argmax id (optional convenience)
    - food_conf: max softmax probability (optional convenience)
    - portion: scalar proxy (stub) or a learned portion value
    """
    food_logits: torch.Tensor            # [K] on CPU
    food_class_id: int
    food_class_name: str
    food_conf: float
    portion: float
    top5_foods: List[Tuple[str, float]] = None  # [(name, prob), ...]


# ---------- Physics head outputs (green block) ----------

@dataclass
class PhysicsOutput:
    """Outputs of the Physics-Based Calorie Estimation block."""
    area_px: float
    volume: float
    calories: float


# ---------- Data containers ----------

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

    # segmentation label (from the segmenter, e.g., COCO)
    seg_class_id: int

    # fusion outputs
    fusion: FusionOutput

    # prediction head outputs (food + portion)
    prediction: PredictionOutput

    # physics head outputs (area/volume/calories)
    physics: PhysicsOutput

    # geometry proxies (kept for convenience)
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
