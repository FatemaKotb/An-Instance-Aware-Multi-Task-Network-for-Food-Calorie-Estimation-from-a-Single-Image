from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch



# ------------------- lvl 1 -------------------
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

# ------------------- lvl 2 -------------------
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

# ------------------- lvl 3 -------------------
@dataclass
class PredictionOutput:
    """Outputs of the Prediction Head.

    Contract (always present, even for stubs):
      - food_class_id
      - food_class_name
      - food_conf
      - top5_foods
      - portion

    Notes:
      - food_logits is also included for debugging/analysis.
        For stubs, you can return torch.empty(0) or a uniform vector.
    """
    food_logits: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [K] on CPU (or empty)
    food_class_id: int = -1
    food_class_name: str = "unknown"
    food_conf: float = 0.0
    portion: float = 1.0

    # [(name, prob), ...] always a list (never None)
    top5_foods: List[Tuple[str, float]] = field(default_factory=list)

# ------------------- lvl 4 -------------------
@dataclass
class PhysicsOutput:
    """Outputs of the Physics-Based Calorie Estimation block.

    Contract (always present, even for stubs):
      - area_px
      - volume
      - calories
      - kcal_per_volume_used (traceability)
    """
    area_px: int = 0
    volume: float = float("nan")
    calories: float = float("nan")
    kcal_per_volume_used: float = 1.0

# ------------------- Full pipeline outputs -------------------
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

# ------------------- Configs -------------------
@dataclass
class DeviceConfig:
    device: str = "cuda"  # or "cpu"
