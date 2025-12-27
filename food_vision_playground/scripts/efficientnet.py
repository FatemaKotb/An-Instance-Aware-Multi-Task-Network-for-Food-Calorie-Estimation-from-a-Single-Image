from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import timm
from PIL import Image

# timm==0.6.x API (works with your pinned timm)
from timm.data import resolve_data_config, create_transform


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


def load_efficientnet(
    model_name: str = "tf_efficientnetv2_s",
    device: str = "cuda",
    mode: Mode = "cls",
    out_indices: Tuple[int, ...] = (1, 2, 3, 4),
):
    """
    Unified EfficientNet loader.

    mode="cls":
      - Loads classification model (with head)
      - Forward returns logits [B, num_classes]

    mode="backbone":
      - Loads features-only backbone (no classifier head)
      - Forward returns list of feature maps (multi-scale), controlled by out_indices

    Returns:
      model, transform
    """
    if mode == "cls":
        model = timm.create_model(model_name, pretrained=True)
    elif mode == "backbone":
        model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}. Expected 'cls' or 'backbone'.")

    model = model.to(device).eval()

    # timm<=0.6 uses resolve_data_config/create_transform
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)

    return model, transform


@torch.inference_mode()
def run_efficientnet(
    model,
    transform,
    img_rgb_uint8: np.ndarray,
    device: str = "cuda",
    mode: Mode = "cls",
    topk: int = 5,
    return_logits: bool = False,
) -> Union[ClassificationResult, BackboneOutput]:
    """
    Run EfficientNet in either classification or backbone mode.

    Input:
      img_rgb_uint8: np.ndarray [H,W,3] uint8 RGB

    Output:
      - mode="cls" -> ClassificationResult
      - mode="backbone" -> BackboneOutput
    """
    img = Image.fromarray(img_rgb_uint8)
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    if mode == "cls":
        logits = model(x)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0]
        vals, idx = torch.topk(probs, k=topk)

        return ClassificationResult(
            topk_indices=idx.detach().cpu().tolist(),
            topk_probs=vals.detach().cpu().tolist(),
            logits=logits.detach().cpu() if return_logits else None,
        )

    if mode == "backbone":
        feats = model(x)  # list[Tensor], each [1,C,H,W]
        return BackboneOutput(features=feats)

    raise ValueError(f"Unknown mode={mode!r}. Expected 'cls' or 'backbone'.")


def describe_features(features: List[torch.Tensor]) -> None:
    """Print feature shapes for sanity checking."""
    for i, f in enumerate(features):
        print(f"F{i}: shape={tuple(f.shape)} dtype={f.dtype} device={f.device}")


def load_imagenet_labels():
    """
    Lightweight ImageNet label loading.
    If you prefer, replace with your own mapping.
    """
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        labels = f.read().decode("utf-8").splitlines()
    return labels
