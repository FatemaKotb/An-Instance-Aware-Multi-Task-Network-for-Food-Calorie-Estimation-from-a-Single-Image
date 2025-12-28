from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class CLIPBundle:
    model: torch.nn.Module
    preprocess: Any
    tokenizer: Any

def load_openclip(model_name: str, pretrained: str, device: str) -> CLIPBundle:
    try:
        import open_clip
    except Exception as e:
        raise ImportError("open_clip_torch is required. Install with: pip install open_clip_torch") from e

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return CLIPBundle(model=model, preprocess=preprocess, tokenizer=tokenizer)

@torch.inference_mode()
def encode_images(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats
