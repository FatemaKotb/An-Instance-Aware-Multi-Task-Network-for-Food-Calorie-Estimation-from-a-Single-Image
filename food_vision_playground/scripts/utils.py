from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image

def load_rgb(path: str) -> np.ndarray:
    """Load an image from disk as RGB uint8 numpy array [H,W,3]."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure input is uint8 RGB [H,W,3]."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape={img.shape}")
    return img

@dataclass
class DeviceConfig:
    device: str = "cuda"  # or "cpu"
