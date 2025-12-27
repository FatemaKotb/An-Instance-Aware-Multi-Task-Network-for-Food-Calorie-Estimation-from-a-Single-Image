from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from pathlib import Path
import numpy as np
import torch

@dataclass
class DepthResult:
    depth: np.ndarray  # [H,W] float32

# ZoeDepth NK checkpoint
_ZOED_NK_URL = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"

def _ensure_zoedepth_importable_from_torchhub() -> None:
    """
    Make `import zoedepth` work by adding the torch.hub cached repo to sys.path.
    This avoids pip installation (repo isn't a pip project).
    """
    hub_dir = Path(torch.hub.get_dir())
    # typical: ~/.cache/torch/hub/isl-org_ZoeDepth_main
    candidates = list(hub_dir.glob("isl-org_ZoeDepth_*"))
    if not candidates:
        # Force-download the repo (no weights) so the folder exists
        torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=False)

        candidates = list(hub_dir.glob("isl-org_ZoeDepth_*"))
        if not candidates:
            raise RuntimeError(f"Could not find ZoeDepth repo in torch hub dir: {hub_dir}")

    repo_path = str(candidates[0])
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

def load_zoedepth(device: str = "cuda"):
    """
    Load ZoeDepth NK without pip installing the repo.
    Avoids strict state_dict loading that fails on some torch/Windows combos.
    """
    _ensure_zoedepth_importable_from_torchhub()

    from zoedepth.utils.config import get_config
    from zoedepth.models.builder import build_model

    # Build model without auto-loading pretrained weights
    config = get_config("zoedepth_nk", "infer", pretrained_resource=None)
    model = build_model(config)

    # Download checkpoint
    ckpt = torch.hub.load_state_dict_from_url(_ZOED_NK_URL, map_location="cpu", progress=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Filter keys that cause the crash (non-learnable buffers)
    sd = {k: v for k, v in sd.items() if "relative_position_index" not in k}

    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model

@torch.inference_mode()
def predict_depth(model, img_rgb_uint8: np.ndarray, device: str = "cuda") -> DepthResult:
    from PIL import Image
    import torchvision.transforms as T

    img = Image.fromarray(img_rgb_uint8)
    x = T.ToTensor()(img).unsqueeze(0).to(device)

    out = model.infer(x)
    depth = out["depth"] if isinstance(out, dict) else out
    if depth.ndim == 4:
        depth = depth[0, 0]
    return DepthResult(depth.detach().float().cpu().numpy().astype(np.float32))
