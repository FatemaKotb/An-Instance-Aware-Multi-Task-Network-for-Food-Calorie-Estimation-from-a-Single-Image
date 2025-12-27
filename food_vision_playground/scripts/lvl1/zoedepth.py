from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from scripts.dtypes import DepthResult


# ZoeDepth NK checkpoint
_ZOED_NK_URL = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"


class ZoeDepthBlock:
    """
    ZoeDepth wrapped as a block.

    Uses torch.hub to fetch the repo and checkpoint (the repo is not a pip project).

    Notes:
      - We load with strict=False and filter out non-learnable keys that can fail on some Windows/PyTorch combos.
      - Output depth is typically relative/affine, not guaranteed metric without calibration.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self.load_zoedepth(device=device)

    def _ensure_zoedepth_importable_from_torchhub(self) -> None:
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

    def load_zoedepth(self, device: str = "cuda"):
        """
        Load ZoeDepth NK without pip installing the repo.
        Avoids strict state_dict loading that fails on some torch/Windows combos.
        """
        self._ensure_zoedepth_importable_from_torchhub()

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
    def predict(self, img_rgb_uint8: np.ndarray) -> DepthResult:
        """Predict a dense depth map for an RGB uint8 image."""
        from PIL import Image
        import torchvision.transforms as T

        img = Image.fromarray(img_rgb_uint8)
        x = T.ToTensor()(img).unsqueeze(0).to(self.device)

        out = self.model.infer(x)
        depth = out["depth"] if isinstance(out, dict) else out
        if depth.ndim == 4:
            depth = depth[0, 0]
        return DepthResult(depth.detach().float().cpu().numpy().astype(np.float32))

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray) -> DepthResult:
        return self.predict(img_rgb_uint8)

    def save_depth_assets(self, out_dir: Path, depth_hw: np.ndarray) -> None:
        """Save raw depth (npy) and a normalized PNG for quick visualization."""
        from PIL import Image

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # save raw
        np.save(out_dir / "depth.npy", depth_hw.astype(np.float32))

        # save viewable PNG (normalize ignoring NaNs)
        d = depth_hw.astype(np.float32)
        finite = np.isfinite(d)
        if finite.any():
            dmin = float(d[finite].min())
            dmax = float(d[finite].max())
            denom = (dmax - dmin) if (dmax > dmin) else 1.0
            vis = (255.0 * (d - dmin) / denom)
            vis[~finite] = 0.0
        else:
            vis = np.zeros_like(d, dtype=np.float32)

        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
        Image.fromarray(vis_u8).save(out_dir / "depth.png")
