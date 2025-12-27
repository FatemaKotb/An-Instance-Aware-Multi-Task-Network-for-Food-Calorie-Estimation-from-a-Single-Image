from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from scripts.dtypes import DepthResult


class ZoeDepthBlock:
    """ZoeDepth wrapper as a block for clean pipeline instantiation."""

    # ZoeDepth NK checkpoint
    _ZOED_NK_URL = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = self._load_zoedepth(device=device)

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

    def _load_zoedepth(self, device: str = "cuda"):
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
        ckpt = torch.hub.load_state_dict_from_url(self._ZOED_NK_URL, map_location="cpu", progress=True)
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
    def predict_depth(self, img_rgb_uint8: np.ndarray) -> DepthResult:
        """
        Predict depth from RGB uint8 image.
        Output is float32 [H,W].
        """
        from PIL import Image
        import torchvision.transforms as T

        img = Image.fromarray(img_rgb_uint8)
        x = T.ToTensor()(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        out = self.model.infer(x)
        depth = out["depth"] if isinstance(out, dict) else out

        if depth.ndim == 4:
            depth = depth[0, 0]

        return DepthResult(depth=depth.detach().float().cpu().numpy().astype(np.float32))

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray) -> DepthResult:
        return self.predict_depth(img_rgb_uint8)

    def save_depth_assets(self, out_dir: Path, depth_hw: np.ndarray) -> None:
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