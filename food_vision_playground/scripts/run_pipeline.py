# run_pipeline.py
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.factory import PipelineFactoryConfig, build_default_pipeline

# Keep terminal output clean by default
warnings.filterwarnings("ignore", message="torch.meshgrid:*")
warnings.filterwarnings("ignore", message="enable_nested_tensor*")


# ---------------- Logging helpers ----------------

SEP = "\n" + ("=" * 88) + "\n"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_pipeline")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when run multiple times (e.g., in notebooks)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class _Section:
    """Context manager that logs START/END with duration and a separator."""

    def __init__(self, logger: logging.Logger, title: str):
        self.logger = logger
        self.title = title
        self.t0 = 0.0

    def __enter__(self):
        self.logger.info(SEP + f"START: {self.title}")
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            self.logger.info(f"END:   {self.title}  |  took {dt:.2f}s")
        else:
            self.logger.exception(f"FAILED: {self.title}  |  after {dt:.2f}s")
        self.logger.info(SEP)
        # Don't swallow exceptions
        return False


# ---------------- Image I/O ----------------

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


def save_uint8_rgb(path: Path, img_rgb_uint8: np.ndarray) -> None:
    Image.fromarray(img_rgb_uint8).save(path)


def _safe_float(x) -> str:
    """Format float-ish values robustly for logs."""
    try:
        if x is None:
            return "None"
        xf = float(x)
        if np.isnan(xf):
            return "nan"
        return f"{xf:.6f}"
    except Exception:
        return repr(x)


def main() -> None:
    logger = _setup_logger()

    parser = argparse.ArgumentParser(description="Build default pipeline via factory, run inference, save outputs.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation score threshold.")
    parser.add_argument("--topk", type=int, default=5, help="How many instances to print.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Default: runs/<image_stem>/")
    args = parser.parse_args()

    # ---- Device selection ----
    with _Section(logger, "Device selection"):
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            # Helpful to debug "why not using GPU"
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version (torch): {torch.version.cuda}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    # ---- Load input image ----
    with _Section(logger, "Load input image"):
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = ensure_uint8_rgb(load_rgb(str(img_path)))
        H, W = img.shape[:2]
        logger.info(f"Path: {img_path}")
        logger.info(f"Image shape: {img.shape}  dtype={img.dtype}")
        logger.info(f"Resolution: {W}x{H}")

    # ---- Build pipeline via factory (Option 1 wiring) ----
    with _Section(logger, "Build pipeline via factory"):
        logger.info(f"seg_thresh: {float(args.seg_thresh)}")
        cfg = PipelineFactoryConfig(device=device, seg_score_thresh=float(args.seg_thresh))
        pipe = build_default_pipeline(cfg)

        # Best-effort: log block class names (helps confirm wiring)
        logger.info(f"Backbone block: {type(pipe.backbone_block).__name__}")
        logger.info(f"Segmentation block: {type(pipe.seg_block).__name__}")
        logger.info(f"Depth block: {type(pipe.depth_block).__name__}")
        logger.info(f"Fusion block: {type(pipe.fusion_block).__name__}")
        logger.info(f"Prediction head: {type(pipe.prediction_head).__name__}")
        logger.info(f"Physics head: {type(pipe.physics_head).__name__}")

    # ---- Run pipeline ----
    with _Section(logger, "Run pipeline inference"):
        out = pipe(img)

        logger.info(f"Instances: {len(out.instance_outputs)}")
        logger.info(f"Backbone feature maps: {len(out.backbone_features)}")

        # Depth summary
        dmin = float(np.nanmin(out.depth_map))
        dmax = float(np.nanmax(out.depth_map))
        logger.info(f"Depth: shape={out.depth_map.shape} dtype={out.depth_map.dtype}")
        logger.info(f"Depth range: min={dmin:.6f} max={dmax:.6f}")

    # ---- Output directory + save artifacts ----
    with _Section(logger, "Save artifacts"):
        stem = img_path.stem
        out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output dir: {out_dir.resolve()}")

        # Save input
        save_uint8_rgb(out_dir / "input.png", img)
        logger.info("Saved: input.png")

        # Save segmentation masks overlay (best-effort)
        try:
            k = min(len(out.instance_outputs), 10)
            pipe.seg_block.save_masks(out_dir, img, out.instance_outputs, topk=k)
            logger.info(f"Saved: masks_overlay.png + top-{k} mask_*.png")
        except Exception as e:
            logger.warning(f"Mask saving skipped: {repr(e)}")

        # Save depth map (raw npy)
        np.save(out_dir / "depth.npy", out.depth_map.astype(np.float32))
        logger.info("Saved: depth.npy")

    # ---- Per-instance summary ----
    with _Section(logger, "Per-instance summary"):
        topk = max(0, int(args.topk))
        logger.info(f"Printing top-{topk} instances")
        if len(out.instance_outputs) == 0:
            logger.info("No instances detected above threshold.")
        else:
            for i, inst in enumerate(out.instance_outputs[:topk]):
                pred = inst.prediction
                phys = inst.physics

                food_class_id = getattr(pred, "food_class_id", None)
                food_class_name = getattr(pred, "food_class_name", None)
                food_conf = getattr(pred, "food_conf", None)
                portion = getattr(pred, "portion", None)

                logger.info(
                    f"[{i}] score={inst.score:.3f} seg_class_id={inst.seg_class_id} "
                    f"area_px={phys.area_px} "
                    f"volume={_safe_float(phys.volume)} calories={_safe_float(phys.calories)} "
                    f"food_class_id={food_class_id} food_class_name={food_class_name} food_conf={_safe_float(food_conf)} portion={_safe_float(portion)}"
                )

    logger.info("DONE: run_pipeline finished successfully.")


if __name__ == "__main__":
    main()
