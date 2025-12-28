from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.pipeline import Pipeline
from scripts.level1_backbone.efficientnet import EfficientNetBackbone
from scripts.level2_depth.zoe_depth import ZoeDepthEstimator
from scripts.level3_instance.mask_rcnn import MaskRCNNInstanceHead
from scripts.level4_physics.physics_head import PhysicsHead
from scripts.dtypes import PipelineOutput

# Suppress warnings for clean output
warnings.filterwarnings("ignore", message="torch.meshgrid:*")
warnings.filterwarnings("ignore", message="enable_nested_tensor*")

# ---------------- Logging helpers ----------------

SEP = "\n" + ("=" * 88) + "\n"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
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
        return False


# ---------------- Image I/O ----------------

def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape={img.shape}")
    return img


def save_uint8_rgb(path: Path, img_rgb_uint8: np.ndarray) -> None:
    Image.fromarray(img_rgb_uint8).save(path)


def _safe_float(x) -> str:
    try:
        if x is None:
            return "None"
        xf = float(x)
        if np.isnan(xf):
            return "nan"
        return f"{xf:.6f}"
    except Exception:
        return repr(x)


# ---------------- Main ----------------

def main() -> None:
    logger = _setup_logger()

    parser = argparse.ArgumentParser(
        description="Run full multi-level food calorie estimation pipeline."
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--topk", type=int, default=5, help="Number of instances to print")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save outputs")
    args = parser.parse_args()

    # ---- Device selection ----
    with _Section(logger, "Device selection"):
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
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
        logger.info(f"Image shape: {img.shape}  dtype={img.dtype}")

    # ---- Build pipeline (explicit wiring) ----
    with _Section(logger, "Build pipeline"):
        backbone = EfficientNetBackbone(device=device)
        seg_block = MaskRCNNInstanceHead(device=device)
        depth_block = ZoeDepthEstimator(device=device)
        fusion_block = lambda feats, d_med, d_mean: feats  # simple placeholder
        prediction_head = lambda img_rgb_uint8, box_xyxy, mask_hw, fusion: PredictionOutput(
            food_class_name="unknown",
            food_class_id=-1,
            score=0.0,
            portion=0.0,
        )
        physics_head = PhysicsHead()

        pipe = Pipeline(
            backbone_block=backbone,
            seg_block=seg_block,
            depth_block=depth_block,
            fusion_block=fusion_block,
            prediction_head=prediction_head,
            physics_head=physics_head,
            device=device,
            seg_score_thresh=args.seg_thresh,
        )

    # ---- Run pipeline inference ----
    with _Section(logger, "Run pipeline inference"):
        out: PipelineOutput = pipe(img)
        logger.info(f"Instances detected: {len(out.instance_outputs)}")

    # ---- Save outputs ----
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_uint8_rgb(out_dir / "input.png", img)

    # ---- Per-instance summary ----
    topk = min(args.topk, len(out.instance_outputs))
    logger.info(f"Printing top-{topk} instances")
    if topk == 0:
        logger.info("No instances detected above threshold.")
    else:
        for i, inst in enumerate(out.instance_outputs[:topk]):
            pred = inst.prediction
            phys = inst.physics

            logger.info(
                f"[{i}] Class: {getattr(pred, 'food_class_name', 'unknown')} | "
                f"Score: {getattr(pred, 'score', 0.0):.3f} | "
                f"Area(px): {phys.area_px} | "
                f"Volume proxy: {_safe_float(phys.volume)} | "
                f"Calories: {_safe_float(phys.calories)}"
            )

    logger.info("Pipeline run completed successfully.")


if __name__ == "__main__":
    main()
