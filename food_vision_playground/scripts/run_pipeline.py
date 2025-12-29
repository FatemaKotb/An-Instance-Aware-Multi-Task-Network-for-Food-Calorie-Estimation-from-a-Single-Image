# run_pipeline.py
from __future__ import annotations

import argparse
import logging
import time
import warnings
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.factory import PipelineFactoryConfig, build_default_pipeline

warnings.filterwarnings("ignore", message="torch.meshgrid:*")
warnings.filterwarnings("ignore", message="enable_nested_tensor*")

SEP = "\n" + ("=" * 88) + "\n"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class _Section:
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


def _nan_to_none(x):
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf
    except Exception:
        return None


def _save_depth_png(path: Path, depth_hw: np.ndarray) -> None:
    d = depth_hw.astype(np.float32)
    finite = np.isfinite(d)
    if finite.any():
        dmin = float(d[finite].min())
        dmax = float(d[finite].max())
        denom = (dmax - dmin) if (dmax > dmin) else 1.0
        vis = 255.0 * (d - dmin) / denom
        vis[~finite] = 0.0
    else:
        vis = np.zeros_like(d, dtype=np.float32)
    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(path)


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


def main() -> None:
    logger = _setup_logger()

    parser = argparse.ArgumentParser(description="Build pipeline via factory, run inference, save outputs.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation score threshold.")
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        choices=["hybrid_clip", "fusion_head"],
        default="hybrid_clip",
        help="Which prediction path to run: hybrid_clip (baseline) or fusion_head (trained classifier).",
    )
    parser.add_argument(
        "--fusion_ckpt",
        type=str,
        default="fusion_head_subset.pt",
        help="Fusion-head checkpoint path (used only when --pipeline_mode=fusion_head).",
    )
    parser.add_argument("--topk", type=int, default=5, help="How many instances to print.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Default: runs/<image_stem>/")
    args = parser.parse_args()

    with _Section(logger, "Device selection"):
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version (torch): {torch.version.cuda}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    with _Section(logger, "Load input image"):
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = ensure_uint8_rgb(load_rgb(str(img_path)))
        H, W = img.shape[:2]
        logger.info(f"Path: {img_path}")
        logger.info(f"Image shape: {img.shape}  dtype={img.dtype}")
        logger.info(f"Resolution: {W}x{H}")

    with _Section(logger, "Build pipeline via factory"):
        logger.info(f"seg_thresh: {float(args.seg_thresh)}")
        logger.info(f"pipeline_mode: {args.pipeline_mode}")
        cfg = PipelineFactoryConfig(
            device=device,
            seg_score_thresh=float(args.seg_thresh),
            pipeline_mode=str(args.pipeline_mode),
            fusion_head_ckpt_path=str(args.fusion_ckpt),
        )
        pipe = build_default_pipeline(cfg)

        logger.info(f"Backbone block: {type(pipe.backbone_block).__name__}")
        logger.info(f"Segmentation block: {type(pipe.seg_block).__name__}")
        logger.info(f"Depth block: {type(pipe.depth_block).__name__}")
        logger.info(f"Fusion block: {type(pipe.fusion_block).__name__}")
        logger.info(f"Prediction head: {type(pipe.prediction_head).__name__}")
        logger.info(f"Physics head: {type(pipe.physics_head).__name__}")

    with _Section(logger, "Run pipeline inference"):
        out = pipe(img)
        logger.info(f"Instances: {len(out.instance_outputs)}")
        logger.info(f"Backbone feature maps: {len(out.backbone_features)}")

        dmin = float(np.nanmin(out.depth_map))
        dmax = float(np.nanmax(out.depth_map))
        logger.info(f"Depth: shape={out.depth_map.shape} dtype={out.depth_map.dtype}")
        logger.info(f"Depth range: min={dmin:.6f} max={dmax:.6f}")

    with _Section(logger, "Save artifacts"):
        stem = img_path.stem
        out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output dir: {out_dir.resolve()}")

        save_uint8_rgb(out_dir / "input.png", img)
        logger.info("Saved: input.png")

        inst_dir = out_dir / "instances"
        inst_dir.mkdir(parents=True, exist_ok=True)

        instances_json = []
        for i, inst in enumerate(out.instance_outputs):
            m = inst.mask.astype(bool)

            masked_rgb = img.copy()
            masked_rgb[~m] = 0
            save_uint8_rgb(inst_dir / f"instance_{i:03d}_masked_rgb.png", masked_rgb)

            depth_masked = out.depth_map.astype(np.float32)
            depth_masked = np.where(m, depth_masked, np.nan).astype(np.float32)
            np.save(inst_dir / f"instance_{i:03d}_depth.npy", depth_masked.astype(np.float32))
            _save_depth_png(inst_dir / f"instance_{i:03d}_depth.png", depth_masked.astype(np.float32))

            pred = inst.prediction
            instances_json.append(
                {
                    "instance_id": int(i),
                    "food_class_name": str(getattr(pred, "food_class_name", "unknown")),
                    "food_conf": _nan_to_none(getattr(pred, "food_conf", None)),
                    "area_px": int(getattr(inst, "area_px", int(m.sum()))),
                    "depth_median": _nan_to_none(getattr(inst, "depth_median", None)),
                    "depth_mean": _nan_to_none(getattr(inst, "depth_mean", None)),
                    "seg_score": _nan_to_none(getattr(inst, "score", None)),
                }
            )

        (out_dir / "instances.json").write_text(
            json.dumps(instances_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved: instances/ + instances.json (objects={len(instances_json)})")

    with _Section(logger, "Per-instance summary"):
        topk = max(0, int(args.topk))
        logger.info(f"Printing top-{topk} instances")
        
        if len(out.instance_outputs) == 0:
            logger.info("No instances detected above threshold.")
            print("PRED_LABEL=None")
        else:
            instances = list(out.instance_outputs)
            instances.sort(
                key=lambda inst: float(getattr(inst.prediction, "food_conf", 0.0) or 0.0),
                reverse=True,
            )

            best = instances[0]  # after sorting
            print(f"PRED_LABEL={getattr(best.prediction, 'food_class_name', None)}")

            for i, inst in enumerate(instances[:topk]):
                pred = inst.prediction
                phys = inst.physics
                logger.info(
                    f"Instance {i:03d}:\n"
                    f"top5_foods={getattr(pred, 'top5_foods', None)}\n\n"
                    f"food_class_id={getattr(pred, 'food_class_id', None)}\n"
                    f"food_class_name={getattr(pred, 'food_class_name', None)}\n"
                    f"food_conf={_safe_float(getattr(pred, 'food_conf', None))}\n"
                )

        logger.info("DONE: run_pipeline finished successfully.")


if __name__ == "__main__":
    main()
