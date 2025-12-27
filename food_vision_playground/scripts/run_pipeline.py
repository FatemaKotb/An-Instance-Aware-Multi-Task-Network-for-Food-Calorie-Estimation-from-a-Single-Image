# run_pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
from PIL import Image

from scripts.factory import PipelineFactoryConfig, build_default_pipeline

# Keep terminal output clean by default
warnings.filterwarnings("ignore", message="torch.meshgrid:*")
warnings.filterwarnings("ignore", message="enable_nested_tensor*")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build default pipeline via factory, run inference, save outputs.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation score threshold.")
    parser.add_argument("--topk", type=int, default=5, help="How many instances to print.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory. Default: runs/<image_stem>/")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- Load input image ----
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = ensure_uint8_rgb(load_rgb(str(img_path)))
    print("Image:", img.shape, img.dtype)

    # ---- Build pipeline via factory (Option 1 wiring) ----
    cfg = PipelineFactoryConfig(device=device, seg_score_thresh=float(args.seg_thresh))
    pipe = build_default_pipeline(cfg)

    # ---- Run pipeline ----
    out = pipe(img)

    # ---- Save a minimal report ----
    stem = img_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_uint8_rgb(out_dir / "input.png", img)

    # save overlay via segmentation block if available
    try:
        pipe.seg_block.save_masks(out_dir, img, out.instance_outputs, topk=min(len(out.instance_outputs), 10))
    except Exception as e:
        print("Mask saving skipped:", repr(e))

    # Save depth map (raw npy)
    np.save(out_dir / "depth.npy", out.depth_map.astype(np.float32))

    # Print summary
    print("\nPipeline ran successfully")
    print("Image:", str(img_path))
    print("Instances:", len(out.instance_outputs))
    print(
        "Depth: "
        f"shape={out.depth_map.shape} dtype={out.depth_map.dtype} "
        f"min={float(np.nanmin(out.depth_map)):.6f} max={float(np.nanmax(out.depth_map)):.6f}"
    )
    print("Backbone feature maps:", len(out.backbone_features))
    print("")

    for i, inst in enumerate(out.instance_outputs[: args.topk]):
        pred = inst.prediction
        phys = inst.physics

        food_class = pred.food_class_id
        food_conf = pred.food_conf
        portion = pred.portion

        food_conf_str = "None" if food_conf is None else f"{float(food_conf):.2f}"
        portion_str = "None" if portion is None else f"{float(portion):.2f}"

        vol_str = "nan" if phys.volume is None or np.isnan(phys.volume) else f"{float(phys.volume):.2f}"
        cal_str = "nan" if phys.calories is None or np.isnan(phys.calories) else f"{float(phys.calories):.2f}"

        print(
            f"[{i}] score={inst.score:.3f} seg_class_id={inst.seg_class_id} "
            f"area_px={phys.area_px} volume={vol_str} calories={cal_str} "
            f"food_class={food_class} food_conf={food_conf_str} portion={portion_str}"
        )

    print("\nSaved assets to:", out_dir.resolve())


if __name__ == "__main__":
    main()
