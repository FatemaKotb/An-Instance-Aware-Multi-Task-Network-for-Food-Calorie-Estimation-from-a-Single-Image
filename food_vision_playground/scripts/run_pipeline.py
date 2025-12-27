# run_pipeline.py
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.efficientnet import EfficientNetBlock
from scripts.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.zoedepth import ZoeDepthBlock
from scripts.fusion_stub import FusionStub
from scripts.pipeline import Pipeline
from scripts.prediction_head_stub import PredictionHeadStub
from scripts.physics_head_stub import PhysicsHeadStub

# Keep terminal output clean by default (you can loosen this if debugging)
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
    parser = argparse.ArgumentParser(description="Instantiate blocks, build pipeline, run inference on one image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation score threshold.")
    parser.add_argument("--topk", type=int, default=1, help="How many instances to print.")
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

    # ---- Instantiate blocks ----
    backbone_block = EfficientNetBlock(
        model_name="tf_efficientnetv2_s",
        device=device,
        mode="backbone",
        out_indices=(1, 2, 3, 4),
    )

    seg_block = MaskRCNNTorchVisionBlock(device=device)
    depth_block = ZoeDepthBlock(device=device)

    # Fusion outputs the three signals: (F_i, f_i, v_i)
    fusion = FusionStub()

    # PredictionHeadStub consumes fusion outputs and returns food+portion placeholders
    pred_head = PredictionHeadStub(num_food_classes=101)

    # PhysicsHeadStub consumes geometry + predicted class and returns area/volume/calories proxies
    phys_head = PhysicsHeadStub(default_kcal_per_volume=1.0)

    # ---- Build pipeline ----
    pipe = Pipeline(
        backbone_block=backbone_block,
        seg_block=seg_block,
        depth_block=depth_block,
        device=device,
        seg_score_thresh=args.seg_thresh,
        fusion=fusion,
        prediction_head=pred_head,
        physics_head=phys_head,
    )

    # ---- Run pipeline ----
    out = pipe(img)

    # ---- Save report assets ----
    REPO_ROOT = Path.cwd()
    stem = img_path.stem
    default_dir = REPO_ROOT / "runs" / stem
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    save_uint8_rgb(out_dir / "input.png", img)

    # Save masks (per-instance + overlay)
    seg_block.save_masks(out_dir, img, out.instance_outputs)

    # Save depth (raw + png)
    depth_block.save_depth_assets(out_dir, out.depth_map)

    # ---- Basic report (also save to file) ----
    lines = []
    lines.append("Pipeline ran successfully")
    lines.append(f"Image: {img_path}")
    lines.append(f"Instances: {len(out.instance_outputs)}")
    lines.append(
        "Depth: "
        f"shape={out.depth_map.shape} dtype={out.depth_map.dtype} "
        f"min={float(np.nanmin(out.depth_map)):.6f} max={float(np.nanmax(out.depth_map)):.6f}"
    )
    lines.append(f"Backbone feature maps: {len(out.backbone_features)}")
    lines.append("")

    # Show a small per-instance summary
    for i, inst in enumerate(out.instance_outputs[: args.topk]):
        # Prediction head outputs
        pred = inst.prediction
        food_id = pred.food_class_id
        food_conf = pred.food_conf

        # Physics head outputs
        phys = inst.physics

        lines.append(
            f"[{i}] seg_score={inst.score:.3f} seg_class_id={inst.seg_class_id} "
            f"area_px={phys.area_px} d_med={inst.depth_median:.4f} d_mean={inst.depth_mean:.4f} "
            f"food_id={food_id} food_conf={food_conf} portion={pred.portion} "
            f"volume={phys.volume:.2f} calories={phys.calories:.2f}"
        )

    report_text = "\n".join(lines)
    print("\n" + report_text)
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")

    print(f"\nSaved assets to: {out_dir}")


if __name__ == "__main__":
    main()
