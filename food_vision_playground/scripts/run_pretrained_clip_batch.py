from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from scripts.factory import PipelineFactoryConfig, build_default_pipeline


DISH_TOKENS = [
    "baba_ganoush",
    "baklava",
    "bamia",
    "basbousa",
    "bechamel_pasta",
    "bessara",
    "egyptian_lentil_soup",
    "falafel",
    "feteer_meshaltet",
    "ful_medames",
    "kahk",
    "molokhia",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {arr.shape} for {path}")
    return arr


def find_images_in_root_by_dish(root: Path, dish: str) -> List[Path]:
    dish = dish.lower()
    imgs: List[Path] = []
    if not root.exists():
        return imgs
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if dish in p.stem.lower():
                imgs.append(p)
    imgs.sort()
    return imgs


def top5_from_best_instance(pipe_out) -> Tuple[List[str], float]:
    insts = getattr(pipe_out, "instance_outputs", [])
    if not insts:
        return [], float("nan")
    best = max(insts, key=lambda x: float(getattr(x, "score", 0.0)))
    pred = best.prediction
    top5 = getattr(pred, "top5_foods", [])
    names = [name for (name, _prob) in top5]
    return names, float(getattr(best, "score", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing images.")
    parser.add_argument("--out_file", type=str, default="pretrained_clip_top5.txt", help="Output file.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seg_thresh", type=float, default=0.5, help="Segmentation score threshold.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    cfg = PipelineFactoryConfig(device=device, seg_score_thresh=float(args.seg_thresh))
    pipe = build_default_pipeline(cfg)

    lines: List[str] = []
    lines.append(f"device={device} seg_thresh={args.seg_thresh}")
    lines.append("format: <image_path> | top5=[c1, c2, c3, c4, c5] | best_inst_score=<score>")
    lines.append("")

    total_imgs = 0
    for dish in DISH_TOKENS:
        imgs = find_images_in_root_by_dish(data_dir, dish)
        lines.append(f"== {dish} ==")

        if not imgs:
            lines.append(f"(no images found in {data_dir} matching token '{dish}')")
            lines.append("")
            continue

        for img_path in imgs:
            total_imgs += 1
            try:
                img = load_rgb_uint8(img_path)
                out = pipe(img)
                top5_names, best_score = top5_from_best_instance(out)
                lines.append(f"{img_path} | top5={top5_names} | best_inst_score={best_score:.4f}")
            except Exception as e:
                lines.append(f"{img_path} | ERROR: {repr(e)}")

        lines.append("")

    out_file = Path(args.out_file)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_file.resolve()}  (images processed: {total_imgs})")


if __name__ == "__main__":
    main()