from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.ops import roi_align
from PIL import Image

# Your blocks (frozen)
from scripts.lvl1.efficientnet import EfficientNetBlock
from scripts.lvl1.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.lvl1.zoedepth import ZoeDepthBlock


@dataclass
class CacheItem:
    cache_path: str
    label: int
    image_path: str


def _to_uint8_rgb(p: str) -> np.ndarray:
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resize_to_roi(arr_hw: np.ndarray, box_xyxy: np.ndarray, out_hw: int = 7) -> np.ndarray:
    """
    Crop arr (H,W) by box and resize to (out_hw,out_hw).
    Uses PIL for simplicity.
    """
    H, W = arr_hw.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = int(max(0, min(W - 1, round(x1))))
    x2 = int(max(0, min(W, round(x2))))
    y1 = int(max(0, min(H - 1, round(y1))))
    y2 = int(max(0, min(H, round(y2))))
    if x2 <= x1 or y2 <= y1:
        # fallback: return zeros
        return np.zeros((out_hw, out_hw), dtype=arr_hw.dtype)

    crop = arr_hw[y1:y2, x1:x2]
    pil = Image.fromarray(crop)
    pil = pil.resize((out_hw, out_hw), resample=Image.NEAREST)
    return np.asarray(pil, dtype=arr_hw.dtype)


def _depth_stats(depth_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """
    Robust depth stats inside mask. Returns float32 vector.
    """
    m = mask_hw.astype(bool)
    vals = depth_hw[m].astype(np.float32)
    if vals.size == 0:
        return np.zeros((8,), dtype=np.float32)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((8,), dtype=np.float32)

    q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90]).astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    mad = float(np.median(np.abs(vals - q50)))
    return np.array([mean, std, mad, q10, q25, q50, q75, q90], dtype=np.float32)


def _largest_instances_first(masks_nhw: np.ndarray) -> np.ndarray:
    areas = masks_nhw.reshape(masks_nhw.shape[0], -1).sum(axis=1)
    return np.argsort(areas)[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--food101_root", type=str, default=".", help="Root where Food101 will be downloaded/stored.")
    ap.add_argument("--split", type=str, choices=["train", "test"], default="train")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write cached instances.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seg_thresh", type=float, default=0.5)
    ap.add_argument("--min_mask_area_ratio", type=float, default=0.20)
    ap.add_argument("--keep_largest_only", action="store_true", help="Use only the largest mask per image.")
    ap.add_argument("--max_instances_per_image", type=int, default=3)
    ap.add_argument("--batch_images", type=int, default=1, help="Kept at 1 because blocks operate per image.")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "items").mkdir(parents=True, exist_ok=True)

    # --- frozen blocks ---
    backbone = EfficientNetBlock(mode="backbone", device=args.device)
    seg = MaskRCNNTorchVisionBlock(device=args.device)  # must apply seg_thresh in call
    depth = ZoeDepthBlock(device=args.device)

    # Food-101
    split = args.split
    ds = Food101(root=args.food101_root, split=split, download=False)

    # DataLoader over indices (we need image path + label)
    def collate(batch):
        # batch is list of (PIL_image, label)
        # We need underlying path; Food101 stores paths in ds._image_files
        return batch

    dl = DataLoader(ds, batch_size=args.batch_images, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    index_path = out_dir / f"index_{split}.jsonl"
    wrote = 0

    with index_path.open("w", encoding="utf-8") as f_index:
        for batch_idx, batch in enumerate(dl):
            # batch size = 1 by design
            pil_img, label = batch[0]
            label = int(label)

            # Get path: Food101 keeps file list as strings
            img_path = str(ds._image_files[batch_idx])
            img = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            H, W = img.shape[:2]

            # --- run frozen blocks ---
            # segmentation
            seg_out = seg(img, score_thresh=float(args.seg_thresh))
            masks = seg_out.masks  # [N,H,W] bool
            boxes = seg_out.boxes_xyxy  # [N,4] float
            scores = seg_out.scores  # [N]

            if masks.shape[0] == 0:
                continue

            # hard tiny-mask rule relative to image size
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
            keep = areas >= (float(args.min_mask_area_ratio) * float(H * W))
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            if masks.shape[0] == 0:
                continue

            # depth
            depth_out = depth(img)
            depth_hw = depth_out.depth.astype(np.float32)  # [H,W]

            # backbone features
            bb = backbone(img)
            # Use last feature level for ROIAlign (most semantic)
            F = bb.features[-1]  # [1,C,Hf,Wf] on device
            _, C, Hf, Wf = F.shape
            spatial_scale = float(Wf) / float(W)

            # choose instance order
            order = _largest_instances_first(masks)
            if args.keep_largest_only:
                order = order[:1]
            else:
                order = order[: int(args.max_instances_per_image)]

            for j in order:
                m = masks[int(j)].astype(np.uint8)  # [H,W] {0,1}
                b = boxes[int(j)].astype(np.float32)  # [4]

                # ROIAlign appearance
                rois = torch.tensor([[0, b[0], b[1], b[2], b[3]]], device=F.device, dtype=torch.float32)
                roi_feat = roi_align(
                    input=F,
                    boxes=rois,
                    output_size=(7, 7),
                    spatial_scale=spatial_scale,
                    sampling_ratio=2,
                    aligned=True,
                )  # [1,C,7,7]
                roi_feat = roi_feat.squeeze(0).detach().cpu().to(torch.float16)  # [C,7,7]

                # ROI mask + ROI depth
                mask_roi = _resize_to_roi(m, b, out_hw=7).astype(np.uint8)  # [7,7]
                depth_roi = _resize_to_roi(depth_hw, b, out_hw=7).astype(np.float32)  # [7,7]
                depth_roi = (depth_roi * (mask_roi > 0)).astype(np.float32)

                d_stats = _depth_stats(depth_hw, m > 0)  # [8] float32

                item = {
                    "roi_feat": roi_feat,  # torch float16 [C,7,7]
                    "mask_roi": torch.from_numpy(mask_roi[None, ...].astype(np.float32)),  # [1,7,7]
                    "depth_roi": torch.from_numpy(depth_roi[None, ...].astype(np.float32)),  # [1,7,7]
                    "depth_stats": torch.from_numpy(d_stats),  # [8]
                    "label": torch.tensor(label, dtype=torch.long),
                }

                item_path = out_dir / "items" / f"{split}_{batch_idx:06d}_inst{int(j):02d}.pt"
                torch.save(item, item_path)

                rec = CacheItem(cache_path=str(item_path), label=label, image_path=img_path)
                f_index.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                wrote += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"[{split}] processed {batch_idx+1}/{len(ds)} images, wrote {wrote} instances")

    print(f"Done. Wrote {wrote} cached instances to {out_dir} and index {index_path}")


if __name__ == "__main__":
    main()
