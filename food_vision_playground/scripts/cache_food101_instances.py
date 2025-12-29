from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.ops import roi_align
from PIL import Image
from tqdm import tqdm

# Your blocks (frozen)
from scripts.lvl1.efficientnet import EfficientNetBlock
from scripts.lvl1.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.lvl1.zoedepth import ZoeDepthBlock


@dataclass
class CacheItem:
    shard_path: str
    item_index: int
    label: int
    image_path: str


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


def _maybe_fix_food101_root(root: Path) -> Path:
    """
    Torchvision expects: root/food-101/meta/train.txt
    Some Kaggle zips extract differently; resolve root robustly.
    """
    if (root / "food-101" / "meta" / "train.txt").exists():
        return root
    if (root / "food-101" / "food-101" / "meta" / "train.txt").exists():
        return root / "food-101"
    # fallback: search for meta/train.txt
    candidates = list(root.rglob("meta/train.txt"))
    if candidates:
        # dataset_dir = .../<DATASET>/meta/train.txt -> .../<DATASET>
        dataset_dir = candidates[0].parent.parent
        return dataset_dir.parent
    return root


def _save_shard(shard_path: Path, items: List[Dict[str, Any]]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, shard_path)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--food101_root", type=str, default=".", help="Root where Food101 is stored (download=False).")
    ap.add_argument("--split", type=str, choices=["train", "test"], default="train")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write cached instances.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seg_thresh", type=float, default=0.5)
    ap.add_argument("--min_mask_area_ratio", type=float, default=0.20)
    ap.add_argument("--keep_largest_only", action="store_true", help="Use only the largest mask per image.")
    ap.add_argument("--max_instances_per_image", type=int, default=3)

    # perf knobs
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")

    # sharding
    ap.add_argument("--shard_size", type=int, default=5000, help="Instances per shard file (reduces I/O).")

    # logging
    ap.add_argument("--log_every", type=int, default=100, help="Log every N images.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen blocks ---
    backbone = EfficientNetBlock(mode="backbone", device=args.device)
    seg = MaskRCNNTorchVisionBlock(device=args.device)
    depth = ZoeDepthBlock(device=args.device)

    # Food-101
    split = args.split
    root = _maybe_fix_food101_root(Path(args.food101_root))
    ds = Food101(root=str(root), split=split, download=False)

    def _collate_one(batch):
        # batch is a list of length 1: [(PIL.Image, label)]
        return batch[0]

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_one,          # ✅ fix
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )

    index_path = out_dir / f"index_{split}.jsonl"
    wrote_instances = 0
    processed_images = 0

    shard_items: List[Dict[str, Any]] = []
    shard_id = 0

    t0 = time.time()
    last_log_t = t0

    pbar = tqdm(total=len(ds), desc=f"cache[{split}] images", dynamic_ncols=True)

    with index_path.open("w", encoding="utf-8") as f_index:
        for img_i, (pil_img, label) in enumerate(dl):
            # pil_img is PIL.Image.Image
            label = int(label)  # works for int or 0-d tensor

            label = int(label.item())

            # Food101 keeps file list
            img_path = str(ds._image_files[img_i])

            img = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            H, W = img.shape[:2]

            # segmentation
            seg_out = seg(img, score_thresh=float(args.seg_thresh))
            masks = seg_out.masks  # [N,H,W] bool
            boxes = seg_out.boxes_xyxy  # [N,4] float
            scores = seg_out.scores  # [N]
            if masks.shape[0] == 0:
                processed_images += 1
                pbar.update(1)
                continue

            # hard tiny-mask rule relative to image size (>= 20% by default)
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
            keep = areas >= (float(args.min_mask_area_ratio) * float(H * W))
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            if masks.shape[0] == 0:
                processed_images += 1
                pbar.update(1)
                continue

            # depth
            depth_out = depth(img)
            # your earlier script used depth_out.depth; keep same behavior
            depth_hw = depth_out.depth.astype(np.float32)  # [H,W]

            # backbone features
            bb = backbone(img)
            F = bb.features[-1]  # [1,C,Hf,Wf] on device
            _, C, Hf, Wf = F.shape
            spatial_scale = float(Wf) / float(W)

            order = _largest_instances_first(masks)
            if args.keep_largest_only:
                order = order[:1]
            else:
                order = order[: int(args.max_instances_per_image)]

            for j in order:
                m = masks[int(j)].astype(np.uint8)  # [H,W]
                b = boxes[int(j)].astype(np.float32)

                rois = torch.tensor([[0, b[0], b[1], b[2], b[3]]], device=F.device, dtype=torch.float32)
                roi_feat = roi_align(
                    input=F,
                    boxes=rois,
                    output_size=(7, 7),
                    spatial_scale=spatial_scale,
                    sampling_ratio=2,
                    aligned=True,
                ).squeeze(0)

                roi_feat_cpu = roi_feat.detach().cpu().to(torch.float16)  # [C,7,7]

                mask_roi = _resize_to_roi(m, b, out_hw=7).astype(np.uint8)
                depth_roi = _resize_to_roi(depth_hw, b, out_hw=7).astype(np.float32)
                depth_roi = (depth_roi * (mask_roi > 0)).astype(np.float32)
                d_stats = _depth_stats(depth_hw, m > 0)

                shard_items.append(
                    {
                        "roi_feat": roi_feat_cpu,  # float16 [C,7,7]
                        "mask_roi": torch.from_numpy(mask_roi[None, ...].astype(np.float32)),  # [1,7,7]
                        "depth_roi": torch.from_numpy(depth_roi[None, ...].astype(np.float32)),  # [1,7,7]
                        "depth_stats": torch.from_numpy(d_stats),  # [8]
                        "label": torch.tensor(label, dtype=torch.long),
                    }
                )

                # write index pointing into shard
                rec = CacheItem(
                    shard_path=str(shards_dir / f"{split}_shard{shard_id:04d}.pt"),
                    item_index=len(shard_items) - 1,
                    label=label,
                    image_path=img_path,
                )
                f_index.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                wrote_instances += 1

                # flush shard if full
                if len(shard_items) >= int(args.shard_size):
                    shard_path = shards_dir / f"{split}_shard{shard_id:04d}.pt"
                    _save_shard(shard_path, shard_items)
                    shard_items.clear()
                    shard_id += 1

            processed_images += 1
            pbar.update(1)

            if (processed_images % int(args.log_every)) == 0:
                now = time.time()
                dt = now - last_log_t
                elapsed = now - t0
                ips = (args.log_every / dt) if dt > 0 else 0.0
                pbar.set_postfix(instances=wrote_instances, img_s=f"{ips:.2f}", elapsed_min=f"{elapsed/60:.1f}")
                last_log_t = now

        # flush remaining
        if shard_items:
            shard_path = shards_dir / f"{split}_shard{shard_id:04d}.pt"
            _save_shard(shard_path, shard_items)
            shard_items.clear()

    pbar.close()
    print(f"Done. processed_images={processed_images} wrote_instances={wrote_instances}")
    print(f"Index: {index_path}")
    print(f"Shards: {shards_dir}")


if __name__ == "__main__":
    main()
