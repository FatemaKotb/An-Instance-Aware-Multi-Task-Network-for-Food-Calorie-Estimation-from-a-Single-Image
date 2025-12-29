from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

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
    label: int          # NEW label in [0..K-1]
    image_path: str
    orig_label: int     # original Food101 label in [0..100]
    class_name: str


def _resize_to_roi(arr_hw: np.ndarray, box_xyxy: np.ndarray, out_hw: int = 7) -> np.ndarray:
    """Crop arr (H,W) by box and resize to (out_hw,out_hw)."""
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
    """Robust depth stats inside mask. Returns float32 vector of length 8."""
    m = mask_hw.astype(bool)
    vals = depth_hw[m].astype(np.float32)
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
    candidates = list(root.rglob("meta/train.txt"))
    if candidates:
        dataset_dir = candidates[0].parent.parent
        return dataset_dir.parent
    return root


def _save_shard(shard_path: Path, items: List[Dict[str, Any]]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, shard_path)


def _choose_k_classes(
    all_classes: List[str],
    k: int,
    seed: int,
) -> List[int]:
    """
    Deterministic class subset selection: shuffle class indices with seed, take first K.
    """
    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(all_classes))
    rng.shuffle(idx)
    idx = idx[: int(k)]
    return sorted(idx.tolist())


def _build_val_index_set(
    labels: List[int],
    chosen_orig_labels: Set[int],
    val_ratio: float,
    seed: int,
) -> Set[int]:
    """
    For the TRAIN split only: build a set of global image indices that go to VAL,
    stratified per chosen class (deterministic).
    """
    rng = np.random.RandomState(int(seed))
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        if y in chosen_orig_labels:
            by_class.setdefault(y, []).append(i)

    val_set: Set[int] = set()
    for y, idxs in by_class.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n_val = int(round(float(val_ratio) * float(len(idxs))))
        if n_val > 0:
            val_set.update(idxs[:n_val].tolist())
    return val_set


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--food101_root", type=str, required=True, help="Root where Food101 is stored (download=False).")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write cached instances.")
    ap.add_argument("--device", type=str, default="cuda")

    # split control
    ap.add_argument("--split", type=str, choices=["train", "test"], default="train")
    ap.add_argument("--val_ratio", type=float, default=0.10, help="Only used when split=train. Stratified per-class.")

    # class subset control
    ap.add_argument("--k_classes", type=int, default=20, help="Number of Food-101 classes to keep (K).")
    ap.add_argument("--class_seed", type=int, default=42, help="Seed for choosing the K classes.")
    ap.add_argument("--split_seed", type=int, default=123, help="Seed for splitting train->train/val (per class).")

    # instance selection
    ap.add_argument("--seg_thresh", type=float, default=0.5)
    ap.add_argument("--min_mask_area_ratio", type=float, default=0.20)  # 20% of image pixels
    ap.add_argument("--keep_largest_only", action="store_true", help="Use only the largest mask per image.")
    ap.add_argument("--max_instances_per_image", type=int, default=3)

    # perf knobs
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")

    # sharding
    ap.add_argument("--shard_size", type=int, default=5000, help="Instances per shard file (reduces I/O).")

    # logging
    ap.add_argument("--log_every", type=int, default=10, help="Log every N images.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen blocks ---
    backbone = EfficientNetBlock(mode="backbone", device=args.device)
    seg = MaskRCNNTorchVisionBlock(device=args.device)
    depth = ZoeDepthBlock(device=args.device)

    # dataset
    root = _maybe_fix_food101_root(Path(args.food101_root))
    ds = Food101(root=str(root), split=args.split, download=False)
    all_class_names: List[str] = list(ds.classes)

    # choose K classes (orig labels)
    chosen_orig = _choose_k_classes(all_class_names, k=int(args.k_classes), seed=int(args.class_seed))
    chosen_orig_set = set(chosen_orig)

    # map orig label -> new label [0..K-1]
    orig_to_new = {orig: new for new, orig in enumerate(chosen_orig)}
    new_to_orig = {v: k for k, v in orig_to_new.items()}
    chosen_names = [all_class_names[i] for i in chosen_orig]

    # write subset metadata once (shared by train/test)
    subset_meta_path = out_dir / "subset_meta.json"
    subset_meta = {
        "k_classes": int(args.k_classes),
        "chosen_orig_labels_sorted": chosen_orig,
        "chosen_class_names": chosen_names,
        "orig_to_new": orig_to_new,
        "new_to_orig": new_to_orig,
        "class_seed": int(args.class_seed),
        "split_seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
    }
    subset_meta_path.write_text(json.dumps(subset_meta, indent=2), encoding="utf-8")

    # build VAL set only for split=train
    val_index_set: Set[int] = set()
    if args.split == "train" and float(args.val_ratio) > 0:
        # Food101 keeps labels internally; different torchvision versions use different names.
        # We rely on dataset.targets if present; fallback to private _labels.
        if hasattr(ds, "targets"):
            labels_list = list(ds.targets)
        else:
            labels_list = list(ds._labels)  # type: ignore[attr-defined]
        val_index_set = _build_val_index_set(
            labels=labels_list,
            chosen_orig_labels=chosen_orig_set,
            val_ratio=float(args.val_ratio),
            seed=int(args.split_seed),
        )

    def _collate_one(batch):
        # batch size is 1: [(PIL.Image, label_int)]
        return batch[0]

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_one,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )

    # output indices for this run
    if args.split == "train":
        index_paths = {
            "train": out_dir / "index_train.jsonl",
            "val": out_dir / "index_val.jsonl",
        }
    else:
        index_paths = {
            "test": out_dir / "index_test.jsonl",
        }

    # open all index files
    f_index = {k: p.open("w", encoding="utf-8") for k, p in index_paths.items()}

    # separate shard buffers per split-name
    shard_items: Dict[str, List[Dict[str, Any]]] = {k: [] for k in index_paths.keys()}
    shard_id: Dict[str, int] = {k: 0 for k in index_paths.keys()}

    processed_images = 0
    wrote_instances = 0

    t0 = time.time()
    last_log_t = t0

    pbar = tqdm(total=len(ds), desc=f"cache[{args.split}] images", dynamic_ncols=True)

    try:
        for img_i, (pil_img, orig_label) in enumerate(dl):
            orig_label = int(orig_label)

            # filter to chosen K classes
            if orig_label not in chosen_orig_set:
                processed_images += 1
                pbar.update(1)
                continue

            new_label = int(orig_to_new[orig_label])
            class_name = chosen_names[new_label]

            # assign split bucket
            if args.split == "train":
                bucket = "val" if (img_i in val_index_set) else "train"
            else:
                bucket = "test"

            # Food101 keeps file list
            img_path = str(ds._image_files[img_i])

            img = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            H, W = img.shape[:2]

            # segmentation
            seg_out = seg(img, score_thresh=float(args.seg_thresh))
            masks = seg_out.masks
            boxes = seg_out.boxes_xyxy
            scores = seg_out.scores
            if masks.shape[0] == 0:
                processed_images += 1
                pbar.update(1)
                continue

            # guaranteed removal of tiny masks relative to image size
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
            keep = areas >= (float(args.min_mask_area_ratio) * float(H * W))
            masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
            if masks.shape[0] == 0:
                processed_images += 1
                pbar.update(1)
                continue

            # depth
            depth_out = depth(img)
            depth_hw = depth_out.depth.astype(np.float32)  # [H,W]

            # backbone features
            bb = backbone(img)
            F = bb.features[-1]  # [1,C,Hf,Wf]
            _, C, Hf, Wf = F.shape
            spatial_scale = float(Wf) / float(W)

            order = _largest_instances_first(masks)
            if args.keep_largest_only:
                order = order[:1]
            else:
                order = order[: int(args.max_instances_per_image)]

            for j in order:
                m = masks[int(j)].astype(np.uint8)
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

                shard_items[bucket].append(
                    {
                        "roi_feat": roi_feat_cpu,
                        "mask_roi": torch.from_numpy(mask_roi[None, ...].astype(np.float32)),
                        "depth_roi": torch.from_numpy(depth_roi[None, ...].astype(np.float32)),
                        "depth_stats": torch.from_numpy(d_stats),
                        "label": torch.tensor(new_label, dtype=torch.long),      # NEW label
                        "orig_label": int(orig_label),
                        "class_name": class_name,
                    }
                )

                shard_path = shards_dir / f"{bucket}_shard{shard_id[bucket]:04d}.pt"
                rec = CacheItem(
                    shard_path=str(shard_path),
                    item_index=len(shard_items[bucket]) - 1,
                    label=new_label,
                    image_path=img_path,
                    orig_label=int(orig_label),
                    class_name=class_name,
                )
                f_index[bucket].write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                wrote_instances += 1

                # flush shard if full
                if len(shard_items[bucket]) >= int(args.shard_size):
                    _save_shard(shard_path, shard_items[bucket])
                    shard_items[bucket].clear()
                    shard_id[bucket] += 1

            processed_images += 1
            pbar.update(1)

            if (processed_images % int(args.log_every)) == 0:
                now = time.time()
                dt = now - last_log_t
                elapsed = now - t0
                ips = (args.log_every / dt) if dt > 0 else 0.0
                pbar.set_postfix(instances=wrote_instances, img_s=f"{ips:.2f}", elapsed_min=f"{elapsed/60:.1f}")
                last_log_t = now

        # flush remaining shards
        for bucket, items in shard_items.items():
            if items:
                shard_path = shards_dir / f"{bucket}_shard{shard_id[bucket]:04d}.pt"
                _save_shard(shard_path, items)
                items.clear()

    finally:
        pbar.close()
        for f in f_index.values():
            f.close()

    print(f"Subset meta: {subset_meta_path}")
    print(f"Done. processed_images={processed_images} wrote_instances={wrote_instances}")
    for k, p in index_paths.items():
        print(f"Index[{k}]: {p}")
    print(f"Shards: {shards_dir}")


if __name__ == "__main__":
    main()
