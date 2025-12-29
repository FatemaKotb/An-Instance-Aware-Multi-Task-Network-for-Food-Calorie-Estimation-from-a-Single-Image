from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from torchvision.ops import roi_align
from PIL import Image
from tqdm import tqdm

# Reuse your project blocks + Pipeline utilities (consistency with pipeline.run)
from scripts.lvl1.efficientnet import EfficientNetBlock
from scripts.lvl1.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.lvl1.zoedepth import ZoeDepthBlock
from scripts.pipeline import Pipeline


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


def _depth_stats_from_mask(depth_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """
    Depth stats used by the fusion head.

    We keep this as an 8-d vector, but we compute the key depth summaries
    (median/mean) using Pipeline’s own implementation for consistency.
    Layout: [mean, std, mad, q10, q25, q50, q75, q90]
    """
    vals = depth_hw[mask_hw.astype(bool)].astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((8,), dtype=np.float32)

    q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90]).astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    mad = float(np.median(np.abs(vals - q50)))
    return np.array([mean, std, mad, q10, q25, q50, q75, q90], dtype=np.float32)


def _largest_instances_first(masks_nhw: np.ndarray) -> np.ndarray:
    """Order instance indices by mask area descending."""
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


def _choose_k_classes(all_classes: List[str], k: int, seed: int) -> List[int]:
    """Deterministic subset: shuffle class indices with seed, take first K (returned sorted)."""
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
    """Stratified train->val split over the chosen classes (indices are ORIGINAL ds indices)."""
    rng = np.random.RandomState(int(seed))
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        if y in chosen_orig_labels:
            by_class.setdefault(y, []).append(i)

    val_set: Set[int] = set()
    for _, idxs in by_class.items():
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

    # backbone config (match project defaults unless overridden)
    ap.add_argument("--backbone_model_name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--backbone_out_indices", type=str, default="1,2,3,4")

    # instance selection
    ap.add_argument("--seg_thresh", type=float, default=0.5)

    # IMPORTANT: these are fed into MaskRCNNTorchVisionBlock (single source of truth)
    ap.add_argument("--min_mask_area_ratio", type=float, default=0.20)
    ap.add_argument("--iou_dup_thresh", type=float, default=0.70)
    ap.add_argument("--contain_thresh", type=float, default=0.97)
    ap.add_argument("--union_area_ratio", type=float, default=1.00)

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
    ap.add_argument("--heartbeat_every", type=float, default=30.0, help="Print heartbeat every N seconds.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # -------------------
    # loud init logs
    # -------------------
    print(f"[start] split={args.split} device={args.device}")
    print(f"[start] out_dir={out_dir}")
    print(f"[start] food101_root={args.food101_root}")
    print(
        "[start] seg_thresh="
        f"{args.seg_thresh} min_mask_area_ratio={args.min_mask_area_ratio} "
        f"iou_dup_thresh={args.iou_dup_thresh} contain_thresh={args.contain_thresh} "
        f"union_area_ratio={args.union_area_ratio}"
    )
    print(
        f"[start] K={args.k_classes} class_seed={args.class_seed} "
        f"split_seed={args.split_seed} val_ratio={args.val_ratio}"
    )

    out_indices = tuple(int(x.strip()) for x in str(args.backbone_out_indices).split(",") if x.strip())
    print(f"[start] backbone_model_name={args.backbone_model_name} backbone_out_indices={out_indices}")

    # -------------------
    # build blocks (reused)
    # -------------------
    print("[init] building blocks (same classes as pipeline uses)...")

    t = time.time()
    backbone = EfficientNetBlock(
        model_name=args.backbone_model_name,
        device=args.device,
        mode="backbone",
        out_indices=out_indices,
    )
    print(f"[init] backbone done in {time.time() - t:.1f}s")

    t = time.time()
    seg = MaskRCNNTorchVisionBlock(
        device=args.device,
        min_mask_area_ratio=float(args.min_mask_area_ratio),
        iou_dup_thresh=float(args.iou_dup_thresh),
        contain_thresh=float(args.contain_thresh),
        union_area_ratio=float(args.union_area_ratio),
    )
    print(f"[init] seg done in {time.time() - t:.1f}s")

    t = time.time()
    depth = ZoeDepthBlock(device=args.device)
    print(f"[init] depth done in {time.time() - t:.1f}s")

    # Create a Pipeline instance ONLY to reuse its utility methods (consistency),
    # we do NOT call pipeline.run() to avoid instantiating prediction/physics heads.
    pipe_utils = Pipeline(
        backbone_block=backbone,
        seg_block=seg,
        depth_block=depth,
        fusion_block=None,
        prediction_head=None,
        physics_head=None,
        device=args.device,
        seg_score_thresh=float(args.seg_thresh),
    )

    # -------------------
    # dataset
    # -------------------
    print("[data] loading Food101...")
    root = _maybe_fix_food101_root(Path(args.food101_root))
    print(f"[data] resolved root={root}")
    ds = Food101(root=str(root), split=args.split, download=False)
    all_class_names: List[str] = list(ds.classes)
    print(f"[data] loaded: total_images={len(ds)} classes={len(all_class_names)}")

    # subset selection (class list)
    chosen_orig = _choose_k_classes(all_class_names, k=int(args.k_classes), seed=int(args.class_seed))
    chosen_orig_set = set(chosen_orig)
    orig_to_new = {orig: new for new, orig in enumerate(chosen_orig)}
    new_to_orig = {v: k for k, v in orig_to_new.items()}
    chosen_names = [all_class_names[i] for i in chosen_orig]

    print("[subset] chosen class names (new_label -> name):")
    for new_label, name in enumerate(chosen_names):
        print(f"  [{new_label:02d}] {name}")

    # Get labels list for building indices / val split (ORIGINAL ds indices)
    if hasattr(ds, "targets"):
        labels_list = [int(x) for x in list(ds.targets)]
    else:
        labels_list = [int(x) for x in list(ds._labels)]  # type: ignore[attr-defined]

    # Build subset indices so we DO NOT iterate over all 75,750 images
    subset_indices = [i for i, y in enumerate(labels_list) if y in chosen_orig_set]
    print(f"[subset] keeping images from chosen classes only: {len(subset_indices)}/{len(ds)}")

    # Write subset metadata (record exact seg params used)
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
        "backbone_model_name": args.backbone_model_name,
        "backbone_out_indices": list(out_indices),
        "seg_thresh": float(args.seg_thresh),
        "min_mask_area_ratio": float(args.min_mask_area_ratio),
        "iou_dup_thresh": float(args.iou_dup_thresh),
        "contain_thresh": float(args.contain_thresh),
        "union_area_ratio": float(args.union_area_ratio),
        "keep_largest_only": bool(args.keep_largest_only),
        "max_instances_per_image": int(args.max_instances_per_image),
    }
    subset_meta_path.write_text(json.dumps(subset_meta, indent=2), encoding="utf-8")
    print(f"[subset] wrote {subset_meta_path}")

    # val split (train only) computed over ORIGINAL indices, but only meaningful on the chosen K
    print("[split] preparing val split (if split=train)...")
    val_index_set: Set[int] = set()
    if args.split == "train" and float(args.val_ratio) > 0:
        val_index_set = _build_val_index_set(
            labels=labels_list,
            chosen_orig_labels=chosen_orig_set,
            val_ratio=float(args.val_ratio),
            seed=int(args.split_seed),
        )
        # Report how many of the SUBSET will be val
        val_in_subset = sum(1 for i in subset_indices if i in val_index_set)
        print(f"[split] val_ratio={args.val_ratio} -> val_images_in_subset={val_in_subset}/{len(subset_indices)}")
    else:
        print("[split] no val split")

    # Build ds subset so dataloader is small and tqdm total is small
    ds_subset = Subset(ds, subset_indices)

    # dataloader (avoid PIL collation issues with workers)
    def _collate_one(batch):
        return batch[0]  # (PIL.Image, int)

    print("[data] building DataLoader over ds_subset...")
    dl = DataLoader(
        ds_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_one,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )
    print(
        "[data] DataLoader ready "
        f"(subset_images={len(ds_subset)}, workers={args.num_workers}, pin_memory={bool(args.pin_memory)}, "
        f"prefetch_factor={args.prefetch_factor}, persistent_workers={bool(args.persistent_workers)})"
    )

    # output indices
    if args.split == "train":
        index_paths = {"train": out_dir / "index_train.jsonl", "val": out_dir / "index_val.jsonl"}
    else:
        index_paths = {"test": out_dir / "index_test.jsonl"}

    print("[io] opening index files:")
    for k, p in index_paths.items():
        print(f"  - {k}: {p}")

    f_index = {k: p.open("w", encoding="utf-8") for k, p in index_paths.items()}
    shard_items: Dict[str, List[Dict[str, Any]]] = {k: [] for k in index_paths.keys()}
    shard_id: Dict[str, int] = {k: 0 for k in index_paths.keys()}

    # warmup (prevents “silent for minutes” on first image)
    print("[warmup] running 1 dummy forward (seg/depth/backbone)...")
    dummy = np.zeros((384, 512, 3), dtype=np.uint8)

    t = time.time()
    _ = seg(dummy, score_thresh=float(args.seg_thresh))
    print(f"[warmup] seg done in {time.time() - t:.1f}s")

    t = time.time()
    _ = depth(dummy)
    print(f"[warmup] depth done in {time.time() - t:.1f}s")

    t = time.time()
    _ = backbone(dummy)
    print(f"[warmup] backbone done in {time.time() - t:.1f}s")
    print("[warmup] done. entering main loop...")

    # main loop
    processed_images = 0
    wrote_instances = 0
    t0 = time.time()
    last_log_t = t0
    last_heartbeat = time.time()

    pbar = tqdm(total=len(ds_subset), desc=f"cache[{args.split}] images", dynamic_ncols=True)

    try:
        for sub_i, (pil_img, orig_label) in enumerate(dl):
            # Recover ORIGINAL index in the underlying Food101 dataset
            img_i = subset_indices[sub_i]

            # label can be int or 0-d tensor depending on env
            orig_label = int(orig_label) if not hasattr(orig_label, "item") else int(orig_label.item())

            # Now that we're using ds_subset, this should always be true, but keep as safety.
            if orig_label not in chosen_orig_set:
                processed_images += 1
                pbar.update(1)
                continue

            new_label = int(orig_to_new[orig_label])
            class_name = chosen_names[new_label]

            if args.split == "train":
                bucket = "val" if (img_i in val_index_set) else "train"
            else:
                bucket = "test"

            img_path = str(ds._image_files[img_i])
            img = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)

            # segmentation (single source of truth: tiny-mask + dedupe + union removal are INSIDE this block)
            seg_out = seg(img, score_thresh=float(args.seg_thresh))
            masks = seg_out.masks
            boxes = seg_out.boxes_xyxy

            if masks.shape[0] == 0:
                processed_images += 1
                pbar.update(1)
                now = time.time()
                if now - last_heartbeat >= float(args.heartbeat_every):
                    elapsed = now - t0
                    print(
                        f"[heartbeat] images={processed_images}/{len(ds_subset)} "
                        f"instances={wrote_instances} elapsed_min={elapsed/60:.1f}"
                    )
                    last_heartbeat = now
                continue

            # depth (your existing class)
            depth_hw = depth(img).depth.astype(np.float32)  # [H,W]

            # backbone (your existing class)
            back_out = backbone(img)
            feats = back_out.features
            F = feats[-1]  # [1,C,Hf,Wf]

            # compute spatial scale from feature map size vs image size
            H_img, W_img = img.shape[:2]
            _, C, Hf, Wf = F.shape
            spatial_scale = float(Wf) / float(W_img)

            order = _largest_instances_first(masks)
            if args.keep_largest_only:
                order = order[:1]
            else:
                order = order[: int(args.max_instances_per_image)]

            for j in order:
                m = masks[int(j)].astype(np.uint8)
                b = boxes[int(j)].astype(np.float32)

                # ROIAlign for a compact spatial representation (kept for fusion head training format)
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

                # mask/depth ROIs
                mask_roi = _resize_to_roi(m, b, out_hw=7).astype(np.uint8)
                depth_roi = _resize_to_roi(depth_hw, b, out_hw=7).astype(np.float32)
                depth_roi = (depth_roi * (mask_roi > 0)).astype(np.float32)

                # IMPORTANT: depth summaries computed using Pipeline utility for consistency
                d_med, d_mean = pipe_utils._mask_pool_depth(depth_hw, m.astype(bool))

                # richer stats for head; overwrite mean/median to match pipeline exactly
                d_stats = _depth_stats_from_mask(depth_hw, m.astype(bool))
                d_stats[0] = float(d_mean)  # mean
                d_stats[5] = float(d_med)   # median (q50)

                shard_items[bucket].append(
                    {
                        "roi_feat": roi_feat_cpu,  # float16 [C,7,7]
                        "mask_roi": torch.from_numpy(mask_roi[None, ...].astype(np.float32)),     # [1,7,7]
                        "depth_roi": torch.from_numpy(depth_roi[None, ...].astype(np.float32)),   # [1,7,7]
                        "depth_stats": torch.from_numpy(d_stats.astype(np.float32)),              # [8]
                        "label": torch.tensor(new_label, dtype=torch.long),
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

            now = time.time()
            if now - last_heartbeat >= float(args.heartbeat_every):
                elapsed = now - t0
                print(
                    f"[heartbeat] images={processed_images}/{len(ds_subset)} "
                    f"instances={wrote_instances} elapsed_min={elapsed/60:.1f}"
                )
                last_heartbeat = now

        print("[io] flushing remaining shard buffers...")
        for bucket, items in shard_items.items():
            if items:
                shard_path = shards_dir / f"{bucket}_shard{shard_id[bucket]:04d}.pt"
                _save_shard(shard_path, items)
                items.clear()

    finally:
        pbar.close()
        for f in f_index.values():
            f.close()

    print(f"[done] subset_meta: {subset_meta_path}")
    print(f"[done] processed_images={processed_images} wrote_instances={wrote_instances}")
    for k, p in index_paths.items():
        print(f"[done] Index[{k}]: {p}")
    print(f"[done] Shards: {shards_dir}")


if __name__ == "__main__":
    main()
