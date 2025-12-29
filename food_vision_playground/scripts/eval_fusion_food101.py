from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

from scripts.lvl2.instance_aware_fusion_head import InstanceAwareFusionHead


class _ShardLRU:
    def __init__(self, max_shards: int = 8):
        self.max_shards = int(max_shards)
        self._cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def get(self, path: str) -> List[Dict[str, Any]]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        shard = torch.load(path, map_location="cpu")
        self._cache[path] = shard
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_shards:
            self._cache.popitem(last=False)
        return shard


class CachedInstancesDataset(Dataset):
    def __init__(self, index_jsonl: str, shard_lru_size: int = 8):
        self.items: List[Dict[str, Any]] = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.lru = _ShardLRU(max_shards=shard_lru_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        shard = self.lru.get(rec["shard_path"])
        return shard[int(rec["item_index"])]


def collate(batch):
    roi_feat = torch.stack([b["roi_feat"].to(torch.float32) for b in batch], dim=0)
    mask_roi = torch.stack([b["mask_roi"].to(torch.float32) for b in batch], dim=0)
    depth_roi = torch.stack([b["depth_roi"].to(torch.float32) for b in batch], dim=0)
    depth_stats = torch.stack([b["depth_stats"].to(torch.float32) for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return roi_feat, mask_roi, depth_roi, depth_stats, labels


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--shard_lru_size", type=int, default=8)

    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.split == "val":
        index_path = cache_dir / "index_val.jsonl"
    else:
        index_path = cache_dir / "index_test.jsonl"
    assert index_path.exists(), f"Missing {index_path}"

    ds = CachedInstancesDataset(str(index_path), shard_lru_size=args.shard_lru_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )

    ck = torch.load(args.ckpt, map_location="cpu")
    C = int(ck["in_channels"])
    num_classes = int(ck["num_classes"])

    model = InstanceAwareFusionHead(in_channels=C, num_classes=num_classes).to(args.device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    correct = 0
    total = 0
    for roi_feat, mask_roi, depth_roi, depth_stats, y in dl:
        roi_feat = roi_feat.to(args.device, non_blocking=True)
        mask_roi = mask_roi.to(args.device, non_blocking=True)
        depth_roi = depth_roi.to(args.device, non_blocking=True)
        depth_stats = depth_stats.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)

        logits = model(roi_feat, mask_roi, depth_roi, depth_stats)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = float(correct) / float(total) if total > 0 else 0.0
    print(f"{args.split}_acc={acc:.4f}  (n={total})")


if __name__ == "__main__":
    main()
