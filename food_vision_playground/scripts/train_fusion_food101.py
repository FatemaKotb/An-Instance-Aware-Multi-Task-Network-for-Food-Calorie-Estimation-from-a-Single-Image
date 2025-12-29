from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

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
def evaluate(model: nn.Module, dl: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for roi_feat, mask_roi, depth_roi, depth_stats, y in dl:
        roi_feat = roi_feat.to(device, non_blocking=True)
        mask_roi = mask_roi.to(device, non_blocking=True)
        depth_roi = depth_roi.to(device, non_blocking=True)
        depth_stats = depth_stats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(roi_feat, mask_roi, depth_roi, depth_stats)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct) / float(total) if total > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--shard_lru_size", type=int, default=8)

    ap.add_argument("--out_ckpt", type=str, default="fusion_head_subset.pt")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "subset_meta.json"
    assert meta_path.exists(), f"Missing {meta_path}"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    num_classes = int(meta["k_classes"])

    train_index = cache_dir / "index_train.jsonl"
    val_index = cache_dir / "index_val.jsonl"
    test_index = cache_dir / "index_test.jsonl"
    assert train_index.exists(), f"Missing {train_index}"
    assert val_index.exists(), f"Missing {val_index}"
    assert test_index.exists(), f"Missing {test_index}"

    ds_tr = CachedInstancesDataset(str(train_index), shard_lru_size=args.shard_lru_size)
    ds_va = CachedInstancesDataset(str(val_index), shard_lru_size=args.shard_lru_size)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
    )

    # infer channels C
    first = ds_tr[0]
    C = int(first["roi_feat"].shape[0])

    model = InstanceAwareFusionHead(in_channels=C, num_classes=num_classes).to(args.device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    log_every = 100  # steps
    global_step = 0
    best = 0.0

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep}/{args.epochs}", dynamic_ncols=True)
        running = 0.0

        for step, (roi_feat, mask_roi, depth_roi, depth_stats, y) in enumerate(pbar, start=1):
            roi_feat = roi_feat.to(args.device, non_blocking=True)
            mask_roi = mask_roi.to(args.device, non_blocking=True)
            depth_roi = depth_roi.to(args.device, non_blocking=True)
            depth_stats = depth_stats.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(roi_feat, mask_roi, depth_roi, depth_stats)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            global_step += 1
            running += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

            if global_step % log_every == 0:
                avg = running / max(1, step)
                print(f"[ep={ep} step={step} global={global_step}] loss={avg:.4f}")

        val_acc = evaluate(model, dl_va, args.device)
        print(f"epoch {ep}: val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "in_channels": C, "num_classes": num_classes, "subset_meta": meta},
                args.out_ckpt,
            )
            print(f"saved best checkpoint to {args.out_ckpt} (val_acc={best:.4f})")

    print(f"done. best_val_acc={best:.4f}")


if __name__ == "__main__":
    main()
