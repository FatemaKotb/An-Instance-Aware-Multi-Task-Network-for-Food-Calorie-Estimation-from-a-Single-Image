from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from scripts.lvl2.instance_aware_fusion_head import InstanceAwareFusionHead


class CachedInstancesDataset(Dataset):
    def __init__(self, index_jsonl: str):
        self.items: List[Dict] = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        blob = torch.load(rec["cache_path"], map_location="cpu")
        return blob


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
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--split", type=str, choices=["train", "test"], default="test")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / f"index_{args.split}.jsonl"
    assert index_path.exists(), f"Missing {index_path}"

    ds = CachedInstancesDataset(str(index_path))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    ck = torch.load(args.ckpt, map_location="cpu")
    C = int(ck["in_channels"])
    model = InstanceAwareFusionHead(in_channels=C, num_classes=101).to(args.device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    correct = 0
    total = 0
    for roi_feat, mask_roi, depth_roi, depth_stats, y in dl:
        roi_feat = roi_feat.to(args.device)
        mask_roi = mask_roi.to(args.device)
        depth_roi = depth_roi.to(args.device)
        depth_stats = depth_stats.to(args.device)
        y = y.to(args.device)

        logits = model(roi_feat, mask_roi, depth_roi, depth_stats)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = float(correct) / float(total) if total > 0 else 0.0
    print(f"{args.split}_acc={acc:.4f}  (n={total})")


if __name__ == "__main__":
    main()
