from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

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
    roi_feat = torch.stack([b["roi_feat"].to(torch.float32) for b in batch], dim=0)       # [B,C,7,7]
    mask_roi = torch.stack([b["mask_roi"].to(torch.float32) for b in batch], dim=0)       # [B,1,7,7]
    depth_roi = torch.stack([b["depth_roi"].to(torch.float32) for b in batch], dim=0)     # [B,1,7,7]
    depth_stats = torch.stack([b["depth_stats"].to(torch.float32) for b in batch], dim=0) # [B,8]
    labels = torch.stack([b["label"] for b in batch], dim=0)                               # [B]
    return roi_feat, mask_roi, depth_roi, depth_stats, labels


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for roi_feat, mask_roi, depth_roi, depth_stats, y in dl:
        roi_feat = roi_feat.to(device)
        mask_roi = mask_roi.to(device)
        depth_roi = depth_roi.to(device)
        depth_stats = depth_stats.to(device)
        y = y.to(device)

        logits = model(roi_feat, mask_roi, depth_roi, depth_stats)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct) / float(total) if total > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True, help="Dir created by cache_food101_instances.py")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_ckpt", type=str, default="fusion_head_food101.pt")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    train_index = cache_dir / "index_train.jsonl"
    test_index = cache_dir / "index_test.jsonl"
    assert train_index.exists(), f"Missing {train_index}"
    assert test_index.exists(), f"Missing {test_index}"

    ds_tr = CachedInstancesDataset(str(train_index))
    ds_te = CachedInstancesDataset(str(test_index))

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    # infer channels C from first item
    sample = torch.load(ds_tr.items[0]["cache_path"], map_location="cpu")
    C = int(sample["roi_feat"].shape[0])

    model = InstanceAwareFusionHead(in_channels=C, num_classes=101).to(args.device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    log_every = 100  # steps

    global_step = 0

    best = 0.0
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep}/{args.epochs}")
        running = 0.0

        for step, (roi_feat, mask_roi, depth_roi, depth_stats, y) in enumerate(pbar, start=1):
            roi_feat = roi_feat.to(args.device)
            mask_roi = mask_roi.to(args.device)
            depth_roi = depth_roi.to(args.device)
            depth_stats = depth_stats.to(args.device)
            y = y.to(args.device)

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

        acc = evaluate(model, dl_te, args.device)
        print(f"epoch {ep}: test_acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_channels": C,
                    "num_classes": 101,
                },
                args.out_ckpt,
            )
            print(f"saved best checkpoint to {args.out_ckpt} (acc={best:.4f})")

    print(f"done. best_test_acc={best:.4f}")


if __name__ == "__main__":
    main()
