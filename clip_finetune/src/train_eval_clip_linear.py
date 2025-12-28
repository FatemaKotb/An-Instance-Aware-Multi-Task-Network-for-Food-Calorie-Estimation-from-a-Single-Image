from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms

# -------------------------
# Dataset / DataLoaders
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_classes(data_root: str) -> List[str]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    classes = [p.name for p in root.iterdir() if p.is_dir()]
    classes.sort()
    if not classes:
        raise ValueError(f"No class folders found under: {root}")
    return classes


def discover_items(data_root: str, class_names: Sequence[str]) -> List[Tuple[str, int]]:
    root = Path(data_root)
    items: List[Tuple[str, int]] = []
    for cid, cname in enumerate(class_names):
        cdir = root / cname
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((str(p), cid))
    if not items:
        raise ValueError(f"No images found under: {root}")
    return items


def build_splits(
    items: Sequence[Tuple[str, int]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[int]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Invalid split ratios.")
    n = len(items)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train : n_train + n_val].tolist(),
        "test": idx[n_train + n_val :].tolist(),
    }


def save_split_files(split_dir: str, items: Sequence[Tuple[str, int]], splits: Dict[str, List[int]]) -> None:
    out = Path(split_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, split_idx in splits.items():
        fp = out / f"{split_name}.txt"
        with fp.open("w", encoding="utf-8") as f:
            for i in split_idx:
                f.write(items[i][0] + "\n")


class FolderDishDataset(Dataset):
    def __init__(self, paths: Sequence[str], path_to_class_id: Dict[str, int], transform):
        self.paths = list(paths)
        self.path_to_class_id = dict(path_to_class_id)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = int(self.path_to_class_id[path])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


def _infer_openclip_size(preprocess) -> int:
    # Try to infer the expected input size from OpenCLIP preprocess.
    size = 224
    if hasattr(preprocess, "transforms"):
        for t in preprocess.transforms:
            if isinstance(t, transforms.CenterCrop):
                if isinstance(t.size, (tuple, list)):
                    size = int(t.size[0])
                else:
                    size = int(t.size)
                break
    return size


def _extract_openclip_normalize(preprocess) -> transforms.Normalize:
    if not hasattr(preprocess, "transforms"):
        raise ValueError("Expected OpenCLIP preprocess to be a torchvision.transforms.Compose with .transforms.")
    for t in preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            return t
    raise ValueError("Could not find torchvision.transforms.Normalize inside OpenCLIP preprocess.")


def build_train_eval_transforms(openclip_preprocess):
    """
    Strong training augmentations + OpenCLIP normalization.

    - Train: RandomResizedCrop + HFlip + ColorJitter + ToTensor + Normalize
    - Eval:  OpenCLIP preprocess as-is
    """
    size = _infer_openclip_size(openclip_preprocess)
    norm = _extract_openclip_normalize(openclip_preprocess)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size, scale=(0.70, 1.00), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
            transforms.ToTensor(),
            norm,
        ]
    )

    eval_tf = openclip_preprocess
    return train_tf, eval_tf


def build_dataloaders(
    data_root: str,
    train_preprocess,
    eval_preprocess,
    artifacts_dir: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, List[int]]]:
    class_names = discover_classes(data_root)
    items = discover_items(data_root, class_names)
    splits = build_splits(items, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    split_dir = str(Path(artifacts_dir) / "splits")
    save_split_files(split_dir, items, splits)

    path_to_class_id = {p: cid for (p, cid) in items}
    train_paths = [items[i][0] for i in splits["train"]]
    val_paths = [items[i][0] for i in splits["val"]]
    test_paths = [items[i][0] for i in splits["test"]]

    ds_train = FolderDishDataset(train_paths, path_to_class_id, train_preprocess)
    ds_val = FolderDishDataset(val_paths, path_to_class_id, eval_preprocess)
    ds_test = FolderDishDataset(test_paths, path_to_class_id, eval_preprocess)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dl_train, dl_val, dl_test, class_names, splits


# -------------------------
# CLIP helpers
# -------------------------

PROMPT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a plate of {}",
    "food: {}",
]


def load_openclip(model_name: str, pretrained: str, device: str):
    import open_clip  # open_clip_torch

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def build_text_features(
    model: nn.Module,
    tokenizer,
    class_names: Sequence[str],
    device: str,
    templates: Sequence[str],
) -> torch.Tensor:
    all_text_feats = []
    for tmpl in templates:
        prompts = [tmpl.format(c.replace("_", " ")) for c in class_names]
        tok = tokenizer(prompts).to(device)
        tf = model.encode_text(tok)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        all_text_feats.append(tf)
    text_features = torch.stack(all_text_feats, dim=0).mean(dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


# -------------------------
# Training / Eval
# -------------------------

@dataclass
class TrainArgs:
    data_root: str
    artifacts_dir: str
    device: str
    model_name: str
    pretrained: str
    seed: int
    val_ratio: float
    test_ratio: float
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    epochs: int
    use_amp: bool
    ckpt_name: str

    # augmentation controls (optional)
    aug_scale_min: float = 0.70
    aug_scale_max: float = 1.00
    aug_ratio_min: float = 0.85
    aug_ratio_max: float = 1.15
    cj_brightness: float = 0.20
    cj_contrast: float = 0.20
    cj_saturation: float = 0.20
    cj_hue: float = 0.05
    hflip_p: float = 0.50


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    return int((topk == y[:, None]).any(dim=1).sum().item())


def train_eval(args: TrainArgs) -> Dict[str, float]:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load CLIP
    model, openclip_preprocess, tokenizer = load_openclip(args.model_name, args.pretrained, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build train/eval transforms (strong train aug)
    size = _infer_openclip_size(openclip_preprocess)
    norm = _extract_openclip_normalize(openclip_preprocess)

    train_preprocess = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=size,
                scale=(args.aug_scale_min, args.aug_scale_max),
                ratio=(args.aug_ratio_min, args.aug_ratio_max),
            ),
            transforms.RandomHorizontalFlip(p=args.hflip_p),
            transforms.ColorJitter(
                brightness=args.cj_brightness,
                contrast=args.cj_contrast,
                saturation=args.cj_saturation,
                hue=args.cj_hue,
            ),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_preprocess = openclip_preprocess

    # Data
    dl_train, dl_val, dl_test, class_names, splits = build_dataloaders(
        data_root=args.data_root,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    # Infer emb dim (safe)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        try:
            emb_dim = model.encode_image(dummy).shape[-1]
        except Exception:
            emb_dim = 512

    head = nn.Linear(emb_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=bool(args.use_amp and device == "cuda"))

    best_val_acc = 0.0
    best_state = None

    # -------- Train loop --------
    for epoch in range(1, args.epochs + 1):
        head.train()
        pbar = tqdm(dl_train, desc=f"train {epoch}/{args.epochs}", leave=True)
        running_loss = 0.0
        seen = 0

        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                z = encode_images(model, x)  # [B,D]
            # robust: make sure z is a normal tensor for autograd through head
            z = z.detach().clone()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.use_amp and device == "cuda")):
                logits = head(z)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

            pbar.set_postfix({"loss": f"{(running_loss/max(1,seen)):.4f}", "step": f"{step}/{len(dl_train)}"})

        train_loss = running_loss / max(1, seen)

        # -------- Val loop --------
        head.eval()
        correct1 = 0
        correct5 = 0
        total = 0

        vbar = tqdm(dl_val, desc=f"val   {epoch}/{args.epochs}", leave=False)
        with torch.no_grad():
            for step, (x, y) in enumerate(vbar, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                z = encode_images(model, x).detach().clone()
                logits = head(z)

                total += int(y.numel())
                correct1 += int((torch.argmax(logits, dim=1) == y).sum().item())
                correct5 += topk_correct(logits, y, 5)

                vbar.set_postfix(
                    {
                        "acc1": f"{(correct1/max(1,total)):.4f}",
                        "acc5": f"{(correct5/max(1,total)):.4f}",
                        "step": f"{step}/{len(dl_val)}",
                    }
                )

        val_acc1 = correct1 / max(1, total)

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_acc1={val_acc1:.4f} best_val_acc1={best_val_acc:.4f}"
        )

    # Save checkpoint
    ckpt_path = Path(args.artifacts_dir) / "ckpts" / args.ckpt_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "classifier_state_dict": best_state if best_state is not None else head.state_dict(),
            "class_names": class_names,
            "clip_model_name": args.model_name,
            "clip_pretrained": args.pretrained,
            "train_args": asdict(args),
            "best_val_acc1": float(best_val_acc),
            "splits": splits,
        },
        ckpt_path,
    )

    # -------- Test eval: trained head --------
    head.load_state_dict(best_state if best_state is not None else head.state_dict(), strict=True)
    head.eval()

    correct1 = 0
    correct5 = 0
    total = 0
    tbar = tqdm(dl_test, desc="test  linear", leave=False)
    with torch.no_grad():
        for step, (x, y) in enumerate(tbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = encode_images(model, x).detach().clone()
            logits = head(z)

            total += int(y.numel())
            correct1 += int((torch.argmax(logits, dim=1) == y).sum().item())
            correct5 += topk_correct(logits, y, 5)

            tbar.set_postfix(
                {"acc1": f"{(correct1/max(1,total)):.4f}", "acc5": f"{(correct5/max(1,total)):.4f}", "step": f"{step}/{len(dl_test)}"}
            )

    linear_top1 = correct1 / max(1, total)
    linear_top5 = correct5 / max(1, total)

    # -------- Test eval: zero-shot baseline --------
    text_features = build_text_features(model, tokenizer, class_names, device, PROMPT_TEMPLATES)
    zs_correct1 = 0
    zs_correct5 = 0
    zs_total = 0

    zbar = tqdm(dl_test, desc="test  zeroshot", leave=False)
    with torch.no_grad():
        for step, (x, y) in enumerate(zbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            img_features = encode_images(model, x)  # [B,D]
            logits = img_features @ text_features.T  # [B,K]

            zs_total += int(y.numel())
            zs_correct1 += int((torch.argmax(logits, dim=1) == y).sum().item())
            zs_correct5 += topk_correct(logits, y, 5)

            zbar.set_postfix(
                {"acc1": f"{(zs_correct1/max(1,zs_total)):.4f}", "acc5": f"{(zs_correct5/max(1,zs_total)):.4f}", "step": f"{step}/{len(dl_test)}"}
            )

    zeroshot_top1 = zs_correct1 / max(1, zs_total)
    zeroshot_top5 = zs_correct5 / max(1, zs_total)

    metrics = {
        "device_used": device,
        "num_classes": float(num_classes),
        "num_train": float(len(dl_train.dataset)),
        "num_val": float(len(dl_val.dataset)),
        "num_test": float(len(dl_test.dataset)),
        "best_val_acc1": float(best_val_acc),
        "test_linear_top1": float(linear_top1),
        "test_linear_top5": float(linear_top5),
        "test_zeroshot_top1": float(zeroshot_top1),
        "test_zeroshot_top5": float(zeroshot_top5),
        "ckpt_path": str(ckpt_path),
    }

    # Save metrics
    report_path = Path(args.artifacts_dir) / "reports" / "metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== FINAL METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Folder-per-class dataset root.")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model_name", type=str, default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--ckpt_name", type=str, default="egypt_clip_linear.pt")

    # optional augmentation knobs (you can ignore and keep defaults)
    p.add_argument("--aug_scale_min", type=float, default=0.70)
    p.add_argument("--aug_scale_max", type=float, default=1.00)
    p.add_argument("--aug_ratio_min", type=float, default=0.85)
    p.add_argument("--aug_ratio_max", type=float, default=1.15)
    p.add_argument("--cj_brightness", type=float, default=0.20)
    p.add_argument("--cj_contrast", type=float, default=0.20)
    p.add_argument("--cj_saturation", type=float, default=0.20)
    p.add_argument("--cj_hue", type=float, default=0.05)
    p.add_argument("--hflip_p", type=float, default=0.50)

    a = p.parse_args()
    return TrainArgs(
        data_root=a.data_root,
        artifacts_dir=a.artifacts_dir,
        device=a.device,
        model_name=a.model_name,
        pretrained=a.pretrained,
        seed=a.seed,
        val_ratio=a.val_ratio,
        test_ratio=a.test_ratio,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        epochs=a.epochs,
        use_amp=bool(a.use_amp),
        ckpt_name=a.ckpt_name,
        aug_scale_min=a.aug_scale_min,
        aug_scale_max=a.aug_scale_max,
        aug_ratio_min=a.aug_ratio_min,
        aug_ratio_max=a.aug_ratio_max,
        cj_brightness=a.cj_brightness,
        cj_contrast=a.cj_contrast,
        cj_saturation=a.cj_saturation,
        cj_hue=a.cj_hue,
        hflip_p=a.hflip_p,
    )


if __name__ == "__main__":
    args = parse_args()
    train_eval(args)
