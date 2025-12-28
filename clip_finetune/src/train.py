from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .config import TrainConfig
from .clip_backbone import load_openclip, encode_images
from .data import build_dataloaders


def _cfg_to_dict(cfg: TrainConfig) -> Dict:
    if hasattr(cfg, "to_dict") and callable(getattr(cfg, "to_dict")):
        return cfg.to_dict()
    try:
        from dataclasses import asdict
        return asdict(cfg)
    except Exception:
        return {"repr": repr(cfg)}


def train_linear_head(cfg: TrainConfig) -> Tuple[str, Dict[str, float]]:
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    clip = load_openclip(cfg.model_name, cfg.pretrained, device)
    clip.model.eval()
    for p in clip.model.parameters():
        p.requires_grad_(False)

    dls = build_dataloaders(
        data_root=cfg.data_root,
        preprocess=clip.preprocess,
        artifacts_dir=cfg.artifacts_dir,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    num_classes = len(dls.class_names)

    # Infer embedding dim (safe)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        try:
            emb_dim = clip.model.encode_image(dummy).shape[-1]
        except Exception:
            emb_dim = 512

    head = nn.Linear(emb_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=bool(cfg.use_amp and device == "cuda"))

    best_val = 0.0
    best_state = None

    for epoch in range(cfg.epochs):
        head.train()
        running_loss = 0.0
        n = 0

        for x, y in tqdm(dls.train, desc=f"train e{epoch+1}/{cfg.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Compute frozen CLIP embeddings WITHOUT tracking grads for CLIP
            with torch.no_grad():
                z = encode_images(clip.model, x)  # [B, D]

            # HARD FIX: convert to a normal tensor (not inference-mode) for autograd through `head`
            z = z.detach().clone()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg.use_amp and device == "cuda")):
                logits = head(z)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item()) * x.shape[0]
            n += x.shape[0]

        train_loss = running_loss / max(1, n)

        # ---- Validation ----
        head.eval()
        correct = 0
        correct5 = 0
        total = 0

        with torch.no_grad():
            for x, y in tqdm(dls.val, desc=f"val   e{epoch+1}/{cfg.epochs}"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                z = encode_images(clip.model, x)
                z = z.detach().clone()
                logits = head(z)

                total += int(y.numel())

                pred1 = torch.argmax(logits, dim=1)
                correct += int((pred1 == y).sum().item())

                k = min(5, logits.shape[1])
                topk = torch.topk(logits, k=k, dim=1).indices
                correct5 += int((topk == y[:, None]).any(dim=1).sum().item())

        val_acc = correct / max(1, total)
        val_top5 = correct5 / max(1, total)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_top5={val_top5:.4f} | best={best_val:.4f}"
        )

    ckpt_path = cfg.ckpt_path()
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "classifier_state_dict": best_state if best_state is not None else head.state_dict(),
            "class_names": dls.class_names,
            "clip_model_name": cfg.model_name,
            "clip_pretrained": cfg.pretrained,
            "train_config": _cfg_to_dict(cfg),
            "best_val_acc": float(best_val),
        },
        ckpt_path,
    )

    return ckpt_path, {"best_val_acc": float(best_val)}
