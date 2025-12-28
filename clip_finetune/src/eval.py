from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from .clip_backbone import load_openclip, encode_images
from .data import build_dataloaders
from .heads import CLIPZeroShotHead

@torch.inference_mode()
def eval_linear_ckpt(
    data_root: str,
    ckpt_path: str,
    artifacts_dir: str = "artifacts",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, float]:
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    class_names = list(ckpt["class_names"])
    model_name = ckpt.get("clip_model_name", "ViT-B-32-quickgelu")
    pretrained = ckpt.get("clip_pretrained", "openai")

    clip = load_openclip(model_name, pretrained, device)

    dls = build_dataloaders(
        data_root=data_root,
        preprocess=clip.preprocess,
        artifacts_dir=artifacts_dir,
        seed=ckpt.get("train_config", {}).get("seed", 42),
        val_ratio=ckpt.get("train_config", {}).get("val_ratio", 0.15),
        test_ratio=ckpt.get("train_config", {}).get("test_ratio", 0.15),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        try:
            emb_dim = clip.model.encode_image(dummy).shape[-1]
        except Exception:
            emb_dim = 512

    head = torch.nn.Linear(emb_dim, len(class_names)).to(device)
    head.load_state_dict(ckpt["classifier_state_dict"], strict=True)
    head.eval()
    clip.model.eval()

    correct1 = 0
    correct5 = 0
    total = 0

    for x, y in tqdm(dls.test, desc="eval linear"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = encode_images(clip.model, x)
        logits = head(z)
        total += int(y.numel())

        pred1 = torch.argmax(logits, dim=1)
        correct1 += int((pred1 == y).sum().item())

        k = min(5, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        correct5 += int((topk == y[:, None]).any(dim=1).sum().item())

    return {
        "linear_top1": correct1 / max(1, total),
        "linear_top5": correct5 / max(1, total),
        "num_test": float(total),
        "num_classes": float(len(class_names)),
    }

@torch.inference_mode()
def eval_zeroshot(
    data_root: str,
    artifacts_dir: str,
    device: str,
    model_name: str,
    pretrained: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, float]:
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

    clip = load_openclip(model_name, pretrained, device)
    dls = build_dataloaders(
        data_root=data_root,
        preprocess=clip.preprocess,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    zs = CLIPZeroShotHead(class_names=dls.class_names, device=device, model_name=model_name, pretrained=pretrained)

    correct1 = 0
    correct5 = 0
    total = 0

    for x, y in tqdm(dls.test, desc="eval zeroshot"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        img_features = encode_images(zs.bundle.clip.model, x)
        logits = img_features @ zs.bundle.text_features.T

        total += int(y.numel())
        pred1 = torch.argmax(logits, dim=1)
        correct1 += int((pred1 == y).sum().item())

        k = min(5, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        correct5 += int((topk == y[:, None]).any(dim=1).sum().item())

    return {
        "zeroshot_top1": correct1 / max(1, total),
        "zeroshot_top5": correct5 / max(1, total),
        "num_test": float(total),
        "num_classes": float(len(dls.class_names)),
    }

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
