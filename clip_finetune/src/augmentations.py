from __future__ import annotations

from typing import Tuple

from torchvision import transforms


def build_train_eval_transforms(openclip_preprocess) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds:
      - train_preprocess: stronger augmentation + OpenCLIP normalize
      - eval_preprocess: the original OpenCLIP preprocess (no heavy aug)

    We *reuse* OpenCLIP's normalization to stay in-distribution.
    """

    # openclip_preprocess is usually a torchvision.transforms.Compose
    # We want to keep its final steps (ToTensor + Normalize).
    if not hasattr(openclip_preprocess, "transforms"):
        raise ValueError("Expected openclip_preprocess to have a .transforms list (torchvision Compose).")

    tlist = list(openclip_preprocess.transforms)

    # Extract Normalize (and keep ToTensor if present) from OpenCLIP preprocess
    norm = None
    to_tensor = None
    for t in tlist:
        if isinstance(t, transforms.Normalize):
            norm = t
        if isinstance(t, transforms.ToTensor):
            to_tensor = t

    if norm is None:
        raise ValueError("Could not find torchvision.transforms.Normalize inside OpenCLIP preprocess.")

    # Target image size: try to infer from Resize/CenterCrop in OpenCLIP preprocess
    # Fall back to 224.
    size = 224
    for t in tlist:
        if isinstance(t, transforms.CenterCrop):
            # CenterCrop(size) where size can be int or (h,w)
            if isinstance(t.size, (tuple, list)):
                size = int(t.size[0])
            else:
                size = int(t.size)
            break

    # Strong-ish, safe augmentations for food images
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(
            size=size,
            scale=(0.70, 1.00),          # crop scale range
            ratio=(0.85, 1.15),          # aspect ratio range
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
        ),
        transforms.ToTensor() if to_tensor is None else to_tensor,
        norm,
    ])

    # Validation / test: use the original OpenCLIP preprocessing
    eval_preprocess = openclip_preprocess

    return train_preprocess, eval_preprocess
