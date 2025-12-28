from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def discover_classes(data_root: str) -> List[str]:
    root = Path(data_root)
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

def build_splits(items: Sequence[Tuple[str, int]], seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15):
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
        "val": idx[n_train:n_train+n_val].tolist(),
        "test": idx[n_train+n_val:].tolist(),
    }

def save_split_files(split_dir: str, items: Sequence[Tuple[str, int]], splits):
    out = Path(split_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, split_idx in splits.items():
        fp = out / f"{split_name}.txt"
        with fp.open("w", encoding="utf-8") as f:
            for i in split_idx:
                f.write(items[i][0] + "\n")

class FolderDishDataset(Dataset):
    def __init__(self, paths: Sequence[str], path_to_class_id: Dict[str, int], transform: Optional[Callable] = None):
        self.paths = list(paths)
        self.path_to_class_id = dict(path_to_class_id)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        y = int(self.path_to_class_id[path])
        img = Image.open(path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        return x, y

@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: List[str]
    items: List[Tuple[str, int]]
    splits: dict

def build_dataloaders(
    data_root: str,
    preprocess: Callable,
    artifacts_dir: str = "artifacts",
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoaders:
    class_names = discover_classes(data_root)
    items = discover_items(data_root, class_names)
    splits = build_splits(items, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    split_dir = str(Path(artifacts_dir) / "splits")
    save_split_files(split_dir, items, splits)

    path_to_class_id = {p: cid for (p, cid) in items}
    train_paths = [items[i][0] for i in splits["train"]]
    val_paths = [items[i][0] for i in splits["val"]]
    test_paths = [items[i][0] for i in splits["test"]]

    ds_train = FolderDishDataset(train_paths, path_to_class_id, transform=preprocess)
    ds_val = FolderDishDataset(val_paths, path_to_class_id, transform=preprocess)
    ds_test = FolderDishDataset(test_paths, path_to_class_id, transform=preprocess)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return DataLoaders(dl_train, dl_val, dl_test, class_names, items, splits)
