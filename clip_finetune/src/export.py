from __future__ import annotations
from pathlib import Path
import shutil

def export_checkpoint(src_ckpt: str, dst_path: str) -> str:
    src = Path(src_ckpt)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)
