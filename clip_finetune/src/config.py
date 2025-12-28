from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TrainConfig:
    # Data
    data_root: str
    artifacts_dir: str = "artifacts"
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # CLIP backbone (match your pipeline head)
    device: str = "cuda"
    model_name: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"

    # Training
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    use_amp: bool = True

    # Checkpoint
    ckpt_name: str = "egypt_clip_linear.pt"

    def ckpt_path(self) -> str:
        return str(Path(self.artifacts_dir) / "ckpts" / self.ckpt_name)

    def to_dict(self):
        return asdict(self)
