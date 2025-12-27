from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config

from scripts.dtypes import BackboneOutput, ClassificationResult, Mode


class EfficientNetBlock:
    """
    EfficientNet (timm) wrapped as a block.

    Supports two modes:
      - mode="cls": standard classification head (ImageNet)
      - mode="backbone": return multi-scale feature maps for downstream tasks

    This keeps the same "dataclasses + simple member functions" style used across the project.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s",
        device: str = "cuda",
        mode: Mode = "cls",
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
    ):
        self.model_name = model_name
        self.device = device
        self.mode = mode
        self.out_indices = tuple(int(x) for x in out_indices)

        self.model = self._create_model()
        self.transform = self._create_transform()

    def _create_model(self):
        """Create timm model and move it to device."""
        if self.mode == "backbone":
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                features_only=True,
                out_indices=self.out_indices,
            )
        else:
            model = timm.create_model(self.model_name, pretrained=True)

        model.eval().to(self.device)
        return model

    def _create_transform(self):
        """Create the correct preprocessing transform for this model."""
        cfg = resolve_data_config({}, model=self.model)
        xform = create_transform(**cfg)
        return xform

    @staticmethod
    def load_imagenet_labels() -> List[str]:
        """Lightweight ImageNet label loading (used only for display)."""
        import urllib.request

        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url) as f:
            labels = f.read().decode("utf-8").splitlines()
        return labels

    @torch.inference_mode()
    def classify_image(self, img_rgb_uint8: np.ndarray, topk: int = 5, return_logits: bool = False) -> ClassificationResult:
        """Run classification on RGB uint8 image and return top-k indices + probs."""
        assert self.mode == "cls", "EfficientNetBlock(mode='cls') required for classify_image"

        img = Image.fromarray(img_rgb_uint8)
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        vals, idx = torch.topk(probs, k=int(topk))

        return ClassificationResult(
            topk_indices=idx.detach().cpu().tolist(),
            topk_probs=vals.detach().cpu().tolist(),
            logits=logits.detach().cpu() if return_logits else None,
        )

    @torch.inference_mode()
    def extract_features(self, img_rgb_uint8: np.ndarray) -> BackboneOutput:
        """Return multi-scale feature maps for an RGB uint8 image."""
        assert self.mode == "backbone", "EfficientNetBlock(mode='backbone') required for extract_features"

        img = Image.fromarray(img_rgb_uint8)
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        # timm features_only model returns list of [B,C,H,W]
        feats = self.model(x)
        return BackboneOutput(features=feats)

    @staticmethod
    def describe_features(out: BackboneOutput) -> None:
        """Print feature map shapes (handy for sanity checks)."""
        for i, f in enumerate(out.features):
            print(f"Feature[{i}] shape={tuple(f.shape)} dtype={f.dtype} device={f.device}")

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray):
        if self.mode == "backbone":
            return self.extract_features(img_rgb_uint8)
        return self.classify_image(img_rgb_uint8)
