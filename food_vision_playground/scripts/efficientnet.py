from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
import timm
from PIL import Image

# timm==0.6.x API (works with your pinned timm)
from timm.data import resolve_data_config, create_transform

from scripts.dtypes import BackboneOutput, ClassificationResult, Mode


class EfficientNetBlock:
    """
    Unified EfficientNet block.

    mode="cls":
      - Loads classification model (with head)
      - Forward returns logits [B, num_classes]

    mode="backbone":
      - Loads features-only backbone (no classifier head)
      - Forward returns list of feature maps (multi-scale), controlled by out_indices
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
        self.out_indices = out_indices

        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self):
        if self.mode == "cls":
            model = timm.create_model(self.model_name, pretrained=True)
        elif self.mode == "backbone":
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                features_only=True,
                out_indices=self.out_indices,
            )
        else:
            raise ValueError(f"Unknown mode={self.mode!r}. Expected 'cls' or 'backbone'.")

        model = model.to(self.device).eval()
        return model

    def _build_transform(self):
        # timm<=0.6 uses resolve_data_config/create_transform
        cfg = resolve_data_config({}, model=self.model)
        transform = create_transform(**cfg)
        return transform

    @torch.inference_mode()
    def classify_image(self, img_rgb_uint8: np.ndarray, topk: int = 5, return_logits: bool = False) -> ClassificationResult:
        """Run classification on RGB uint8 image and return top-k indices + probs."""
        if self.mode != "cls":
            raise ValueError("EfficientNetBlock is not in classification mode. Use mode='cls'.")

        img = Image.fromarray(img_rgb_uint8)
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        vals, idx = torch.topk(probs, k=topk)
        return ClassificationResult(
            topk_indices=idx.detach().cpu().tolist(),
            topk_probs=vals.detach().cpu().tolist(),
            logits=logits.detach().cpu() if return_logits else None,
        )

    @torch.inference_mode()
    def extract_backbone_features(self, img_rgb_uint8: np.ndarray) -> BackboneOutput:
        """
        Input: RGB uint8 image [H,W,3]
        Output: list of feature maps from the backbone
        """
        if self.mode != "backbone":
            raise ValueError("EfficientNetBlock is not in backbone mode. Use mode='backbone'.")

        img = Image.fromarray(img_rgb_uint8)
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1,3,H,W]
        feats = self.model(x)  # list[Tensor], each [1,C,H,W]
        return BackboneOutput(features=feats)

    @torch.inference_mode()
    def __call__(
        self,
        img_rgb_uint8: np.ndarray,
        topk: int = 5,
        return_logits: bool = False,
    ) -> Union[ClassificationResult, BackboneOutput]:
        """Dispatch based on mode."""
        if self.mode == "cls":
            return self.classify_image(img_rgb_uint8, topk=topk, return_logits=return_logits)
        if self.mode == "backbone":
            return self.extract_backbone_features(img_rgb_uint8)
        raise ValueError(f"Unknown mode={self.mode!r}. Expected 'cls' or 'backbone'.")

    @staticmethod
    def describe_features(features: List[torch.Tensor]) -> None:
        """Print feature shapes for sanity checking."""
        for i, f in enumerate(features):
            print(f"F{i}: shape={tuple(f.shape)} dtype={f.dtype} device={f.device}")

    @staticmethod
    def load_imagenet_labels():
        """
        Lightweight ImageNet label loading.
        If you prefer, replace with your own mapping.
        """
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url) as f:
            labels = f.read().decode("utf-8").splitlines()
        return labels
