from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from scripts.dtypes import FusionOutput, PredictionOutput


@dataclass
class _FinetunedCLIPBundle:
    model: torch.nn.Module
    preprocess: object
    head: torch.nn.Module
    class_names: List[str]


class PredictionHeadFinetunedCLIPLinear:
    """
    Prediction head that uses:
      - a pretrained OpenCLIP image encoder (frozen)
      - a *fine-tuned linear classifier* trained on CLIP embeddings

    Checkpoint format expected (from your Kaggle training script):
      {
        "classifier_state_dict": ...,
        "class_names": [...],
        "clip_model_name": "...",
        "clip_pretrained": "...",
        ...
      }

    It predicts:
      - food_class_id
      - food_conf (max softmax prob)
      - food_logits (softmax probs over your Egyptian classes)
      - top5_foods
      - portion (stub = 1.0)

    Notes:
      - Uses box crop (and optionally mask) to focus on the instance.
      - Fusion is ignored by design (keeps interface stable).
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        apply_mask: bool = True,
        # Optional overrides (if you want to force a specific backbone);
        # by default we read from checkpoint.
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
    ):
        self.device = device
        self.ckpt_path = ckpt_path
        self.apply_mask = bool(apply_mask)
        self.model_name_override = model_name
        self.pretrained_override = pretrained

        self._bundle = self._load_finetuned()

    def _load_finetuned(self) -> _FinetunedCLIPBundle:
        # --- load checkpoint ---
        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        if "classifier_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing key 'classifier_state_dict': {self.ckpt_path}")
        if "class_names" not in ckpt:
            raise ValueError(f"Checkpoint missing key 'class_names': {self.ckpt_path}")

        class_names = list(ckpt["class_names"])
        num_classes = len(class_names)
        if num_classes <= 1:
            raise ValueError(f"Expected >=2 classes in checkpoint, got {num_classes}")

        # Determine CLIP backbone to load
        model_name = self.model_name_override or ckpt.get("clip_model_name") or "ViT-B-32-quickgelu"
        pretrained = self.pretrained_override or ckpt.get("clip_pretrained") or "openai"

        # --- load open_clip ---
        try:
            import open_clip
        except Exception as e:
            raise ImportError(
                "open_clip_torch is required for PredictionHeadFinetunedCLIPLinear. "
                "Install with: pip install open_clip_torch"
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Infer embedding dim from classifier weights if possible (most reliable)
        state = ckpt["classifier_state_dict"]
        # Typical Linear keys: "weight" [K, D], "bias" [K]
        if "weight" in state and hasattr(state["weight"], "shape"):
            emb_dim = int(state["weight"].shape[1])
        else:
            # Fallback to probing model output dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                emb_dim = int(model.encode_image(dummy).shape[-1])

        head = torch.nn.Linear(emb_dim, num_classes).to(self.device)
        # Load classifier weights
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Not fatal, but helpful if someone saved different key names.
            # You can tighten this to strict=True if you want.
            pass
        head.eval()

        return _FinetunedCLIPBundle(
            model=model,
            preprocess=preprocess,
            head=head,
            class_names=class_names,
        )

    def _crop_instance(self, img_rgb_uint8: np.ndarray, box_xyxy: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
        H, W = img_rgb_uint8.shape[:2]
        x1, y1, x2, y2 = box_xyxy.astype(np.float32)

        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(W, np.ceil(x2)))
        y2 = int(min(H, np.ceil(y2)))

        if x2 <= x1 or y2 <= y1:
            return img_rgb_uint8  # fallback

        crop = img_rgb_uint8[y1:y2, x1:x2].copy()

        if self.apply_mask:
            m = mask_hw[y1:y2, x1:x2].astype(bool)
            # black background outside mask
            crop[~m] = 0

        return crop

    @torch.inference_mode()
    def __call__(
        self,
        img_rgb_uint8: np.ndarray,
        box_xyxy: np.ndarray,
        mask_hw: np.ndarray,
        fusion: Optional[FusionOutput] = None,
    ) -> PredictionOutput:
        # 1) Crop instance
        crop = self._crop_instance(img_rgb_uint8, box_xyxy, mask_hw)
        pil = Image.fromarray(crop)

        # 2) Preprocess for CLIP
        x = self._bundle.preprocess(pil).unsqueeze(0).to(self.device)  # [1,3,224,224]

        # 3) Encode image -> normalized embedding
        img_features = self._bundle.model.encode_image(x)  # [1,D]
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # 4) Linear head -> logits over your Egyptian classes
        logits = self._bundle.head(img_features).squeeze(0)  # [K]
        probs = torch.softmax(logits, dim=0)                 # [K]

        conf, cls = torch.max(probs, dim=0)
        cls_id = int(cls.item())
        cls_name = self._bundle.class_names[cls_id] if 0 <= cls_id < len(self._bundle.class_names) else "unknown"

        k = min(5, probs.numel())
        vals, idx = torch.topk(probs, k=k)
        top5_foods: List[Tuple[str, float]] = [
            (self._bundle.class_names[int(i)], float(v))
            for v, i in zip(vals.detach().cpu(), idx.detach().cpu())
        ]

        return PredictionOutput(
            food_logits=probs.detach().cpu(),
            food_class_id=cls_id,
            food_class_name=cls_name,
            food_conf=float(conf.item()),
            portion=1.0,
            top5_foods=top5_foods,
        )
