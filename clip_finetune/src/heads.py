from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .clip_backbone import CLIPBundle, load_openclip, encode_images

PROMPT_TEMPLATES: List[str] = [
    "a photo of {}",
    "a close-up photo of {}",
    "a plate of {}",
    "food: {}",
]

def crop_instance(img_rgb_uint8: np.ndarray, box_xyxy: np.ndarray, mask_hw: np.ndarray, apply_mask: bool = True) -> np.ndarray:
    # Based on the provided reference head's _crop_instance logic.
    H, W = img_rgb_uint8.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    x1 = int(max(0, np.floor(x1)))
    y1 = int(max(0, np.floor(y1)))
    x2 = int(min(W, np.ceil(x2)))
    y2 = int(min(H, np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return img_rgb_uint8
    crop = img_rgb_uint8[y1:y2, x1:x2].copy()
    if apply_mask:
        m = mask_hw[y1:y2, x1:x2].astype(bool)
        crop[~m] = 0
    return crop

@dataclass
class ZeroShotBundle:
    clip: CLIPBundle
    text_features: torch.Tensor  # [K,D] on device

class CLIPZeroShotHead:
    def __init__(
        self,
        class_names: Sequence[str],
        device: str = "cuda",
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        prompt_templates: Optional[Sequence[str]] = None,
    ):
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_names = list(class_names)
        self.prompt_templates = list(prompt_templates) if prompt_templates is not None else list(PROMPT_TEMPLATES)
        self.bundle = self._build()

    def _build(self) -> ZeroShotBundle:
        clip = load_openclip(self.model_name, self.pretrained, self.device)
        model = clip.model
        with torch.inference_mode():
            all_text_feats = []
            for tmpl in self.prompt_templates:
                prompts = [tmpl.format(name.replace('_', ' ')) for name in self.class_names]
                text = clip.tokenizer(prompts).to(self.device)
                tf = model.encode_text(text)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                all_text_feats.append(tf)
            text_features = torch.stack(all_text_feats, dim=0).mean(dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return ZeroShotBundle(clip=clip, text_features=text_features)

    @torch.inference_mode()
    def predict_pil(self, pil: Image.Image, topk: int = 5) -> Dict[str, Any]:
        x = self.bundle.clip.preprocess(pil).unsqueeze(0).to(self.device)
        img_features = encode_images(self.bundle.clip.model, x)  # [1,D]
        logits = (img_features @ self.bundle.text_features.T).squeeze(0)  # [K]
        probs = torch.softmax(logits, dim=0)
        conf, cls = torch.max(probs, dim=0)
        k = min(topk, probs.numel())
        vals, idx = torch.topk(probs, k=k)
        topk_list = [(self.class_names[int(i)], float(v)) for v, i in zip(vals.detach().cpu(), idx.detach().cpu())]
        return {
            "probs": probs.detach().cpu(),
            "class_id": int(cls.item()),
            "class_name": self.class_names[int(cls.item())],
            "conf": float(conf.item()),
            "topk": topk_list,
        }

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray, box_xyxy: np.ndarray, mask_hw: np.ndarray, topk: int = 5, apply_mask: bool = True):
        crop = crop_instance(img_rgb_uint8, box_xyxy, mask_hw, apply_mask=apply_mask)
        return self.predict_pil(Image.fromarray(crop), topk=topk)

class CLIPLinearHead(torch.nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        device: str = "cuda",
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_names = list(class_names)

        self.clip = load_openclip(model_name, pretrained, self.device)
        self.clip.model.eval()
        for p in self.clip.model.parameters():
            p.requires_grad_(False)

        # Infer embedding dim
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            try:
                emb_dim = self.clip.model.encode_image(dummy).shape[-1]
            except Exception:
                emb_dim = 512

        self.classifier = torch.nn.Linear(emb_dim, len(self.class_names)).to(self.device)
        self.eval()

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.classifier.load_state_dict(ckpt["classifier_state_dict"], strict=True)
        if "class_names" in ckpt:
            self.class_names = list(ckpt["class_names"])

    @torch.inference_mode()
    def encode_pil(self, pil: Image.Image) -> torch.Tensor:
        x = self.clip.preprocess(pil).unsqueeze(0).to(self.device)
        z = encode_images(self.clip.model, x).squeeze(0)  # [D]
        return z

    @torch.inference_mode()
    def predict_pil(self, pil: Image.Image, topk: int = 5) -> Dict[str, Any]:
        z = self.encode_pil(pil)
        logits = self.classifier(z)
        probs = torch.softmax(logits, dim=0)
        conf, cls = torch.max(probs, dim=0)
        k = min(topk, probs.numel())
        vals, idx = torch.topk(probs, k=k)
        topk_list = [(self.class_names[int(i)], float(v)) for v, i in zip(vals.detach().cpu(), idx.detach().cpu())]
        return {
            "probs": probs.detach().cpu(),
            "class_id": int(cls.item()),
            "class_name": self.class_names[int(cls.item())],
            "conf": float(conf.item()),
            "topk": topk_list,
        }

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray, box_xyxy: np.ndarray, mask_hw: np.ndarray, topk: int = 5, apply_mask: bool = True):
        crop = crop_instance(img_rgb_uint8, box_xyxy, mask_hw, apply_mask=apply_mask)
        return self.predict_pil(Image.fromarray(crop), topk=topk)
