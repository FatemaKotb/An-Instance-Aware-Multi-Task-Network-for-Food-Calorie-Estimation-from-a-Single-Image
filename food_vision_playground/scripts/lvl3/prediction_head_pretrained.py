from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from scripts.dtypes import FusionOutput, PredictionOutput


# Food-101 class names (canonical order used by many datasets)
FOOD101_CLASSES: List[str] = [
    "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare","beet_salad","beignets","bibimbap","bread_pudding",
    "breakfast_burrito","bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake","ceviche","cheese_plate","cheesecake",
    "chicken_curry","chicken_quesadilla","chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder","club_sandwich",
    "crab_cakes","creme_brulee","croque_madame","cup_cakes","deviled_eggs","donuts","dumplings","edamame","eggs_benedict",
    "escargots","falafel","filet_mignon","fish_and_chips","foie_gras","french_fries","french_onion_soup","french_toast",
    "fried_calamari","fried_rice","frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich","grilled_salmon",
    "guacamole","gyoza","hamburger","hot_and_sour_soup","hot_dog","huevos_rancheros","hummus","ice_cream","lasagna","lobster_bisque",
    "lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup","mussels","nachos","omelette","onion_rings","oysters",
    "pad_thai","paella","pancakes","panna_cotta","peking_duck","pho","pizza","pork_chop","poutine","prime_rib","pulled_pork_sandwich",
    "ramen","ravioli","red_velvet_cake","risotto","samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits",
    "spaghetti_bolognese","spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake","sushi","tacos","takoyaki","tiramisu",
    "tuna_tartare","waffles",
]

PROMPT_TEMPLATES: List[str] = [
    "a photo of {}",
    "a close-up photo of {}",
    "a plate of {}",
    "food: {}",
]

@dataclass
class _CLIPBundle:
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    text_features: torch.Tensor  # [K, D] on device


class PredictionHeadPretrainedCLIP:
    """
    Prediction head that uses a *pretrained* CLIP model (zero-shot) over Food-101 labels.

    This bypasses fusion for classification (by design), but keeps the pipeline interface stable.
    It predicts:
      - food_class_id
      - food_conf (max softmax prob)
      - food_logits (softmax logits over Food-101 classes)
      - portion (stub = 1.0 for now)

    Dependencies:
      pip install open_clip_torch

    Notes:
      - Uses box crop (and optionally mask) to focus on the instance.
      - You can later replace this with a trained head that consumes FusionOutput.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        class_names: Optional[List[str]] = None,
        prompt_templates: Optional[List[str]] = None,
        apply_mask: bool = True,
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_names = class_names or FOOD101_CLASSES
        self.prompt_templates = prompt_templates or PROMPT_TEMPLATES
        self.apply_mask = bool(apply_mask)

        self._bundle = self._load_clip()

    def _load_clip(self) -> _CLIPBundle:
        try:
            import open_clip
        except Exception as e:
            raise ImportError(
                "open_clip_torch is required for PredictionHeadPretrainedCLIP. "
                "Install with: pip install open_clip_torch"
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)

        model = model.to(self.device).eval()

        # Precompute text embeddings for all classes with prompt ensembling
        with torch.inference_mode():
            all_text_feats = []
            for tmpl in self.prompt_templates:
                prompts = [tmpl.format(name.replace('_', ' ')) for name in self.class_names]
                text = tokenizer(prompts).to(self.device)              # [K, ...]
                tf = model.encode_text(text)                           # [K, D]
                tf = tf / tf.norm(dim=-1, keepdim=True)                # normalize per prompt
                all_text_feats.append(tf)

            # [T, K, D] -> average over prompts -> [K, D]
            text_features = torch.stack(all_text_feats, dim=0).mean(dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return _CLIPBundle(model=model, preprocess=preprocess, tokenizer=tokenizer, text_features=text_features)

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
        # Crop instance
        crop = self._crop_instance(img_rgb_uint8, box_xyxy, mask_hw)
        pil = Image.fromarray(crop)

        # Preprocess for CLIP
        x = self._bundle.preprocess(pil).unsqueeze(0).to(self.device)  # [1,3,224,224] typically

        # Image features
        img_features = self._bundle.model.encode_image(x)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)  # [1,D]

        # Similarity -> logits
        logits = (img_features @ self._bundle.text_features.T).squeeze(0)  # [K]
        probs = torch.softmax(logits, dim=0)

        conf, cls = torch.max(probs, dim=0)

        cls_id = int(cls.item())
        cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else None

        topk = 5
        k = min(topk, probs.numel())
        vals, idx = torch.topk(probs, k=k)
        top5_foods = [(self.class_names[int(i)], float(v)) for v, i in zip(vals.detach().cpu(), idx.detach().cpu())]

        return PredictionOutput(
            food_logits=probs.detach().cpu(),
            food_class_id=cls_id,
            food_class_name=cls_name,
            food_conf=float(conf.item()),
            portion=1.0,
            top5_foods=top5_foods,
        )
