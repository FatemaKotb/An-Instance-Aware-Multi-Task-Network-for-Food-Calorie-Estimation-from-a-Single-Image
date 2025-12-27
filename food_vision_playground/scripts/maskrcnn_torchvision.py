from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

from scripts.dtypes import InstanceSegmentationResult


class MaskRCNNTorchVisionBlock:
    """
    TorchVision Mask R-CNN wrapper as a block for clean pipeline instantiation.

    Notes:
    - COCO Mask R-CNN won't label 'pizza', 'rice', etc. (no food classes in COCO).
    - It can still produce useful masks for plates/bowls/objects.
    - Later you can swap to a food-finetuned checkpoint (MMDetection, etc.).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = self._build()

    def _build(self):
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        model = model.to(self.device).eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        return model, preprocess

    @torch.inference_mode()
    def predict_instances(self, img_rgb_uint8: np.ndarray, score_thresh: float = 0.5) -> InstanceSegmentationResult:
        """Run instance segmentation and return masks/boxes/scores/labels."""
        img = Image.fromarray(img_rgb_uint8)
        x = self.preprocess(img).to(self.device)  # [C,H,W]

        out = self.model([x])[0]
        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= score_thresh

        boxes = out["boxes"].detach().cpu().numpy()[keep]           # [N,4]
        class_ids = out["labels"].detach().cpu().numpy()[keep]      # [N]
        scrs = scores[keep]                                         # [N]
        masks = out["masks"].detach().cpu().numpy()[keep, 0] >= 0.5 # [N,H,W] bool

        return InstanceSegmentationResult(
            boxes_xyxy=boxes,
            scores=scrs,
            class_ids=class_ids,
            masks=masks,
        )

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray, score_thresh: float = 0.5) -> InstanceSegmentationResult:
        return self.predict_instances(img_rgb_uint8, score_thresh=score_thresh)

    def save_masks(self, out_dir: Path, img_rgb_uint8: np.ndarray, instance_outputs, topk: int = 5) -> None:
        # overlay image
        overlay = img_rgb_uint8.copy().astype(np.float32)

        # deterministic per-instance colors (RGB), repeated if > len(colors)
        colors = np.array([
            [255,   0,   0],  # red
            [  0, 255,   0],  # green
            [  0,   0, 255],  # blue
            [255, 255,   0],  # yellow
            [255,   0, 255],  # magenta
            [  0, 255, 255],  # cyan
            [255, 128,   0],  # orange
            [128,   0, 255],  # purple
            [  0, 128, 255],  # sky
            [128, 255,   0],  # lime
        ], dtype=np.float32)

        alpha = 0.45  # how strong the color overlay is

        for i, inst in enumerate(instance_outputs[:topk]):
            m = inst.mask.astype(bool)

            # save individual mask
            mask_u8 = (m.astype(np.uint8) * 255)
            Image.fromarray(mask_u8).save(out_dir / f"mask_{i:03d}.png")

            # apply colored overlay
            color = colors[i % len(colors)]  # [3]
            overlay[m] = (1.0 - alpha) * overlay[m] + alpha * color

        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_dir / "masks_overlay.png")
