from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.dtypes import InstanceSegmentationResult


class MaskRCNNTorchVisionBlock:
    """
    TorchVision pretrained Mask R-CNN (COCO) wrapped as a block.

    Notes:
      - COCO labels won't include fine-grained food classes ("pizza", "rice", ...).
      - The masks are still useful for pipeline testing and later swapping in a food-finetuned checkpoint.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = self._load_maskrcnn(device=device)

    def _load_maskrcnn(self, device: str = "cuda"):
        """Load TorchVision pretrained Mask R-CNN (COCO)."""
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        return model, preprocess

    @torch.inference_mode()
    def predict(self, img_rgb_uint8: np.ndarray, score_thresh: float = 0.5) -> InstanceSegmentationResult:
        """
        Input: RGB uint8 [H,W,3]
        Output: InstanceSegmentationResult
        """
        img_pil = Image.fromarray(img_rgb_uint8)
        x = self.preprocess(img_pil).to(self.device)  # [C,H,W]

        out = self.model([x])[0]
        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= float(score_thresh)

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
        return self.predict(img_rgb_uint8, score_thresh=score_thresh)

    def save_masks(self, out_dir: Path, img_rgb_uint8: np.ndarray, instance_outputs, topk: int = 10) -> None:
        """
        Save per-instance binary masks and an overlay visualization.

        - mask_000.png, mask_001.png, ...: binary masks
        - masks_overlay.png: different color per instance
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        overlay = img_rgb_uint8.copy().astype(np.float32)

        # deterministic per-instance colors (RGB), repeated if > len(colors)
        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [255, 128, 0],
                [128, 0, 255],
                [0, 128, 255],
                [128, 255, 0],
            ],
            dtype=np.float32,
        )

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
