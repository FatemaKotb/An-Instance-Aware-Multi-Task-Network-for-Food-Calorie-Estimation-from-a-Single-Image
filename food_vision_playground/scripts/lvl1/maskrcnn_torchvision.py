# maskrcnn_torchvision.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.dtypes import InstanceSegmentationResult


class MaskRCNNTorchVisionBlock:
    """
    TorchVision pretrained Mask R-CNN (COCO) wrapped as a block.

    What’s added here:
      1) Guaranteed removal of tiny masks (relative to image size), even if confident.
      2) Dedupe + “union mask” removal that prefers separate instances (A and B)
         over a merged mask (A+B).

    Tiny mask rule:
      - drop any mask with area < (min_mask_area_ratio * H * W)

    Union/duplicate removal:
      - drop near-identical duplicates by IoU
      - drop big “union” masks that mostly contain 2+ kept masks
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
    # Drop any mask that covers less than this fraction of the whole image.
    # Purpose: guarantees removal of tiny false positives even if confidence is high.
    min_mask_area_ratio: float = 0.20,

    # If two masks overlap so much that IoU >= this value, treat them as duplicates.
    # Purpose: removes repeated detections of the same dish (almost identical masks).
    iou_dup_thresh: float = 0.70,

    # Containment threshold: if ~97% of one mask lies inside another mask, treat as redundant.
    # Purpose: catches “subset/superset” cases and helps detect merged/union artifacts.
    contain_thresh: float = 0.97,

    # Area ratio used to label a larger containing mask as a likely "union" mask.
    # If big_mask_area >= union_area_ratio * small_mask_area (and containment is high),
    # the big mask is considered A+B (merged) and should be dropped in favor of separate masks.
    union_area_ratio: float = 1.00,
    ):
        self.device = device

        # tiny-mask filter (absolute vs image size)
        self.min_mask_area_ratio = float(min_mask_area_ratio)

        # dedupe / union removal
        self.iou_dup_thresh = float(iou_dup_thresh)
        self.contain_thresh = float(contain_thresh)
        self.union_area_ratio = float(union_area_ratio)

        self.model, self.preprocess = self._load_maskrcnn(device=device)

    def _load_maskrcnn(self, device: str = "cuda"):
        """Load TorchVision pretrained Mask R-CNN (COCO)."""
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        return model, preprocess

    # ----------------- Post-processing helpers -----------------

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU of 2 boolean masks [H,W]."""
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _containment(a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of mask a covered by mask b. a,b: bool [H,W]."""
        a_sum = a.sum()
        if a_sum == 0:
            return 0.0
        inter = np.logical_and(a, b).sum()
        return float(inter) / float(a_sum)

    def _filter_tiny_masks(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
        masks_bool: np.ndarray,
        *,
        H: int,
        W: int,
    ):
        """
        Guaranteed removal of tiny masks relative to image size.
        masks_bool: [N,H,W] bool
        """
        if masks_bool.shape[0] == 0:
            return boxes, class_ids, scores, masks_bool

        areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1).astype(np.float32)
        min_area_px = float(self.min_mask_area_ratio) * float(H * W)

        keep = areas >= min_area_px
        return boxes[keep], class_ids[keep], scores[keep], masks_bool[keep]

    def _dedupe_masks_prefer_separate(self, masks_bool: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Returns indices to keep (relative to current arrays).
        Prefers separate instances (A, B) over a merged union (A+B).
        """
        if masks_bool.shape[0] == 0:
            return np.array([], dtype=np.int64)

        mbool = masks_bool
        areas = mbool.reshape(mbool.shape[0], -1).sum(axis=1).astype(np.float32)

        order = np.argsort(scores)[::-1]  # high score first
        keep: list[int] = []

        # Pass 1: greedy keep with union protection
        for idx in order:
            m = mbool[idx]
            a = areas[idx]
            drop = False

            for k in keep:
                mk = mbool[k]
                ak = areas[k]

                # 1) near-identical duplicate => drop current
                iou = self._mask_iou(m, mk)
                if iou >= self.iou_dup_thresh:
                    drop = True
                    break

                # 2) containment checks
                c_m_in_k = self._containment(m, mk)   # current inside kept
                c_k_in_m = self._containment(mk, m)   # kept inside current

                # If current is a big union that contains an already-kept mask => drop current
                if c_k_in_m >= self.contain_thresh and a >= self.union_area_ratio * ak:
                    drop = True
                    break

                # If current is small and inside a kept big mask:
                # likely the kept mask is union/merge; keep current (don't drop).
                if c_m_in_k >= self.contain_thresh and ak >= self.union_area_ratio * a:
                    continue

            if not drop:
                keep.append(int(idx))

        keep_arr = np.array(keep, dtype=np.int64)

        # Pass 2: remove remaining masks that look like "union" covering 2+ kept masks
        final_keep: list[int] = []
        for idx in keep_arr:
            m = mbool[idx]
            a = areas[idx]

            contained_smalls = 0
            for j in keep_arr:
                if j == idx:
                    continue
                mj = mbool[j]
                aj = areas[j]
                if aj == 0:
                    continue

                # if j is almost entirely inside idx, and idx is much larger => idx is union-ish
                if self._containment(mj, m) >= self.contain_thresh and a >= self.union_area_ratio * aj:
                    contained_smalls += 1

            # If this mask contains 2+ other kept masks, it's almost certainly a union => drop it
            if contained_smalls >= 2:
                continue

            final_keep.append(int(idx))

        return np.array(final_keep, dtype=np.int64)

    # ----------------- Main API -----------------

    @torch.inference_mode()
    def predict(self, img_rgb_uint8: np.ndarray, score_thresh: float = 0.5) -> InstanceSegmentationResult:
        """
        Post-processing order:
          1) score threshold
          2) tiny-mask removal (relative to image size)
          3) dedupe + union-mask removal
        """
        H, W = int(img_rgb_uint8.shape[0]), int(img_rgb_uint8.shape[1])

        img_pil = Image.fromarray(img_rgb_uint8)
        x = self.preprocess(img_pil).to(self.device)  # [C,H,W]

        out = self.model([x])[0]
        scores_all = out["scores"].detach().cpu().numpy()

        # ---- (1) threshold by score ----
        keep = scores_all >= float(score_thresh)
        boxes = out["boxes"].detach().cpu().numpy()[keep]                 # [N,4]
        class_ids = out["labels"].detach().cpu().numpy()[keep]            # [N]
        scrs = scores_all[keep]                                           # [N]
        masks_bool = out["masks"].detach().cpu().numpy()[keep, 0] >= 0.5  # [N,H,W] bool

        # ---- (2) guaranteed tiny-mask removal ----
        boxes, class_ids, scrs, masks_bool = self._filter_tiny_masks(
            boxes, class_ids, scrs, masks_bool, H=H, W=W
        )

        # ---- (3) dedupe + union removal (prefer separate A,B over A+B) ----
        keep2 = self._dedupe_masks_prefer_separate(masks_bool, scrs)

        boxes = boxes[keep2]
        class_ids = class_ids[keep2]
        scrs = scrs[keep2]
        masks_bool = masks_bool[keep2]

        return InstanceSegmentationResult(
            boxes_xyxy=boxes,
            scores=scrs,
            class_ids=class_ids,
            masks=masks_bool,
        )

    @torch.inference_mode()
    def __call__(self, img_rgb_uint8: np.ndarray, score_thresh: float = 0.5) -> InstanceSegmentationResult:
        return self.predict(img_rgb_uint8, score_thresh=score_thresh)

    def save_masks(self, out_dir: Path, img_rgb_uint8: np.ndarray, instance_outputs, topk: int = 10) -> None:
        """
        Debug-only: Save per-instance binary masks and an overlay visualization.
        (You can simply not call this if you don’t want those files.)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        overlay = img_rgb_uint8.copy().astype(np.float32)

        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [255, 127, 0],
                [127, 0, 255],
                [0, 127, 255],
                [127, 255, 0],
            ],
            dtype=np.float32,
        )
        alpha = 0.45

        for i, inst in enumerate(instance_outputs[:topk]):
            m = inst.mask.astype(bool)
            Image.fromarray((m.astype(np.uint8) * 255)).save(out_dir / f"mask_{i:03d}.png")
            color = colors[i % len(colors)]
            overlay[m] = (1.0 - alpha) * overlay[m] + alpha * color

        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_dir / "masks_overlay.png")
