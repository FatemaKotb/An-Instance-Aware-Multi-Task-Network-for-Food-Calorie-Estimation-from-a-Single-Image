from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class InstanceSegmentationResult:
    boxes_xyxy: np.ndarray     # [N,4]
    scores: np.ndarray         # [N]
    class_ids: np.ndarray      # [N]
    masks: np.ndarray          # [N,H,W] bool

def load_maskrcnn(device: str = "cuda"):
    """
    TorchVision pretrained Mask R-CNN (COCO).
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess

@torch.inference_mode()
def predict_instances(model, preprocess, img_rgb_uint8: np.ndarray, device: str = "cuda", score_thresh: float = 0.5):
    """
    Input: RGB uint8 [H,W,3]
    Output: InstanceSegmentationResult
    """
    import torchvision
    from PIL import Image

    img = Image.fromarray(img_rgb_uint8)
    x = preprocess(img).to(device)  # tensor [C,H,W], normalized
    outputs = model([x])[0]

    scores = outputs["scores"].detach().cpu().numpy()
    keep = scores >= score_thresh

    boxes = outputs["boxes"].detach().cpu().numpy()[keep]
    scores = scores[keep]
    class_ids = outputs["labels"].detach().cpu().numpy()[keep]

    # masks: [N,1,H,W] -> [N,H,W] boolean
    masks = outputs["masks"].detach().cpu().numpy()[keep, 0]
    masks = masks >= 0.5

    return InstanceSegmentationResult(
        boxes_xyxy=boxes,
        scores=scores,
        class_ids=class_ids,
        masks=masks.astype(bool),
    )
