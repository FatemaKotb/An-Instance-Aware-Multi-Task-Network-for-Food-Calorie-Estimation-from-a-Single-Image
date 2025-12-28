from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from scripts.pipeline import Pipeline
from scripts.lvl1.efficientnet import EfficientNetBlock
from scripts.lvl1.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.lvl1.zoedepth import ZoeDepthBlock
from scripts.lvl2.fusion_stub import FusionStub
from scripts.lvl3.prediction_head_pretrained import PredictionHeadPretrainedCLIP
from scripts.lvl4.physics_head_stub import PhysicsHeadStub


@dataclass
class PipelineFactoryConfig:
    device: str = "cuda"
    seg_score_thresh: float = 0.5

    # Backbone
    backbone_model_name: str = "tf_efficientnetv2_s"
    backbone_out_indices: tuple = (1, 2, 3, 4)

    # Physics
    density_by_name: Optional[Dict[str, float]] = None
    kcal_per_g_by_name: Optional[Dict[str, float]] = None

def build_default_pipeline(cfg: PipelineFactoryConfig) -> Pipeline:
    """Build a fully-wired pipeline with explicit blocks (Option 1)."""
    backbone = EfficientNetBlock(
        model_name=cfg.backbone_model_name,
        device=cfg.device,
        mode="backbone",
        out_indices=cfg.backbone_out_indices,
    )
    seg = MaskRCNNTorchVisionBlock(device=cfg.device)
    depth = ZoeDepthBlock(device=cfg.device)

    fusion = FusionStub()
    pred_head = PredictionHeadPretrainedCLIP(device=cfg.device)
    phys_head = PhysicsHeadStub()

    return Pipeline(
        backbone_block=backbone,
        seg_block=seg,
        depth_block=depth,
        fusion_block=fusion,
        prediction_head=pred_head,
        physics_head=phys_head,
        device=cfg.device,
        seg_score_thresh=cfg.seg_score_thresh,
    )
