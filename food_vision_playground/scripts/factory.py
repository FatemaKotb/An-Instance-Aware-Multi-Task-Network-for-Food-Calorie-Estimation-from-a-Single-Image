from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from scripts.pipeline import Pipeline
from scripts.lvl1.efficientnet import EfficientNetBlock
from scripts.lvl1.maskrcnn_torchvision import MaskRCNNTorchVisionBlock
from scripts.lvl1.zoedepth import ZoeDepthBlock
from scripts.lvl2.fusion_stub import FusionStub
from scripts.lvl2.instance_aware_fusion_block import InstanceAwareFusionBlock
from scripts.lvl3.prediction_head_pretrained import PredictionHeadPretrainedCLIP
from scripts.lvl3.prediction_head_finetuned import PredictionHeadFinetunedCLIPLinear
from scripts.lvl3.prediction_head_hybrid import PredictionHeadHybridCLIP
from scripts.lvl3.prediction_head_fusion_model import PredictionHeadFusionModel
from scripts.lvl4.physics_head_stub import PhysicsHeadStub


@dataclass
class PipelineFactoryConfig:
    device: str = "cuda"
    seg_score_thresh: float = 0.5

    backbone_model_name: str = "tf_efficientnetv2_s"
    backbone_out_indices: tuple = (1, 2, 3, 4)

    density_by_name: Optional[Dict[str, float]] = None
    kcal_per_g_by_name: Optional[Dict[str, float]] = None

    finetuned_clip_ckpt_path: str = "egypt_clip_linear.pt"

    # Pipeline mode
    # - "hybrid_clip": Hybrid CLIP-only head (baseline)
    # - "fusion_head": trained fusion-head classifier
    pipeline_mode: str = "hybrid_clip"

    # Used only when pipeline_mode == "fusion_head"
    fusion_head_ckpt_path: str = "fusion_head_subset.pt"


def build_pipeline(cfg: PipelineFactoryConfig) -> Pipeline:
    backbone = EfficientNetBlock(
        model_name=cfg.backbone_model_name,
        device=cfg.device,
        mode="backbone",
        out_indices=cfg.backbone_out_indices,
    )
    seg = MaskRCNNTorchVisionBlock(device=cfg.device)
    depth = ZoeDepthBlock(device=cfg.device)

    if str(cfg.pipeline_mode).lower() == "fusion_head":
        fusion = InstanceAwareFusionBlock()
        pred_head = PredictionHeadFusionModel(ckpt_path=cfg.fusion_head_ckpt_path, device=cfg.device)
    else:
        fusion = FusionStub()
        food101_head = PredictionHeadPretrainedCLIP(device=cfg.device)
        egypt_head = PredictionHeadFinetunedCLIPLinear(
            ckpt_path=cfg.finetuned_clip_ckpt_path,
            device=cfg.device,
        )
        pred_head = PredictionHeadHybridCLIP(
            food101_head=food101_head,
            egypt_head=egypt_head,
            egypt_conf_thresh=0.60,
        )

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


# Backward-compatible alias
build_default_pipeline = build_pipeline
