from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceAwareFusionHead(nn.Module):
    """
    Instance-aware fusion head (trainable) that uses:
      - ROI backbone features:  [B, C, 7, 7]
      - ROI mask:              [B, 1, 7, 7]
      - ROI depth (masked ok): [B, 1, 7, 7]
      - depth stats:           [B, 8]

    Design:
      1) Encode (mask, depth) -> FiLM parameters (gamma,beta) over channels C
      2) Modulate ROI features
      3) Masked pooling (only pixels inside mask)
      4) MLP classifier -> logits
    """

    def __init__(self, in_channels: int, num_classes: int = 101, stats_dim: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        # mask+depth encoder -> FiLM (gamma,beta)
        self.md_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.film = nn.Linear(64, 2 * self.in_channels)

        # classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels + stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, self.num_classes),
        )

        # init FiLM close to identity
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        roi_feat: torch.Tensor,    # [B,C,7,7]
        mask_roi: torch.Tensor,    # [B,1,7,7] float {0,1}
        depth_roi: torch.Tensor,   # [B,1,7,7]
        depth_stats: torch.Tensor  # [B,8]
    ) -> torch.Tensor:
        B, C, H, W = roi_feat.shape
        assert C == self.in_channels, f"in_channels mismatch: got {C}, expected {self.in_channels}"

        md = torch.cat([mask_roi, depth_roi], dim=1)  # [B,2,7,7]
        h = self.md_encoder(md).flatten(1)            # [B,64]
        gb = self.film(h)                             # [B,2C]
        gamma, beta = gb[:, :C], gb[:, C:]            # [B,C], [B,C]

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        x = roi_feat * (1.0 + gamma) + beta           # FiLM (near identity at init)

        # masked average pool
        wmask = (mask_roi > 0.5).float()              # [B,1,7,7]
        denom = wmask.sum(dim=(2, 3), keepdim=False).clamp(min=1.0)  # [B,1]
        pooled = (x * wmask).sum(dim=(2, 3)) / denom  # [B,C]

        feat = torch.cat([pooled, depth_stats], dim=1)  # [B, C+8]
        logits = self.mlp(feat)
        return logits
