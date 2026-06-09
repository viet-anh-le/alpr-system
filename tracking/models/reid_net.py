"""
tracking/models/reid_net.py — Vehicle Re-ID network for appearance-based tracking.

MobileNetV3-Small backbone (ImageNet pretrained) + two-layer projection head.
Produces L2-normalised 128-d embeddings used by BoT-SORT's appearance module.

Integration:
  After training, export weights to weights/tracking/vehicle_reid.pt and point
  BoT-SORT's reid_weights config key at that file (botsort.yaml).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class BaseReIDModel(nn.Module):
    """Base class for Re-ID appearance models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VehicleReIDNet(BaseReIDModel):
    """
    Input:  (B, 3, 256, 128)  — vehicle crops, RGB, ImageNet-normalised
    Output: (B, 128)           — L2-normalised embeddings

    Architecture:
      MobileNetV3-Small features (ImageNet pretrained)
      → AdaptiveAvgPool → flatten (576-d)
      → Linear(576→256) + BN + ReLU
      → Linear(256→128)
      → L2 normalise

    Optional classification head (training only):
      Linear(128 → num_ids) for softmax-CE loss alongside triplet loss.
    """

    def __init__(self, embedding_dim: int = 128, num_ids: int | None = None) -> None:
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_ids) if num_ids else None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)   # (B, 576)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        emb = F.normalize(self.proj(feat), dim=1)   # (B, 128) — unit norm
        if return_logits and self.classifier is not None:
            return emb, self.classifier(emb)
        return emb
