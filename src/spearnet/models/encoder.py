"""Lightweight encoder wrapper (Section 4.1).

Wraps a `timm` backbone in ``features_only`` mode (MobileNetV3-Small ~2-3 M params, or
EfficientNet-lite0 ~4-6 M params), ImageNet-pretrained. Exposes the multi-scale feature
maps (strides 2..32) and their channel counts so the decoder can build gated skips.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TimmEncoder(nn.Module):
    def __init__(self, name: str = "mobilenetv3_small_100", pretrained: bool = True):
        super().__init__()
        import timm

        # out_indices default returns the deepest 5 feature scales for most backbones.
        self.backbone = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
        )
        self.feature_channels: List[int] = list(self.backbone.feature_info.channels())
        self.reductions: List[int] = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
