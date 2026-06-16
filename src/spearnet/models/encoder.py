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
    def __init__(self, name: str = "mobilenetv3_small_100", pretrained: bool = True,
                 in_chans: int = 3):
        super().__init__()
        import timm

        # timm adapts the ImageNet-pretrained stem to in_chans>3 by tiling/scaling the
        # RGB filters into the extra (NIR/SWIR) channels.
        self.backbone = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.feature_channels: List[int] = list(self.backbone.feature_info.channels())
        self.reductions: List[int] = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
