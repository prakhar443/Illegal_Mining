"""Output heads (Section 4.3 / 4.4): segmentation head + edge/boundary head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import DSConv


class SegHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.refine = DSConv(in_ch, in_ch)
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.refine(x))


class EdgeHead(nn.Module):
    """Deep-supervision boundary head: predicts a single-channel edge logit map."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.refine = DSConv(in_ch, max(in_ch // 2, 8))
        self.classifier = nn.Conv2d(max(in_ch // 2, 8), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.refine(x))
