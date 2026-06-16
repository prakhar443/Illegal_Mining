"""Spectral-prior attention gate — core novelty (Section 4.2).

The 4 spectral priors (PISP or CSP, computed on the fly) pass through a tiny conv ->
depthwise 3x3 -> sigmoid to produce a single-channel spatial attention map. That map
modulates each decoder skip:

    F = F_skip * (1 + alpha * attn)

where ``alpha`` is a learnable per-scale scalar. The gate is shared and explainable, so
the attention map can be overlaid on detections to show "which spectral index fired where".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.priors import N_PRIORS


class PriorGate(nn.Module):
    """Map stacked spectral priors -> a sigmoid spatial-attention map."""

    def __init__(self, n_priors: int = N_PRIORS, hidden: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(n_priors, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, priors: torch.Tensor) -> torch.Tensor:
        """Args: priors (B, n_priors, H, W). Returns attention map (B, 1, H, W) in (0, 1)."""
        return torch.sigmoid(self.proj(priors))


# Backwards-compatible alias (older imports/tests used CSPGate).
CSPGate = PriorGate


class GatedSkip(nn.Module):
    """Modulate a skip feature map by the prior attention: F = F_skip * (1 + alpha*attn)."""

    def __init__(self, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, skip: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if attn.shape[-2:] != skip.shape[-2:]:
            attn = F.interpolate(attn, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * (1.0 + self.alpha * attn)
