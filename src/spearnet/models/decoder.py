"""Depthwise-separable U-Net decoder with CSP-gated skips (Section 4).

Progressive upsample + fuse. At each decoder stage the corresponding encoder skip is
(optionally) modulated by the CSP attention map before concatenation.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csp import GatedSkip


class DSConv(nn.Module):
    """Depthwise-separable conv: depthwise 3x3 + pointwise 1x1, each BN+ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample the running feature, fuse with a (gated) skip, refine with DSConvs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_gate: bool, alpha_init: float):
        super().__init__()
        self.gate = GatedSkip(alpha_init) if use_gate else None
        self.conv = nn.Sequential(
            DSConv(in_ch + skip_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor], attn: Optional[torch.Tensor]
    ) -> torch.Tensor:
        target = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        if skip is not None:
            if self.gate is not None and attn is not None:
                skip = self.gate(skip, attn)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """U-Net decoder consuming encoder features (low->high stride) with gated skips."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        use_gate: bool,
        alpha_init: float,
    ):
        super().__init__()
        # encoder_channels: shallow..deep (stride small..large). Reverse for decoding.
        enc = list(encoder_channels)
        self.deepest = enc[-1]
        skips = enc[:-1][::-1]  # skip channels from deep->shallow

        # Ensure we have one decoder width per skip (+ a final stage).
        dec = list(decoder_channels)
        while len(dec) < len(skips):
            dec.append(dec[-1] // 2 if dec[-1] > 1 else dec[-1])
        dec = dec[: len(skips)]

        blocks = []
        in_ch = self.deepest
        for skip_ch, out_ch in zip(skips, dec):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch, use_gate, alpha_init))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_ch

    def forward(
        self, features: List[torch.Tensor], attn: Optional[torch.Tensor]
    ) -> torch.Tensor:
        feats = list(features)
        x = feats[-1]
        skips = feats[:-1][::-1]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip, attn)
        return x
