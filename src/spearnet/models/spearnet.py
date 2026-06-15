"""SPEAR-Net — the full model assembling encoder + CSP + gated decoder + heads."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.priors import N_CSP_PRIORS
from .csp import CSPGate
from .decoder import UNetDecoder
from .encoder import TimmEncoder
from .heads import EdgeHead, SegHead


class SPEARNet(nn.Module):
    """Lightweight color-prior-gated recall-optimized segmentation network.

    Args:
        num_classes:    number of segmentation classes for the active task.
        backbone:       timm backbone name (mobilenetv3_small_100 / efficientnet_lite0 ...).
        pretrained:     load ImageNet weights.
        decoder_channels: decoder widths (deep->shallow).
        csp_mode:       "gate" (attention-modulated skips), "concat" (priors as input
                        channels), or "none" (no prior — ablation baseline).
        csp_alpha_init: initial per-scale gate strength.
        use_edge_head:  add the deep-supervision boundary head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        decoder_channels=(128, 64, 48, 32),
        csp_mode: str = "gate",
        csp_alpha_init: float = 1.0,
        use_edge_head: bool = True,
    ):
        super().__init__()
        assert csp_mode in {"gate", "concat", "none"}
        self.csp_mode = csp_mode
        self.num_classes = num_classes

        # --- input stem (concat mode appends the 4 priors to the RGB input) ---
        self.in_chans = 3 + (N_CSP_PRIORS if csp_mode == "concat" else 0)
        if self.in_chans != 3:
            # adapt to a 3-channel pretrained stem with a learned 1x1 down-projection
            self.input_proj: Optional[nn.Module] = nn.Conv2d(self.in_chans, 3, kernel_size=1)
        else:
            self.input_proj = None

        self.encoder = TimmEncoder(backbone, pretrained=pretrained)

        self.csp = CSPGate() if csp_mode == "gate" else None

        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=list(decoder_channels),
            use_gate=(csp_mode == "gate"),
            alpha_init=csp_alpha_init,
        )

        dec_ch = self.decoder.out_channels
        self.seg_head = SegHead(dec_ch, num_classes)
        self.edge_head = EdgeHead(dec_ch) if use_edge_head else None

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args: rgb (B, 3, H, W) in [0, 1]. Returns dict with 'logits', optional 'edge'/'attn'."""
        h, w = rgb.shape[-2:]

        attn = self.csp(rgb) if self.csp is not None else None

        if self.csp_mode == "concat":
            from ..data.priors import compute_csp_priors

            x = torch.cat([rgb, compute_csp_priors(rgb)], dim=1)
            x = self.input_proj(x)
        else:
            x = rgb

        feats = self.encoder(x)
        dec = self.decoder(feats, attn)

        logits = self.seg_head(dec)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if self.edge_head is not None:
            edge = self.edge_head(dec)
            if edge.shape[-2:] != (h, w):
                edge = F.interpolate(edge, size=(h, w), mode="bilinear", align_corners=False)
            out["edge"] = edge
        if attn is not None:
            out["attn"] = attn
        return out
