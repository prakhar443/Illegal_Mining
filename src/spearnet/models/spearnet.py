"""SPEAR-Net — encoder + spectral-prior gate + gated decoder + heads (v4, 6-band S2)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.priors import N_PRIORS, band_index, compute_priors
from .csp import PriorGate
from .decoder import UNetDecoder
from .encoder import TimmEncoder
from .heads import EdgeHead, SegHead


class SPEARNet(nn.Module):
    """Lightweight spectral-prior-gated recall-optimized segmentation network.

    Args:
        num_classes:    segmentation classes for the active task.
        in_chans:       number of input bands (6 for S2 B,G,R,NIR,SWIR1,SWIR2; 3 for RGB).
        band_order:     names of the input bands, used to locate bands for prior math.
        prior_type:     "pisp" (Sentinel-2 spectral) or "csp" (RGB-only, ablation).
        backbone:       timm backbone name.
        csp_mode:       "gate" (attention-modulated skips), "concat" (priors as extra
                        input channels), or "none" (no prior — ablation baseline).
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 6,
        band_order: Optional[List[str]] = None,
        prior_type: str = "pisp",
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
        self.prior_type = prior_type
        self.num_classes = num_classes
        self.in_chans = in_chans
        if band_order is None:
            band_order = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"][:in_chans]
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.band_idx = band_index(band_order)

        # concat mode appends the 4 priors to the input -> project back to in_chans so the
        # pretrained stem shape is preserved.
        enc_in = in_chans
        if csp_mode == "concat":
            self.input_proj: Optional[nn.Module] = nn.Conv2d(in_chans + N_PRIORS, in_chans, 1)
        else:
            self.input_proj = None

        self.encoder = TimmEncoder(backbone, pretrained=pretrained, in_chans=enc_in)
        self.gate = PriorGate(N_PRIORS) if csp_mode == "gate" else None
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=list(decoder_channels),
            use_gate=(csp_mode == "gate"),
            alpha_init=csp_alpha_init,
        )
        dec_ch = self.decoder.out_channels
        self.seg_head = SegHead(dec_ch, num_classes)
        self.edge_head = EdgeHead(dec_ch) if use_edge_head else None

    def _priors(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.csp_mode == "none":
            return None
        return compute_priors(x, self.prior_type, self.band_idx)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args: x (B, in_chans, H, W), reflectance-scaled. Returns logits/edge/attn dict."""
        h, w = x.shape[-2:]
        priors = self._priors(x)
        attn = self.gate(priors) if (self.gate is not None and priors is not None) else None

        if self.csp_mode == "concat" and priors is not None:
            inp = self.input_proj(torch.cat([x, priors], dim=1))
        else:
            inp = x

        feats = self.encoder(inp)
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
