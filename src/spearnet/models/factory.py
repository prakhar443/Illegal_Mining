"""Model factory: SPEAR-Net + Colab-runnable baselines (Section 6).

Baselines are provided through ``segmentation-models-pytorch`` so the comparison table
(plain U-Net, Attention U-Net == Unet w/ attention decoder, DeepLabV3+, U-Net++) is
reproducible with the same training loop. All models return a dict with key ``logits``
(and SPEAR-Net additionally ``edge`` / ``attn``).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import Config
from .spearnet import SPEARNet


class _SMPWrapper(nn.Module):
    """Wrap an smp model so it returns the SPEAR-Net-style output dict."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits": self.model(rgb)}


def build_model(cfg: Config) -> nn.Module:
    name = cfg.model.name.lower()
    nc = cfg.num_classes

    if name == "spearnet":
        return SPEARNet(
            num_classes=nc,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            decoder_channels=cfg.model.decoder_channels,
            csp_mode=cfg.model.csp_mode,
            csp_alpha_init=cfg.model.csp_alpha_init,
            use_edge_head=cfg.model.use_edge_head,
        )

    import segmentation_models_pytorch as smp

    enc = cfg.model.backbone
    weights = "imagenet" if cfg.model.pretrained else None
    common = dict(encoder_name=_smp_encoder(enc), encoder_weights=weights,
                  in_channels=3, classes=nc)

    if name == "unet":
        model = smp.Unet(**common)
    elif name in ("attn_unet", "attention_unet"):
        model = smp.Unet(decoder_attention_type="scse", **common)
    elif name in ("deeplabv3p", "deeplabv3plus"):
        model = smp.DeepLabV3Plus(**common)
    elif name in ("unetpp", "unetplusplus"):
        model = smp.UnetPlusPlus(**common)
    elif name == "segformer":
        model = smp.Segformer(**common)
    else:
        raise ValueError(f"Unknown model.name '{cfg.model.name}'")

    return _SMPWrapper(model)


def _smp_encoder(timm_name: str) -> str:
    """Map a few common timm backbone names to smp encoder names."""
    mapping = {
        "mobilenetv3_small_100": "timm-mobilenetv3_small_100",
        "mobilenetv3_large_100": "timm-mobilenetv3_large_100",
        "efficientnet_lite0": "timm-efficientnet-lite0",
        "efficientnet_b0": "efficientnet-b0",
        "resnet34": "resnet34",
    }
    return mapping.get(timm_name, timm_name)
