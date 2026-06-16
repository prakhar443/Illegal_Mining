"""Spectral priors computed on the fly (Section 4.2).

Two families:

* **PISP** (Physically-Informed Spectral Prior, v4 core novelty) — needs Sentinel-2
  NIR/SWIR bands: NDVI, MNDWI, NDTI, BSI.
* **CSP** (RGB-only color prior, the v3 fallback / RGB-vs-spectral ablation): ExG,
  Brightness, Redness, RGB-VI.

Both return 4 channel-stacked priors so the gate architecture is identical regardless of
which family is selected. Band positions are resolved by name via a ``band_index`` map so
the same code works for 3-band (R,G,B) and 6-band (B,G,R,NIR,SWIR1,SWIR2) inputs.
"""
from __future__ import annotations

from typing import Dict, List

import torch

CSP_PRIOR_NAMES: List[str] = ["ExG", "Brightness", "Redness", "RGB_VI"]
PISP_PRIOR_NAMES: List[str] = ["NDVI", "MNDWI", "NDTI", "BSI"]
N_CSP_PRIORS: int = 4
N_PISP_PRIORS: int = 4
N_PRIORS: int = 4  # both families emit 4 priors

# Default band orderings.
RGB_BAND_ORDER = ["R", "G", "B"]
S2_BAND_ORDER = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]

_EPS = 1e-6


def band_index(band_order: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(band_order)}


def _ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) / (a + b + _EPS)


def compute_csp_priors(x: torch.Tensor, idx: Dict[str, int] = None) -> torch.Tensor:
    """RGB color priors. ``x`` is (..., C, H, W); ``idx`` maps band names to channels.

    Backwards-compatible: if ``idx`` is None, assume the legacy 3-channel R,G,B order.
    """
    if idx is None:
        idx = band_index(RGB_BAND_ORDER)
    r = x[..., idx["R"], :, :]
    g = x[..., idx["G"], :, :]
    b = x[..., idx["B"], :, :]

    exg = 2.0 * g - r - b                       # greenness / clearance contrast
    brightness = (r + g + b) / 3.0              # exposed ground / structures
    redness = _ratio(r, g)                      # iron-oxide / reddish tailings
    g2, rb = g * g, r * b
    rgb_vi = (g2 - rb) / (g2 + rb + _EPS)       # vegetation context
    return torch.stack([exg, brightness, redness, rgb_vi], dim=-3)


def compute_pisp_priors(x: torch.Tensor, idx: Dict[str, int] = None) -> torch.Tensor:
    """Physically-informed Sentinel-2 spectral priors (needs NIR + SWIR).

    NDVI  = (NIR-R)/(NIR+R)              vegetation -> cleared-land loss
    MNDWI = (G-SWIR1)/(G+SWIR1)          mining ponds / lagoons
    NDTI  = (R-G)/(R+G)                  turbid / muddy mining water
    BSI   = ((SWIR1+R)-(NIR+B))/(sum)    bare soil / tailings
    """
    if idx is None:
        idx = band_index(S2_BAND_ORDER)
    b = x[..., idx["B"], :, :]
    g = x[..., idx["G"], :, :]
    r = x[..., idx["R"], :, :]
    nir = x[..., idx["NIR"], :, :]
    swir1 = x[..., idx["SWIR1"], :, :]

    ndvi = _ratio(nir, r)
    mndwi = _ratio(g, swir1)
    ndti = _ratio(r, g)
    bsi = _ratio(swir1 + r, nir + b)
    return torch.stack([ndvi, mndwi, ndti, bsi], dim=-3)


def compute_priors(x: torch.Tensor, prior_type: str, idx: Dict[str, int] = None) -> torch.Tensor:
    if prior_type == "pisp":
        return compute_pisp_priors(x, idx)
    if prior_type == "csp":
        return compute_csp_priors(x, idx)
    raise ValueError(f"Unknown prior_type '{prior_type}' (expected 'pisp' or 'csp')")


def prior_names(prior_type: str) -> List[str]:
    return PISP_PRIOR_NAMES if prior_type == "pisp" else CSP_PRIOR_NAMES
