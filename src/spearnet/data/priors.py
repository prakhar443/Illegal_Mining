"""Color-Spectral Prior (CSP) computation — Section 4.2 of the methodology.

Four physically-motivated priors are computed on the fly from the RGB patch (no
preprocessing). They are returned channel-stacked and individually min-max friendly
(roughly bounded to [-1, 1] or [0, 1]) so they can be either concatenated to the input
(``concat`` mode) or fed to the CSP gate (``gate`` mode).
"""
from __future__ import annotations

from typing import List

import torch

CSP_PRIOR_NAMES: List[str] = ["ExG", "Brightness", "Redness", "RGB_VI"]
N_CSP_PRIORS: int = len(CSP_PRIOR_NAMES)

_EPS = 1e-6


def compute_csp_priors(rgb: torch.Tensor) -> torch.Tensor:
    """Compute the four CSP priors from an RGB tensor in [0, 1].

    Args:
        rgb: float tensor of shape ``(..., 3, H, W)`` with channels ordered R, G, B and
             values in [0, 1].

    Returns:
        Tensor of shape ``(..., 4, H, W)`` stacking [ExG, Brightness, Redness, RGB-VI].
    """
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected 3 channels at dim -3, got shape {tuple(rgb.shape)}")

    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]

    # ExG = 2G - R - B  (residual vegetation / clearance contrast). Range ~[-2, 2].
    exg = 2.0 * g - r - b

    # Brightness = (R+G+B)/3 (exposed ground, structures, plants). Range [0, 1].
    brightness = (r + g + b) / 3.0

    # Redness = (R-G)/(R+G) (iron-oxide / reddish tailings). Range ~[-1, 1].
    redness = (r - g) / (r + g + _EPS)

    # RGB-VI = (G^2 - R*B)/(G^2 + R*B) (vegetation context). Range ~[-1, 1].
    g2 = g * g
    rb = r * b
    rgb_vi = (g2 - rb) / (g2 + rb + _EPS)

    return torch.stack([exg, brightness, redness, rgb_vi], dim=-3)
