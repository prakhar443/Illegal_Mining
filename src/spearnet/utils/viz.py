"""Visualization & explainability (Section 5/6): mask colorization, CSP attention
overlays, and side-by-side prediction panels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# A fixed 10-colour palette (RGB, 0-255) for the LAMES classes.
PALETTE = np.array(
    [
        [0, 0, 0],         # 0 other
        [220, 20, 60],     # 1 ASM (crimson — illegal-prone, highlighted)
        [30, 144, 255],    # 2 LSM
        [255, 165, 0],     # 3 heap leaching
        [148, 0, 211],     # 4 mine facility
        [60, 179, 113],    # 5 open pit
        [255, 215, 0],     # 6 processing plant
        [139, 69, 19],     # 7 stockyard
        [255, 105, 180],   # 8 tailings storage
        [128, 128, 128],   # 9 waste rock dump
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map an integer (H, W) mask to an (H, W, 3) uint8 RGB image."""
    mask = np.asarray(mask).astype(np.int64)
    pal = PALETTE
    if mask.max() >= len(pal):
        # extend palette deterministically for >10 classes
        rng = np.random.RandomState(0)
        extra = rng.randint(0, 255, size=(mask.max() + 1 - len(pal), 3)).astype(np.uint8)
        pal = np.concatenate([pal, extra], axis=0)
    return pal[mask]


def _to_hwc_uint8(image: torch.Tensor, rgb_idx=None) -> np.ndarray:
    """Convert a (C,H,W) tensor to an (H,W,3) uint8 display image.

    For multi-band (e.g. 6-band S2 in B,G,R,NIR,SWIR1,SWIR2 order) pass ``rgb_idx`` to
    pick the visible bands; defaults to [2,1,0] for >3 bands (R,G,B) else first 3.
    A 2–98 percentile stretch makes Sentinel-2 reflectance visible.
    """
    img = image.detach().cpu().float()
    if img.dim() == 3:
        c = img.shape[0]
        if rgb_idx is None:
            rgb_idx = [2, 1, 0] if c >= 6 else list(range(min(3, c)))
        img = img[rgb_idx] if c > 1 else img.repeat(3, 1, 1)
        img = img.permute(1, 2, 0)
    arr = img.numpy()
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def overlay_csp_attention(
    image: torch.Tensor, attn: torch.Tensor, alpha: float = 0.5
) -> np.ndarray:
    """Overlay the CSP attention heatmap on the RGB image. Returns (H, W, 3) uint8."""
    import matplotlib.cm as cm

    img = _to_hwc_uint8(image).astype(np.float32)
    a = attn.detach().cpu().float()
    if a.dim() == 3:
        a = a[0]
    a = a.numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    heat = (cm.get_cmap("jet")(a)[..., :3] * 255.0)
    if heat.shape[:2] != img.shape[:2]:
        import torch.nn.functional as F

        heat_t = torch.from_numpy(heat).permute(2, 0, 1)[None]
        heat_t = F.interpolate(heat_t, size=img.shape[:2], mode="bilinear", align_corners=False)
        heat = heat_t[0].permute(1, 2, 0).numpy()
    out = (1 - alpha) * img + alpha * heat
    return out.clip(0, 255).astype(np.uint8)


def save_prediction_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    path: str,
    attn: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None,
) -> None:
    """Save an image | GT | prediction (| CSP attention) panel to ``path``."""
    import matplotlib.pyplot as plt

    cols = 4 if attn is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    axes[0].imshow(_to_hwc_uint8(image)); axes[0].set_title("RGB")
    axes[1].imshow(colorize_mask(target.cpu().numpy())); axes[1].set_title("Ground truth")
    axes[2].imshow(colorize_mask(pred.cpu().numpy())); axes[2].set_title("Prediction")
    if attn is not None:
        axes[3].imshow(overlay_csp_attention(image, attn)); axes[3].set_title("CSP attention")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
