"""Lightweight, dependency-free transforms.

We keep augmentation minimal (flips / 90-degree rotations) so the pipeline stays robust
across torch versions and so the CSP priors (computed from RGB) remain valid.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F


def random_flip_rotate(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a random horizontal/vertical flip and a random 0/90/180/270 rotation.

    Args:
        image: (3, H, W) float tensor.
        mask:  (H, W) long tensor.
    """
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])
    k = random.randint(0, 3)
    if k:
        image = torch.rot90(image, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])
    return image.contiguous(), mask.contiguous()


# Sobel kernels registered lazily so they live on the right device.
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t().contiguous()


def sobel_edges(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Derive a binary boundary map (deep-supervision target) from an integer mask.

    The mask is one-hot encoded and a Sobel gradient is taken per class; any non-zero
    gradient marks a class boundary. Works for a single mask ``(H, W)`` or a batch
    ``(B, H, W)`` and returns edges of shape ``(1, H, W)`` or ``(B, 1, H, W)``.
    """
    squeeze = mask.dim() == 2
    if squeeze:
        mask = mask.unsqueeze(0)  # (1, H, W)

    b, h, w = mask.shape
    onehot = F.one_hot(mask.long().clamp(min=0), num_classes).permute(0, 3, 1, 2).float()

    kx = _SOBEL_X.to(mask.device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(mask.device).view(1, 1, 3, 3)
    kx = kx.repeat(num_classes, 1, 1, 1)
    ky = ky.repeat(num_classes, 1, 1, 1)

    gx = F.conv2d(onehot, kx, padding=1, groups=num_classes)
    gy = F.conv2d(onehot, ky, padding=1, groups=num_classes)
    grad = (gx.abs() + gy.abs()).sum(dim=1, keepdim=True)  # (B,1,H,W)

    edges = (grad > 0.5).float()
    return edges.squeeze(0) if squeeze else edges
