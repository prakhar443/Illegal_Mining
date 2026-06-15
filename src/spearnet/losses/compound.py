"""Recall-weighted, boundary-aware compound loss (Section 4.3).

    L = w_ce*CE + w_dice*Dice + w_tversky*Tversky(alpha<beta)
        + w_boundary*Boundary + w_edge*BCE(edge)

Tversky with beta > alpha directly penalises false negatives, recovering the rare,
small, fragmented classes (heap-leaching, tailings, processing plant) that vanilla
cross-entropy ignores under severe imbalance.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()


def _present_mean(per_class_loss: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Average a per-class loss over only the classes present in the batch GT.

    Averaging over *all* classes (including those absent from the batch) is what makes
    imbalanced segmentation collapse to background: every step an absent rare class adds
    a ~1.0 penalty for any probability mass spent on it, so the optimiser learns to never
    predict it. Restricting the mean to present classes removes that bias.
    """
    present = (onehot.sum(dim=(0, 2, 3)) > 0).float()  # (C,)
    denom = present.sum().clamp(min=1.0)
    return (per_class_loss * present).sum() / denom


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, present_only: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.present_only = present_only

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = _one_hot(target, self.num_classes)
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        loss = 1.0 - dice
        return _present_mean(loss, onehot) if self.present_only else loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss; alpha weights FPs, beta weights FNs. beta>alpha => recall-first."""

    def __init__(self, num_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, present_only: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.present_only = present_only

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = _one_hot(target, self.num_classes)
        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky
        return _present_mean(loss, onehot) if self.present_only else loss.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss: down-weights easy (background) pixels, focusing gradient on
    hard / rare-class pixels. Provides the per-pixel signal that pure region losses lack.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target.clamp(min=0), weight=self.weight,
                        ignore_index=self.ignore_index, reduction="none")
        p = torch.exp(-ce)
        return (((1 - p) ** self.gamma) * ce).mean()


class BoundaryLoss(nn.Module):
    """Boundary (BD) term: penalise prediction error concentrated at class boundaries.

    Uses the Sobel boundary of the target as a spatial weight on the per-pixel CE,
    sharpening small fragmented structures.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.clamp(min=0), reduction="none")  # (B,H,W)
        w = edge.squeeze(1)  # (B,H,W) in {0,1}
        denom = w.sum().clamp(min=1.0)
        return (ce * w).sum() / denom


class CompoundLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_ce: float = 0.0,
        w_dice: float = 1.0,
        w_tversky: float = 1.0,
        w_boundary: float = 0.5,
        w_edge: float = 0.5,
        w_focal: float = 0.0,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        present_only: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary
        self.w_edge = w_edge
        self.w_focal = w_focal
        self.ignore_index = ignore_index

        self.dice = DiceLoss(num_classes, present_only=present_only)
        self.tversky = TverskyLoss(num_classes, tversky_alpha, tversky_beta, present_only=present_only)
        self.boundary = BoundaryLoss(num_classes)
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else None, persistent=False
        )
        self.focal = FocalLoss(focal_gamma, weight=class_weights, ignore_index=ignore_index)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        logits = outputs["logits"]
        target = batch["mask"]
        comps: Dict[str, torch.Tensor] = {}

        total = logits.new_zeros(())

        if self.w_ce > 0:
            ce = F.cross_entropy(
                logits, target.clamp(min=0),
                weight=self.class_weights, ignore_index=self.ignore_index,
            )
            comps["ce"] = ce
            total = total + self.w_ce * ce

        if self.w_focal > 0:
            fl = self.focal(logits, target)
            comps["focal"] = fl
            total = total + self.w_focal * fl

        if self.w_dice > 0:
            d = self.dice(logits, target)
            comps["dice"] = d
            total = total + self.w_dice * d

        if self.w_tversky > 0:
            t = self.tversky(logits, target)
            comps["tversky"] = t
            total = total + self.w_tversky * t

        if self.w_boundary > 0 and "edge" in batch:
            bd = self.boundary(logits, target, batch["edge"])
            comps["boundary"] = bd
            total = total + self.w_boundary * bd

        if self.w_edge > 0 and "edge" in outputs and "edge" in batch:
            edge_loss = F.binary_cross_entropy_with_logits(outputs["edge"], batch["edge"])
            comps["edge"] = edge_loss
            total = total + self.w_edge * edge_loss

        comps["total"] = total
        return total, comps
