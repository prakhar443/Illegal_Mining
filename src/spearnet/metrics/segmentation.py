"""Segmentation metrics (Section 5): mIoU, per-class IoU/precision/recall/F1, and
area-stratified recall over small / medium / large connected components.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class SegMetrics:
    """Accumulate a confusion matrix over batches, then derive metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).view(-1).cpu().numpy()
        gt = target.view(-1).cpu().numpy()
        mask = (gt >= 0) & (gt < self.num_classes)
        idx = self.num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        eps = 1e-9
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        present = (cm.sum(axis=1) > 0)
        miou = float(iou[present].mean()) if present.any() else 0.0
        pix_acc = float(tp.sum() / (cm.sum() + eps))

        per_class = {
            self.class_names[i]: {
                "iou": float(iou[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm.sum(axis=1)[i]),
            }
            for i in range(self.num_classes)
        }
        return {
            "miou": miou,
            "pixel_acc": pix_acc,
            "mean_f1": float(f1[present].mean()) if present.any() else 0.0,
            "mean_recall": float(recall[present].mean()) if present.any() else 0.0,
            "per_class": per_class,
        }


def _connected_components(binary: np.ndarray) -> np.ndarray:
    """Label 4-connected components without SciPy (simple BFS). binary: (H,W) bool."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    from collections import deque

    for i in range(h):
        for j in range(w):
            if binary[i, j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                labels[i, j] = cur
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            q.append((ny, nx))
    return labels


def area_stratified_recall(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    small: int = 32 * 32,
    large: int = 96 * 96,
) -> Dict[str, float]:
    """Component-level recall stratified by ground-truth component area.

    A target component (per foreground class) counts as recovered if any pixel in it is
    predicted as that same class. Areas: small (<small), medium, large (>=large) px.

    Args:
        preds:   (N, H, W) int array of predictions.
        targets: (N, H, W) int array of ground truth.
    """
    buckets = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [recovered, total]

    for p, t in zip(preds, targets):
        for c in range(1, num_classes):  # skip background
            gt_c = t == c
            if not gt_c.any():
                continue
            labels = _connected_components(gt_c)
            for comp_id in range(1, labels.max() + 1):
                comp = labels == comp_id
                area = int(comp.sum())
                bucket = "small" if area < small else ("large" if area >= large else "medium")
                recovered = bool((p[comp] == c).any())
                buckets[bucket][1] += 1
                if recovered:
                    buckets[bucket][0] += 1

    return {
        k: (v[0] / v[1] if v[1] > 0 else float("nan"))
        for k, v in buckets.items()
    } | {f"{k}_count": v[1] for k, v in buckets.items()}
