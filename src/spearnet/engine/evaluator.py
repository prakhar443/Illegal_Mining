"""Evaluation loop: confusion-matrix metrics + optional area-stratified recall."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..metrics import SegMetrics, area_stratified_recall


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    compute_area_stratified: bool = False,
    max_area_batches: int = 16,
) -> Dict[str, object]:
    model.eval()
    metrics = SegMetrics(cfg.num_classes, cfg.class_names)

    preds_acc, tgts_acc = [], []
    for bi, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        out = model(image)
        metrics.update(out["logits"], target)

        if compute_area_stratified and bi < max_area_batches:
            preds_acc.append(out["logits"].argmax(1).cpu().numpy())
            tgts_acc.append(target.cpu().numpy())

    results = metrics.compute()

    if compute_area_stratified and preds_acc:
        preds = np.concatenate(preds_acc, axis=0)
        tgts = np.concatenate(tgts_acc, axis=0)
        results["area_stratified_recall"] = area_stratified_recall(preds, tgts, cfg.num_classes)

    return results
