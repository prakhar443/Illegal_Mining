#!/usr/bin/env python
"""Train SPEAR-Net (or a baseline) on LAMES.

Examples:
    python scripts/train.py --config configs/spearnet_10class.yaml
    python scripts/train.py --config configs/spearnet_3class.yaml --set optim.epochs=5 data.subset_train=500
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spearnet.config import load_config
from spearnet.data import build_dataloaders
from spearnet.engine import Trainer
from spearnet.models import build_model
from spearnet.utils import count_parameters, set_seed


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        key, _, val = item.partition("=")
        overrides[key] = _coerce(val)
    return overrides


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--set", nargs="*", dest="overrides", help="dotted overrides key=value")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    set_seed(cfg.run.seed)
    os.makedirs(cfg.run.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.run.out_dir, "config.yaml"))

    print(f"Task={cfg.data.task} ({cfg.num_classes} classes) model={cfg.model.name} "
          f"backbone={cfg.model.backbone} csp={cfg.model.csp_mode}")

    print("Building dataloaders (loading LAMES from Hugging Face)...")
    loaders = build_dataloaders(cfg, splits=("train", "val"))
    print(f"  train batches={len(loaders['train'])}  val batches={len(loaders['val'])}")

    model = build_model(cfg)
    p = count_parameters(model)
    print(f"  params: {p['params_total_M']:.2f} M total, {p['params_trainable_M']:.2f} M trainable")

    trainer = Trainer(model, loaders, cfg)
    summary = trainer.train()
    print(f"Done. best {cfg.run.save_best_metric} = {summary['best_metric']:.4f}")
    print(f"Checkpoints in {cfg.run.out_dir}")


if __name__ == "__main__":
    main()
