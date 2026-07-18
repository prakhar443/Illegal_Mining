#!/usr/bin/env python
"""Evaluate a SPEAR-Net checkpoint on the LAMES val/test split and dump metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spearnet.config import load_config
from spearnet.data import build_dataloaders
from spearnet.engine import evaluate
from spearnet.models import build_model
from spearnet.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--area-stratified", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg, splits=(args.split,))
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    results = evaluate(model, loaders[args.split], cfg, device,
                       compute_area_stratified=args.area_stratified)

    skip = {"per_class", "confusion_matrix", "class_names"}
    print(json.dumps({k: v for k, v in results.items() if k not in skip}, indent=2))
    print(f"\nmIoU={results['miou']:.4f}  foreground-mIoU={results.get('fg_miou', float('nan')):.4f}")
    print("\nPer-class (precision / recall / F1 for Tables 2-4):")
    for cls, m in results["per_class"].items():
        print(f"  {cls:>18}: IoU={m['iou']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} support={m['support']}")
    if "confusion_matrix" in results:
        print("\nConfusion matrix (rows = ground truth, cols = prediction):")
        names = results.get("class_names", [])
        print("        " + "".join(f"{n[:10]:>12}" for n in names))
        for r, row in enumerate(results["confusion_matrix"]):
            print(f"  {names[r][:6]:>6}" + "".join(f"{v:>12d}" for v in row))

    out = args.out or os.path.join(cfg.run.out_dir, f"metrics_{args.split}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {out}")


if __name__ == "__main__":
    main()
