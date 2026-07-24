#!/usr/bin/env python
"""Phase-B model analysis: learned gate strengths (B7) and PISP faithfulness (B8).

B7  Extract the learned per-scale gate scalars alpha_s from a checkpoint (CPU, seconds).
    Direct evidence of whether/how strongly the spectral prior is used.
B8  PISP faithfulness: use the attention map A as a foreground detector and compute its
    ROC-AUC against the ground-truth foreground mask over the evaluation set. Converts
    "explainable" from an adjective into a number.

Usage:
    python scripts/analyze_model.py --config configs/v4_scale3.yaml \
        --checkpoint runs/.../best.pt --split test --out analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from spearnet.config import load_config
from spearnet.data import build_dataloaders
from spearnet.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--chips-root", default=None,
                    help="chips dir (overrides the config; manifest is <chips-root>/manifest.csv)")
    ap.add_argument("--manifest", default=None, help="explicit manifest.csv path")
    ap.add_argument("--max-batches", type=int, default=64,
                    help="batches to sample for the AUC pass; <=0 uses the full split "
                         "(avoids the regional clustering of the first-N-in-order batches)")
    ap.add_argument("--out", default="analysis.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.chips_root:
        cfg.data.chips_root = args.chips_root
        cfg.data.manifest = os.path.join(args.chips_root, "manifest.csv")
    if args.manifest:
        cfg.data.manifest = args.manifest
    cfg.data.subset_test = None; cfg.data.subset_val = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    out = {"checkpoint": args.checkpoint, "config": args.config, "split": args.split}

    # ---- B7: learned alpha_s (the per-scale GatedSkip scalars) ----
    alphas = {k: float(v) for k, v in model.state_dict().items()
              if k.endswith("alpha") or ".alpha" in k}
    out["alpha_s"] = alphas
    print("B7  learned gate strengths alpha_s:")
    for k, v in alphas.items():
        print(f"    {k}: {v:+.4f}")
    if not alphas:
        print("    (no alpha parameters found -- model has no gate, e.g. concat/none variant)")

    # ---- B8: PISP faithfulness AUC (attention vs GT foreground) ----
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    loaders = build_dataloaders(cfg, splits=(args.split,))
    scores, labels = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loaders[args.split]):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            x = batch["image"].to(device)
            res = model(x)
            if "attn" not in res:
                print("B8  model has no attention map (non-gate variant); skipping AUC.")
                break
            a = res["attn"][:, 0].float().cpu().numpy()          # (B,H,W) in (0,1)
            fg = (batch["mask"] > 0).long().numpy()              # (B,H,W)
            # subsample pixels to keep memory/time bounded
            a = a.reshape(-1)[::13]; fg = fg.reshape(-1)[::13]
            scores.append(a); labels.append(fg)
    if scores and roc_auc_score is not None:
        s = np.concatenate(scores); y = np.concatenate(labels)
        if y.min() != y.max():
            auc = float(roc_auc_score(y, s))
            out["pisp_faithfulness_auc"] = auc
            out["pisp_auc_n_pixels"] = int(y.size)
            print(f"B8  PISP faithfulness ROC-AUC = {auc:.4f}  (n={y.size} pixels, "
                  f"positives={int(y.sum())})")
        else:
            print("B8  only one class present in sampled pixels; AUC undefined.")
    elif roc_auc_score is None:
        print("B8  scikit-learn not available; install it to compute AUC.")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
