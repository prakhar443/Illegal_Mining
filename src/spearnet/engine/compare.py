"""Ablation + baseline comparison runner (Methodology Section 6).

Trains a set of experiment variants under a shared, reduced budget so they are directly
comparable, and records mIoU / per-class IoU / params / GFLOPs for the paper's main table.

Robust for Colab: results persist to a JSON on disk and completed experiments are skipped,
so a crash/disconnect resumes the table where it left off (each experiment also resumes its
own ``last.pt``). A failing experiment is recorded and skipped, not fatal.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from typing import Dict, List, Optional

import torch

from ..config import Config, override_config
from ..data import build_dataloaders
from ..models import build_model
from ..utils import measure_efficiency
from .trainer import Trainer
from .evaluator import evaluate


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _load_results(path: str) -> List[Dict]:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_results(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def run_comparison(
    base_cfg: Config,
    experiments: Dict[str, Dict],
    results_path: str,
    runs_root: str,
    epochs: int = 12,
    subset_train: Optional[int] = 4000,
    subset_val: Optional[int] = 1000,
    device: Optional[str] = None,
) -> List[Dict]:
    """Train each experiment under a shared budget and return the result rows.

    Args:
        base_cfg:     the reference Config (e.g. loaded from configs/v4_scale3.yaml).
        experiments:  name -> dotted-override dict, e.g. {"U-Net": {"model.name": "unet"}}.
        results_path: JSON file to persist/resume the table.
        runs_root:    directory for per-experiment checkpoints.
        epochs/subset_*: shared, reduced budget so variants are comparable and fast.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_results(results_path)
    done = {r["name"] for r in rows}

    for name, ov in experiments.items():
        if name in done:
            print(f"[compare] skip (done): {name}")
            continue
        print(f"\n========== [compare] {name} ==========", flush=True)
        try:
            cfg = override_config(base_cfg, ov)
            cfg.optim.epochs = epochs
            cfg.data.subset_train = subset_train
            cfg.data.subset_val = subset_val
            cfg.run.out_dir = os.path.join(runs_root, _safe(name))
            os.makedirs(cfg.run.out_dir, exist_ok=True)
            last = os.path.join(cfg.run.out_dir, "last.pt")
            cfg.run.resume = last if os.path.exists(last) else None

            loaders = build_dataloaders(cfg, splits=("train", "val"))
            model = build_model(cfg)
            eff = measure_efficiency(model, cfg.data.image_size, dev, in_chans=cfg.data.bands)

            Trainer(model, loaders, cfg).train()

            best = os.path.join(cfg.run.out_dir, "best.pt")
            if os.path.exists(best):
                model.load_state_dict(torch.load(best, map_location=dev)["model"])
            res = evaluate(model, loaders["val"], cfg, torch.device(dev))

            row = {
                "name": name,
                "task": cfg.data.task,
                "model": cfg.model.name,
                "prior": cfg.model.prior_type if cfg.model.name == "spearnet" else "-",
                "csp_mode": cfg.model.csp_mode if cfg.model.name == "spearnet" else "-",
                "bands": cfg.data.bands,
                "mIoU": round(res["miou"], 4),
                "mRecall": round(res["mean_recall"], 4),
                "params_M": round(eff["params_total_M"], 3),
                "gflops": round(eff["gflops"], 3) if eff["gflops"] == eff["gflops"] else None,
                "epochs": epochs,
            }
            for cls, m in res["per_class"].items():
                row[f"IoU_{cls}"] = round(m["iou"], 4)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[compare] FAILED {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})

        _save_results(results_path, rows)  # persist after every experiment

    print(f"\n[compare] done. Results -> {results_path}")
    return rows
