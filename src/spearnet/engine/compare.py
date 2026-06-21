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
from ..utils import measure_efficiency, set_seed
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
        epochs/subset_*: shared budget so variants are comparable. For the final paper
                      table use epochs=40 and subset_train=subset_val=None (full data).
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
            row = _run_one(base_cfg, name, ov, runs_root, epochs, subset_train, subset_val, dev)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[compare] FAILED {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})

        _save_results(results_path, rows)  # persist after every experiment

    print(f"\n[compare] done. Results -> {results_path}")
    return rows


def _run_one(base_cfg, name, ov, runs_root, epochs, subset_train, subset_val, dev) -> Dict:
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
        "name": name, "task": cfg.data.task, "model": cfg.model.name,
        "prior": cfg.model.prior_type if cfg.model.name == "spearnet" else "-",
        "csp_mode": cfg.model.csp_mode if cfg.model.name == "spearnet" else "-",
        "bands": cfg.data.bands,
        "mIoU": round(res["miou"], 4), "mRecall": round(res["mean_recall"], 4),
        "params_M": round(eff["params_total_M"], 3),
        "gflops": round(eff["gflops"], 3) if eff["gflops"] == eff["gflops"] else None,
        "epochs": epochs,
    }
    for cls, m in res["per_class"].items():
        row[f"IoU_{cls}"] = round(m["iou"], 4)
    return row


def run_one_experiment(
    base_cfg: Config, name: str, overrides: Dict, results_path: str, runs_root: str,
    epochs: int = 40, subset_train: Optional[int] = None, subset_val: Optional[int] = None,
    device: Optional[str] = None, force: bool = False,
) -> Dict:
    """Train/evaluate a SINGLE named variant and append it to the shared results JSON.

    Resumable: if ``name`` is already in the results (and not ``force``), it's skipped;
    the variant's own ``last.pt`` is resumed if present. Lets you run each variant in its
    own notebook cell.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_results(results_path)
    existing = next((r for r in rows if r.get("name") == name), None)
    if existing is not None and not force:
        print(f"[compare] '{name}' already in results (pass force=True to redo):\n  {existing}")
        return existing
    if existing is not None:
        rows = [r for r in rows if r.get("name") != name]

    print(f"\n========== [variant] {name} ==========", flush=True)
    try:
        row = _run_one(base_cfg, name, overrides, runs_root, epochs, subset_train, subset_val, dev)
    except Exception as e:  # noqa: BLE001
        print(f"[variant] FAILED {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        row = {"name": name, "error": f"{type(e).__name__}: {e}"}
    rows.append(row)
    _save_results(results_path, rows)
    print(f"[variant] {name}: {row}")
    return row


def run_seeds(
    base_cfg: Config, seeds: List[int], results_path: str, runs_root: str,
    overrides: Optional[Dict] = None, epochs: int = 40,
    subset_train: Optional[int] = None, subset_val: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """Train the (chosen) model under several seeds and report mean +/- std.

    Resumable: seeds already recorded are skipped. Returns
    {"rows": [...per-seed...], "summary": {metric: {"mean":, "std":, "n":}}}.
    """
    import statistics

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_results(results_path)
    done = {r["seed"] for r in rows if "seed" in r}

    for s in seeds:
        if s in done:
            print(f"[seeds] skip (done): seed {s}")
            continue
        print(f"\n========== [seed {s}] ==========", flush=True)
        try:
            cfg = override_config(base_cfg, overrides or {})
            cfg.run.seed = s
            set_seed(s)
            row = _run_one(base_cfg=cfg, name=f"seed_{s}", ov={}, runs_root=runs_root,
                           epochs=epochs, subset_train=subset_train, subset_val=subset_val, dev=dev)
            row["seed"] = s
        except Exception as e:  # noqa: BLE001
            print(f"[seeds] FAILED seed {s}: {type(e).__name__}: {e}")
            traceback.print_exc()
            row = {"seed": s, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        _save_results(results_path, rows)

    valid = [r for r in rows if "mIoU" in r]
    metrics = ["mIoU", "mRecall", "IoU_background", "IoU_artisanal", "IoU_industrial",
               "IoU_mining"]
    summary = {}
    for k in metrics:
        vals = [r[k] for r in valid if k in r]
        if vals:
            summary[k] = {"mean": round(statistics.mean(vals), 4),
                          "std": round(statistics.pstdev(vals) if len(vals) == 1
                                       else statistics.stdev(vals), 4),
                          "n": len(vals)}
    print("\n[seeds] summary (mean +/- std):")
    for k, v in summary.items():
        print(f"  {k:>16}: {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")
    return {"rows": rows, "summary": summary}
