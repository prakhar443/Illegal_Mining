#!/usr/bin/env python
"""Efficiency + operational-feasibility benchmark (reviewer items B6, B9, F10, F11).

For SPEAR-Net and each baseline: params, GFLOPs, latency (ms/chip), throughput (chips/s)
and peak VRAM on GPU and (optionally) CPU. Then derive the operationally meaningful figure:
hours to screen one Sentinel-2 tile (110x110 km ~ 1,850 chips at 256^2) and a national AOI.

Usage:
    python scripts/benchmark.py --config configs/v4_scale3.yaml \
        --models spearnet unet deeplabv3p unetpp --cpu --out benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spearnet.config import load_config, override_config
from spearnet.models import build_model
from spearnet.utils import measure_efficiency

# One Sentinel-2 tile at 10 m ~ 10,980^2 px; non-overlapping 256^2 chips:
CHIPS_PER_S2_TILE = (10980 // 256) ** 2   # = 1849


def _profile(cfg, name, device, iters):
    over = {"model.name": "spearnet"} if name == "spearnet" else {"model.name": name}
    if name != "spearnet":
        over["model.backbone"] = "resnet34"
    c = override_config(cfg, over)
    c.model.pretrained = False
    model = build_model(c)
    eff = measure_efficiency(model, c.data.image_size, device, in_chans=c.data.bands,
                             iters=iters, warmup=5)
    eff["throughput_chips_per_s"] = round(1000.0 / eff["latency_ms"], 2) if eff["latency_ms"] else None
    eff["model"] = name
    eff["device"] = device
    return eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v4_scale3.yaml")
    ap.add_argument("--models", nargs="+",
                    default=["spearnet", "unet", "deeplabv3p", "unetpp"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cpu", action="store_true", help="also benchmark on CPU")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.data.image_size = args.image_size
    devices = ["cuda"] if torch.cuda.is_available() else []
    if args.cpu or not devices:
        devices.append("cpu")

    rows = []
    for dev in devices:
        for name in args.models:
            try:
                r = _profile(cfg, name, dev, args.iters)
                rows.append(r)
                print(f"[{dev}] {name:>11}: {r['params_total_M']:.2f} M | "
                      f"{r['gflops']:.2f} GFLOPs | {r['latency_ms']:.1f} ms | "
                      f"{r['throughput_chips_per_s']} chips/s | "
                      f"VRAM {r.get('peak_vram_mb', float('nan')):.0f} MB")
            except Exception as e:  # noqa: BLE001
                print(f"[{dev}] {name}: FAILED {type(e).__name__}: {e}")
                rows.append({"model": name, "device": dev, "error": str(e)})

    # B9 / F10: hours to screen one Sentinel-2 tile (GPU rows)
    print(f"\nB9  operational feasibility  ({CHIPS_PER_S2_TILE} chips per S2 tile):")
    feas = {}
    for r in rows:
        if r.get("device") == ("cuda" if torch.cuda.is_available() else "cpu") \
                and r.get("throughput_chips_per_s"):
            sec = CHIPS_PER_S2_TILE / r["throughput_chips_per_s"]
            feas[r["model"]] = {"sec_per_s2_tile": round(sec, 1),
                                "min_per_s2_tile": round(sec / 60, 2)}
            print(f"    {r['model']:>11}: {sec/60:.2f} min/tile "
                  f"({r['throughput_chips_per_s']} chips/s)")
    # honest FLOP ratios (F11)
    sp = next((r for r in rows if r.get("model") == "spearnet" and "gflops" in r), None)
    if sp:
        print("\nF11 efficiency ratios vs SPEAR-Net (report the honest range, not '30x'):")
        for r in rows:
            if r.get("model") not in (None, "spearnet") and r.get("gflops") and r.get("device") == sp["device"]:
                print(f"    {r['model']:>11}: {r['params_total_M']/sp['params_total_M']:.1f}x params, "
                      f"{r['gflops']/sp['gflops']:.1f}x FLOPs")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "chips_per_s2_tile": CHIPS_PER_S2_TILE,
                   "feasibility": feas}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
