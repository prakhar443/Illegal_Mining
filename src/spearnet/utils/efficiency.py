"""Efficiency reporting (Section 5/7): params, GFLOPs, latency, peak VRAM.

These numbers back the deployability claims (<= ~6 M params, < ~15 GFLOPs at 256^2,
sub-second T4 inference).
"""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total_M": total / 1e6, "params_trainable_M": trainable / 1e6}


def _gflops(model: nn.Module, image_size: int, device: torch.device, in_chans: int = 3) -> float:
    """Best-effort GFLOPs via ptflops; returns nan if unavailable."""
    try:
        from ptflops import get_model_complexity_info

        # ptflops expects a model returning a tensor; wrap to extract logits.
        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                return out["logits"] if isinstance(out, dict) else out

        wrapped = _Wrap(model).to(device).eval()
        macs, _ = get_model_complexity_info(
            wrapped, (in_chans, image_size, image_size),
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        return float(macs) * 2 / 1e9  # MACs -> FLOPs -> GFLOPs
    except Exception:
        return float("nan")


@torch.no_grad()
def measure_efficiency(
    model: nn.Module,
    image_size: int = 256,
    device: str = "cuda",
    in_chans: int = None,
    warmup: int = 5,
    iters: int = 30,
) -> Dict[str, float]:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    if in_chans is None:
        in_chans = getattr(model, "in_chans", 3)  # SPEARNet exposes .in_chans
    x = torch.randn(1, in_chans, image_size, image_size, device=dev)

    out = dict(count_parameters(model))
    out["gflops"] = _gflops(model, image_size, dev, in_chans)

    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    out["latency_ms"] = (t1 - t0) / iters * 1000.0
    out["throughput_fps"] = iters / (t1 - t0)
    if dev.type == "cuda":
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    else:
        out["peak_vram_mb"] = float("nan")
    out["device"] = dev.type
    return out
