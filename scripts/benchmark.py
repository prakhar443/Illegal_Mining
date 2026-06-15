#!/usr/bin/env python
"""Report params / GFLOPs / latency / peak VRAM for the configured model (Section 5/7)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spearnet.config import load_config
from spearnet.models import build_model
from spearnet.utils import measure_efficiency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = measure_efficiency(model, image_size=args.image_size, device=device)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
