#!/usr/bin/env python
"""Fetch & store the global mine-segmentation dataset (v4).

Examples:
    # metadata only — prints artisanal/industrial counts
    python scripts/fetch_data.py --metadata-only

    # fetch all artisanal tiles, then a capped industrial set (appends to the manifest)
    python scripts/fetch_data.py --scale artisanal  --subset 0  --out chips
    python scripts/fetch_data.py --scale industrial --subset 80 --out chips
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spearnet.data.fetch import fetch_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chips", help="output dir for chips + manifest")
    ap.add_argument("--annot-dir", default="mine_data")
    ap.add_argument("--metadata-only", action="store_true", help="print counts, no fetch")
    ap.add_argument("--scale", default=None, choices=[None, "artisanal", "industrial"],
                    help="filter tiles by verified scale")
    ap.add_argument("--subset", type=int, default=60,
                    help="tiles per split (0 or negative = all)")
    ap.add_argument("--chip-size", type=int, default=256)
    ap.add_argument("--keep-empty-frac", type=float, default=0.3)
    args = ap.parse_args()

    subset = None if args.subset is not None and args.subset <= 0 else args.subset
    fetch_dataset(
        out_dir=args.out,
        annot_dir=args.annot_dir,
        fetch_imagery=not args.metadata_only,
        scale_filter=args.scale,
        subset_per_split=subset,
        chip_size=args.chip_size,
        keep_empty_frac=args.keep_empty_frac,
    )


if __name__ == "__main__":
    main()
