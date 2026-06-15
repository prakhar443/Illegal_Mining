#!/usr/bin/env python
"""Extract the LAMES zips and auto-detect image/mask pairs (verification step).

Use this before training to confirm the dataset was detected correctly.

Examples:
    # From a mounted Google Drive folder (Colab):
    python scripts/prepare_data.py --source /content/drive/MyDrive/LAMES --work data/lames

    # From a public Drive folder via gdown:
    python scripts/prepare_data.py --drive-url "https://drive.google.com/drive/folders/XXXX" \
        --download-to data/_drive --work data/lames
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spearnet.data.local import download_drive_folder, prepare_local_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None,
                    help="folder containing the .zip files (e.g. a mounted Drive path)")
    ap.add_argument("--drive-url", type=str, default=None,
                    help="public Drive folder URL to download with gdown")
    ap.add_argument("--download-to", type=str, default="data/_drive")
    ap.add_argument("--work", type=str, default="data/lames")
    ap.add_argument("--zip-glob", type=str, default="*.zip")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.0)
    ap.add_argument("--image-hint", nargs="*", default=None)
    ap.add_argument("--mask-hint", nargs="*", default=None)
    args = ap.parse_args()

    source = args.source
    if source is None:
        if not args.drive_url:
            ap.error("provide either --source (local/mounted folder) or --drive-url")
        source = download_drive_folder(args.drive_url, args.download_to)

    prepare_local_dataset(
        source_dir=source,
        work_dir=args.work,
        zip_glob=args.zip_glob,
        image_hint=args.image_hint,
        mask_hint=args.mask_hint,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print("Prepared. Point configs at data.local_root =", args.work)


if __name__ == "__main__":
    main()
