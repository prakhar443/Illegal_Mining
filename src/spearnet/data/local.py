"""Local dataset ingestion for the Google-Drive-hosted LAMES zips.

The Drive folder ships the dataset as one or more ``.zip`` archives, but the exact
internal layout is not known ahead of time. This module therefore:

  1. (optionally) downloads the Drive folder with ``gdown`` (public-link mode), or works
     against an already-mounted Drive path;
  2. extracts the zips into a working directory;
  3. **auto-detects** image/mask pairs with a content-based classifier that does not rely
     on a fixed folder structure — masks are recognised as low-cardinality, small-valued,
     single-channel rasters; images as multi-channel, broad-valued rasters;
  4. respects explicit ``train`` / ``val`` / ``test`` split directories if present,
     otherwise produces a deterministic random split.

Every step prints a summary so the detection can be verified before training.
"""
from __future__ import annotations

import os
import random
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Keyword hints (used only to break ties; detection is primarily content-based).
_MASK_KEYWORDS = ["mask", "label", "gt", "annot", "seg", "target", "class", "_m"]
_IMAGE_KEYWORDS = ["image", "img", "rgb", "input", "sat", "tile", "patch", "scene", "_x"]

# Tokens stripped from a filename stem when matching an image to its mask.
_STRIP_TOKENS = ["_mask", "_label", "_gt", "_seg", "_annotation", "_annot",
                 "_target", "_class", "_image", "_img", "_rgb", "_x", "_m",
                 "mask", "label", "image", "img"]


# --------------------------------------------------------------------------------------
# Download / extract
# --------------------------------------------------------------------------------------
def download_drive_folder(url: str, dest: str) -> str:
    """Download a *public* Google Drive folder with gdown. Returns the local path.

    For a private folder, mount Drive in Colab instead and pass the mounted path as the
    source directory to :func:`prepare_local_dataset`.
    """
    import gdown

    os.makedirs(dest, exist_ok=True)
    gdown.download_folder(url=url, output=dest, quiet=False, use_cookies=False)
    return dest


def extract_zips(source_dir: str, work_dir: str, zip_glob: str = "*.zip") -> str:
    """Extract every archive matching ``zip_glob`` under ``source_dir`` into ``work_dir``.

    Already-extracted archives are skipped (idempotent via a per-zip marker file).
    Returns ``work_dir``.
    """
    import glob

    os.makedirs(work_dir, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(source_dir, "**", zip_glob), recursive=True))
    if not zips:
        # Maybe the source dir already holds extracted files, not zips.
        print(f"[extract] no archives matching '{zip_glob}' under {source_dir}; "
              f"assuming the folder is already extracted.")
        return source_dir

    for zp in zips:
        marker = os.path.join(work_dir, "." + os.path.basename(zp) + ".done")
        if os.path.exists(marker):
            print(f"[extract] skip (already done): {os.path.basename(zp)}")
            continue
        print(f"[extract] {os.path.basename(zp)} -> {work_dir}")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(work_dir)
        open(marker, "w").close()
    return work_dir


# --------------------------------------------------------------------------------------
# Auto-detection of image/mask pairs
# --------------------------------------------------------------------------------------
def _list_rasters(root: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _classify(path: str, sample_cache: Dict[str, dict]) -> str:
    """Return 'mask' or 'image' for a raster, using keywords then content."""
    low = path.lower()
    kw_mask = any(k in low for k in _MASK_KEYWORDS)
    kw_image = any(k in low for k in _IMAGE_KEYWORDS)
    if kw_mask and not kw_image:
        return "mask"
    if kw_image and not kw_mask:
        return "image"

    # Content-based fallback.
    info = sample_cache.get(path)
    if info is None:
        info = _probe(path)
        sample_cache[path] = info
    # Masks: single channel, small max value, few unique values.
    if info["channels"] == 1 and info["max"] <= 40 and info["n_unique"] <= 25:
        return "mask"
    return "image"


def _probe(path: str) -> dict:
    from PIL import Image

    try:
        with Image.open(path) as im:
            mode = im.mode
            arr = np.asarray(im)
    except Exception:
        return {"channels": 3, "max": 255, "n_unique": 256}
    channels = 1 if arr.ndim == 2 else arr.shape[-1]
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # treat near-grayscale 3-channel as 3-channel image regardless
        channels = arr.shape[-1]
    sub = arr.reshape(-1, channels)[::97] if arr.ndim >= 1 else arr  # subsample for speed
    return {
        "channels": channels,
        "max": int(np.max(arr)) if arr.size else 0,
        "n_unique": int(np.unique(sub).size),
        "mode": mode,
    }


def _norm_key(path: str) -> str:
    """Normalise a filename stem so an image and its mask map to the same key."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for tok in sorted(_STRIP_TOKENS, key=len, reverse=True):
        stem = stem.replace(tok, "")
    return stem.strip("_-. ")


@dataclass
class PairSet:
    pairs: List[Tuple[str, str]]   # (image_path, mask_path)
    split: str


def discover_pairs(
    root: str,
    image_hint: Optional[List[str]] = None,
    mask_hint: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Find (image, mask) pairs under ``root`` regardless of folder layout."""
    global _MASK_KEYWORDS, _IMAGE_KEYWORDS
    if mask_hint:
        _MASK_KEYWORDS = list(dict.fromkeys(_MASK_KEYWORDS + [h.lower() for h in mask_hint]))
    if image_hint:
        _IMAGE_KEYWORDS = list(dict.fromkeys(_IMAGE_KEYWORDS + [h.lower() for h in image_hint]))

    rasters = _list_rasters(root)
    if not rasters:
        raise FileNotFoundError(f"No image files found under {root}")

    cache: Dict[str, dict] = {}
    images, masks = [], []
    for p in rasters:
        (masks if _classify(p, cache) == "mask" else images).append(p)

    # ---- pairing strategies ----
    pairs = _pair_by_key(images, masks)
    if not pairs:
        pairs = _pair_by_parallel_order(images, masks)
    if not pairs:
        raise RuntimeError(
            f"Found {len(images)} image-like and {len(masks)} mask-like files but could "
            f"not pair them. Set data.image_hint / data.mask_hint to disambiguate."
        )
    return pairs


def _pair_by_key(images: List[str], masks: List[str]) -> List[Tuple[str, str]]:
    mask_by_key: Dict[str, str] = {}
    for m in masks:
        mask_by_key.setdefault(_norm_key(m), m)
    pairs = []
    for img in images:
        m = mask_by_key.get(_norm_key(img))
        if m is not None:
            pairs.append((img, m))
    return pairs


def _pair_by_parallel_order(images: List[str], masks: List[str]) -> List[Tuple[str, str]]:
    """Fallback: equal counts -> pair by sorted relative path (mirrored trees)."""
    if len(images) != len(masks) or not images:
        return []
    imgs = sorted(images, key=lambda p: _norm_key(p))
    msks = sorted(masks, key=lambda p: _norm_key(p))
    return list(zip(imgs, msks))


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def _split_of(path: str) -> Optional[str]:
    low = path.lower().replace("\\", "/")
    parts = low.split("/")
    for token, name in [("train", "train"), ("valid", "val"), ("val", "val"),
                        ("dev", "val"), ("test", "test")]:
        if any(token == p or token in p for p in parts):
            return name
    return None


def split_pairs(
    pairs: List[Tuple[str, str]],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """Use explicit split dirs if the paths reveal them, else random-split deterministically."""
    by_split: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    tagged = 0
    for img, msk in pairs:
        s = _split_of(img) or _split_of(msk)
        if s:
            by_split[s].append((img, msk))
            tagged += 1

    if tagged >= 0.5 * len(pairs) and by_split["train"]:
        # Honour the dataset's own split directories.
        if not by_split["val"] and val_fraction > 0:
            by_split["val"] = _carve(by_split["train"], val_fraction, seed)
        return by_split

    # Otherwise random split.
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def _carve(items: List, frac: float, seed: int) -> List:
    rng = random.Random(seed)
    items_sorted = sorted(items)
    rng.shuffle(items_sorted)
    k = int(len(items_sorted) * frac)
    carved = items_sorted[:k]
    remaining = set(map(tuple, items_sorted[k:]))
    items[:] = [it for it in items if tuple(it) in remaining]
    return carved


def prepare_local_dataset(
    source_dir: str,
    work_dir: str,
    zip_glob: str = "*.zip",
    image_hint: Optional[List[str]] = None,
    mask_hint: Optional[List[str]] = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """End-to-end: extract zips (if any) -> discover pairs -> split. Prints a summary."""
    extracted = extract_zips(source_dir, work_dir, zip_glob)
    pairs = discover_pairs(extracted, image_hint, mask_hint)
    splits = split_pairs(pairs, val_fraction, test_fraction, seed)

    print("\n=== LAMES local dataset summary ===")
    print(f"  root          : {extracted}")
    print(f"  total pairs   : {len(pairs)}")
    for s in ("train", "val", "test"):
        print(f"  {s:<6} pairs  : {len(splits[s])}")
    if pairs:
        ex_img, ex_msk = pairs[0]
        print(f"  example image : {ex_img}")
        print(f"  example mask  : {ex_msk}")
        try:
            from PIL import Image
            uniq = np.unique(np.asarray(Image.open(ex_msk)))
            print(f"  mask uniques  : {uniq[:15].tolist()}{' ...' if uniq.size > 15 else ''}")
        except Exception:
            pass
    print("===================================\n")
    return splits
