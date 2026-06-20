"""GeoTIFF chip dataset for the v4 mine-segmentation pipeline.

Consumes the ``manifest.csv`` written by :mod:`spearnet.data.fetch`: one row per chip with
its image/mask paths, the verified ``scale`` (artisanal/industrial) and ``region``
(continent) labels. Builds:

* **binary** masks  {0: background, 1: mining}
* **scale3** masks  {0: background, 1: artisanal, 2: industrial} — mining pixels are
  relabelled by the chip's verified scale.

Region info supports the leave-one-region-out generalization protocol: held-out regions
are removed from train/val and exposed as an ``ood`` split for out-of-region testing.
"""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .transforms import random_flip_rotate, sobel_edges


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_tif(path: str) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read()  # (bands, H, W)


class GeoTiffDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], cfg: Config, train: bool):
        self.rows = rows
        self.cfg = cfg
        self.train = train
        self.size = cfg.data.image_size
        self.bands = cfg.data.bands
        self.scale_div = cfg.data.reflectance_scale
        self.task = cfg.data.task
        self.num_classes = cfg.num_classes

    def __len__(self) -> int:
        return len(self.rows)

    def _scale_id(self, scale: str) -> int:
        return 2 if scale.lower().startswith("indus") else 1  # artisanal=1, industrial=2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        img = _load_tif(row["img"]).astype("float32")[: self.bands]
        img = torch.from_numpy(img) / self.scale_div
        img = img.clamp(0, 1)

        mask = _load_tif(row["mask"])[0].astype("int64")  # (H, W), 0/1
        mask = (mask > 0).astype("int64")
        if self.task == "scale3":
            mask = mask * self._scale_id(row.get("scale", ""))
        mask = torch.from_numpy(mask).long()

        img, mask = self._resize(img, mask)
        if self.train:
            img, mask = random_flip_rotate(img, mask)

        sample = {"image": img.contiguous(), "mask": mask.contiguous()}
        if self.cfg.data.edge_from_mask:
            sample["edge"] = sobel_edges(mask, self.num_classes)
        return sample

    def _resize(self, img, mask):
        if img.shape[-1] != self.size or img.shape[-2] != self.size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.size, self.size), mode="bilinear",
                align_corners=False).squeeze(0)
            mask = torch.nn.functional.interpolate(
                mask[None, None].float(), size=(self.size, self.size),
                mode="nearest")[0, 0].long()
        return img, mask

    def load_mask(self, idx: int) -> np.ndarray:
        """Load & relabel only the mask (no image decode) — for class-weight estimation."""
        row = self.rows[idx]
        mask = (_load_tif(row["mask"])[0] > 0).astype("int64")
        if self.task == "scale3":
            mask = mask * self._scale_id(row.get("scale", ""))
        return mask

    # fast metadata for oversampling / weighting (no image decode)
    def has_rare(self, idx: int) -> bool:
        row = self.rows[idx]
        if self.task == "scale3":
            return row.get("scale", "").lower().startswith("artis")
        return row.get("has_mining", "0") in ("1", "True", "true")


# --------------------------------------------------------------------------------------
def build_mining_examples(cfg: Config) -> Dict[str, List[Dict[str, str]]]:
    """Read the manifest and split rows by train/val/test, honouring region hold-out."""
    manifest = cfg.data.manifest or os.path.join(cfg.data.chips_root or "chips", "manifest.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"manifest not found: {manifest}. Run the fetch step first "
            f"(spearnet.data.fetch.fetch_dataset(..., fetch_imagery=True))."
        )
    rows = read_manifest(manifest)
    holdout = {r.lower() for r in (cfg.data.holdout_regions or [])}
    chips_root = cfg.data.chips_root or os.path.dirname(manifest)

    def _fix(path: str, split: str) -> str:
        """Make manifest paths portable: if the stored absolute path is gone (e.g. chips
        were restored to a different dir), rebuild it under chips_root/split/."""
        if path and os.path.exists(path):
            return path
        return os.path.join(chips_root, split, os.path.basename(path))

    out: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": [], "ood": []}
    for row in rows:
        split = row.get("split", "train")
        row["img"] = _fix(row.get("img", ""), split)
        row["mask"] = _fix(row.get("mask", ""), split)
        region = row.get("region", "").lower()
        if holdout and region in holdout:
            out["ood"].append(row)            # out-of-region test pool
            continue
        out.setdefault(split, []).append(row)
    return out


def _oversample_indices(ds: GeoTiffDataset, factor: int = 3) -> List[int]:
    base = list(range(len(ds)))
    extra: List[int] = []
    for i in range(len(ds)):
        if ds.has_rare(i):
            extra.extend([i] * (factor - 1))
    return base + extra


def build_mining_dataloaders(cfg: Config, splits) -> Dict[str, DataLoader]:
    all_rows = build_mining_examples(cfg)
    caps = {"train": cfg.data.subset_train, "val": cfg.data.subset_val,
            "test": cfg.data.subset_test, "ood": None}

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        rows = list(all_rows.get(split, []))
        if not rows:
            raise ValueError(f"No chips for split '{split}'. Check manifest / holdout_regions.")
        # Shuffle EVERY split deterministically before capping. The manifest is ordered
        # artisanal-first then industrial (separate fetch passes), so capping an unshuffled
        # val/test would drop a whole class from evaluation; shuffling keeps subsets
        # class-representative. (Per-split seed so val != train ordering.)
        _split_off = {"train": 0, "val": 1, "test": 2, "ood": 3}.get(split, 9)
        random.Random(cfg.run.seed + _split_off).shuffle(rows)
        cap = caps.get(split)
        if cap is not None:
            rows = rows[:cap]

        is_train = split == "train"
        ds = GeoTiffDataset(rows, cfg, train=is_train)

        sampler, shuffle = None, is_train
        if is_train and cfg.data.oversample_rare:
            sampler = torch.utils.data.SubsetRandomSampler(_oversample_indices(ds))
            shuffle = False

        loaders[split] = DataLoader(
            ds, batch_size=cfg.data.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=is_train,
        )
    return loaders
