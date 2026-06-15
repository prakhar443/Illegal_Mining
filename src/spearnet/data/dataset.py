"""LAMES dataset wrapper — Section 1 & 5 of the methodology.

Zero preprocessing: we load ready image+mask tiles straight from Hugging Face
(``maduschek/LAMES``), scale RGB to [0, 1], optionally remap the mask to the 3-class or
binary task, and (optionally) oversample patches that contain rare classes.

Robustness notes baked in (from the methodology's risk table):
  * handles PIL palette ('P'), grayscale ('L') and RGB-encoded masks;
  * verifies the label range and clamps unexpected ids;
  * supports streaming and SUBSET caps so the ~20 GB set fits free-tier Colab.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .transforms import random_flip_rotate, sobel_edges

# Candidate field names in the HF dataset (auto-detected).
_IMAGE_KEYS = ["image", "img", "rgb", "input"]
_MASK_KEYS = ["mask", "label", "labels", "segmentation", "annotation", "seg"]
# LAMES uses non-standard split names in some revisions; map them.
_SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "validation", "valid", "dev"],
    "test": ["test", "testing", "eval"],
}


def _pick_key(example: Dict, candidates: List[str], kind: str) -> str:
    for k in candidates:
        if k in example:
            return k
    # fall back to any key whose value looks like a PIL image
    for k, v in example.items():
        if hasattr(v, "size") and hasattr(v, "mode"):
            return k
    raise KeyError(f"Could not find a '{kind}' field; available keys: {list(example.keys())}")


def _resolve_split(dsdict, want: str):
    available = list(dsdict.keys())
    for alias in _SPLIT_ALIASES[want]:
        if alias in dsdict:
            return dsdict[alias]
    # streaming IterableDatasetDict or unusual names: best-effort positional fallback
    if want == "train" and available:
        return dsdict[available[0]]
    if want == "val" and len(available) > 1:
        return dsdict[available[1]]
    if want == "test" and len(available) > 2:
        return dsdict[available[2]]
    raise KeyError(f"Split '{want}' not found; available: {available}")


def remap_mask(mask: np.ndarray, remap: Optional[Dict[int, int]], num_classes: int) -> np.ndarray:
    """Remap raw LAMES ids to the active task and clamp to a valid range."""
    mask = mask.astype(np.int64)
    if remap is not None:
        out = np.zeros_like(mask)
        for src, dst in remap.items():
            out[mask == src] = dst
        mask = out
    # Guard against stray ids (risk: mask encoding mismatch).
    mask = np.clip(mask, 0, num_classes - 1)
    return mask


def _mask_to_array(mask_obj) -> np.ndarray:
    """Convert a PIL mask (P/L/RGB), file path, or array-like into an (H, W) id array."""
    if isinstance(mask_obj, str):
        from PIL import Image

        mask_obj = Image.open(mask_obj)
    if isinstance(mask_obj, np.ndarray):
        arr = mask_obj
    elif hasattr(mask_obj, "mode"):  # PIL Image
        mode = mask_obj.mode
        if mode in ("P", "L", "I", "I;16"):
            arr = np.array(mask_obj)
        elif mode in ("RGB", "RGBA"):
            # Some exports encode class id in the red channel; collapse to it.
            arr = np.array(mask_obj)[..., 0]
        else:
            arr = np.array(mask_obj.convert("L"))
    else:
        arr = np.asarray(mask_obj)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _image_to_tensor(img_obj, size: int) -> torch.Tensor:
    """Convert a PIL/array/path RGB image to a (3, H, W) float tensor in [0, 1]."""
    if isinstance(img_obj, str):
        from PIL import Image

        img_obj = Image.open(img_obj)
    if hasattr(img_obj, "mode"):
        if img_obj.mode != "RGB":
            img_obj = img_obj.convert("RGB")
        arr = np.array(img_obj)
    else:
        arr = np.asarray(img_obj)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3]
    t = torch.from_numpy(arr).float()
    if t.max() > 1.5:  # uint8 imagery
        t = t / 255.0
    t = t.permute(2, 0, 1).contiguous()  # (3, H, W)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return t


class LAMESDataset(Dataset):
    """Map-style dataset over an in-memory (materialised) list of LAMES examples.

    For streaming use, materialise a capped subset first (see :func:`build_dataloaders`).
    """

    def __init__(
        self,
        examples: List[Dict],
        cfg: Config,
        train: bool,
        image_key: Optional[str] = None,
        mask_key: Optional[str] = None,
    ):
        self.examples = examples
        self.cfg = cfg
        self.train = train
        self.size = cfg.data.image_size
        self.num_classes = cfg.num_classes
        self.remap = cfg.class_remap

        if examples:
            self.image_key = image_key or _pick_key(examples[0], _IMAGE_KEYS, "image")
            self.mask_key = mask_key or _pick_key(examples[0], _MASK_KEYS, "mask")
        else:
            self.image_key, self.mask_key = image_key or "image", mask_key or "mask"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        image = _image_to_tensor(ex[self.image_key], self.size)

        raw_mask = _mask_to_array(ex[self.mask_key])
        mask = remap_mask(raw_mask, self.remap, self.num_classes)
        mask = torch.from_numpy(mask).long()
        if mask.shape[-1] != self.size or mask.shape[-2] != self.size:
            mask = torch.nn.functional.interpolate(
                mask[None, None].float(), size=(self.size, self.size), mode="nearest"
            )[0, 0].long()

        if self.train:
            image, mask = random_flip_rotate(image, mask)

        sample = {"image": image, "mask": mask}
        if self.cfg.data.edge_from_mask:
            sample["edge"] = sobel_edges(mask, self.num_classes)  # (1, H, W)
        return sample


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _materialise(split, cap: Optional[int], streaming: bool) -> List[Dict]:
    """Pull up to ``cap`` examples into memory from a (possibly streaming) HF split."""
    out: List[Dict] = []
    if streaming:
        for i, ex in enumerate(split):
            if cap is not None and i >= cap:
                break
            out.append(ex)
    else:
        n = len(split)
        idxs = list(range(n))
        if cap is not None and cap < n:
            idxs = idxs[:cap]
        for i in idxs:
            out.append(split[i])
    return out


def _rare_oversample_indices(
    dataset: LAMESDataset, rare_ids: List[int], factor: int = 3
) -> List[int]:
    """Build an index list that repeats samples containing rare classes ``factor`` times."""
    base = list(range(len(dataset)))
    extra: List[int] = []
    rare_set = set(rare_ids)
    for i in range(len(dataset)):
        # cheap check on the (already remapped) mask
        m = dataset[i]["mask"]
        present = set(torch.unique(m).tolist())
        if present & rare_set:
            extra.extend([i] * (factor - 1))
    return base + extra


def _split_examples(cfg: Config) -> Dict[str, List[Dict]]:
    """Return {split: [ {image, mask}, ... ]} for the configured data source."""
    caps = {
        "train": cfg.data.subset_train,
        "val": cfg.data.subset_val,
        "test": cfg.data.subset_test,
    }

    if cfg.data.source == "local":
        from .local import prepare_local_dataset

        source = cfg.data.local_root
        if source is None:
            raise ValueError("data.local_root must be set for source='local'")
        work = source.rstrip("/") + "_extracted"
        split_pairs = prepare_local_dataset(
            source_dir=source,
            work_dir=work,
            zip_glob=cfg.data.zip_glob,
            image_hint=cfg.data.image_hint,
            mask_hint=cfg.data.mask_hint,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            seed=cfg.run.seed,
        )
        out: Dict[str, List[Dict]] = {}
        for split, pairs in split_pairs.items():
            cap = caps.get(split)
            if cap is not None:
                pairs = pairs[:cap]
            out[split] = [{"image": ip, "mask": mp} for ip, mp in pairs]
        return out

    # ----- Hugging Face source -----
    from datasets import load_dataset  # lazy import: smoke tests need no network

    dsdict = load_dataset(cfg.data.hf_name, streaming=cfg.data.streaming)
    out = {}
    for split in ("train", "val", "test"):
        try:
            hf_split = _resolve_split(dsdict, split)
        except KeyError:
            out[split] = []
            continue
        out[split] = _materialise(hf_split, caps[split], cfg.data.streaming)
    return out


def build_dataloaders(
    cfg: Config, splits: Tuple[str, ...] = ("train", "val")
) -> Dict[str, DataLoader]:
    """Build PyTorch dataloaders from the configured source (local zips or HuggingFace).

    Returns a dict mapping split name -> DataLoader for each requested split.
    """
    all_examples = _split_examples(cfg)

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        examples = all_examples.get(split, [])
        if not examples:
            raise ValueError(
                f"No examples found for split '{split}'. Check data.source / local_root."
            )
        is_train = split == "train"
        ds = LAMESDataset(examples, cfg, train=is_train)

        sampler = None
        shuffle = is_train
        if is_train and cfg.data.oversample_rare and cfg.data.rare_class_ids:
            idxs = _rare_oversample_indices(ds, cfg.data.rare_class_ids)
            sampler = torch.utils.data.SubsetRandomSampler(idxs)
            shuffle = False

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=is_train,
        )
    return loaders
