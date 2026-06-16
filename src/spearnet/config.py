"""Dataclass-based experiment configuration with YAML (de)serialization.

A single ``Config`` object fully specifies a run: data, model, loss, optimisation and
logging. YAML files in ``configs/`` only need to override the fields they care about;
everything else falls back to the defaults defined here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------------------
# Task definitions: the LAMES 10-class space, and how to remap it to 3-class / binary.
# --------------------------------------------------------------------------------------
LAMES_CLASSES: List[str] = [
    "other",                 # 0
    "ASM site",              # 1  artisanal / small-scale (illegal-prone)
    "LSM site",              # 2  large-scale
    "heap leaching",         # 3
    "mine facility",         # 4
    "open pit",              # 5
    "processing plant",      # 6
    "stockyard",             # 7
    "tailings storage",      # 8
    "waste rock dump",       # 9
]

# 3-class {other, ASM, LSM}: ASM stays ASM; everything that is a large-scale structure
# collapses to LSM; id 0 stays "other".
THREE_CLASS_NAMES: List[str] = ["other", "ASM", "LSM"]
THREE_CLASS_MAP: Dict[int, int] = {
    0: 0,  # other -> other
    1: 1,  # ASM   -> ASM
    2: 2,  # LSM   -> LSM
    3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,  # all large-scale structures -> LSM
}

BINARY_NAMES: List[str] = ["background", "mining"]
BINARY_MAP: Dict[int, int] = {0: 0, **{i: 1 for i in range(1, 10)}}

# v4: artisanal-vs-industrial scale task. The chip mask is binary {bg, mining}; the
# geotiff dataset turns mining pixels into class 1 (artisanal) or 2 (industrial) using the
# per-chip verified `scale` label, so there is no fixed id remap here.
SCALE3_NAMES: List[str] = ["background", "artisanal", "industrial"]

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # v4 (mine-segmentation) tasks
    "binary": {"num_classes": 2, "names": BINARY_NAMES, "remap": BINARY_MAP},
    "scale3": {"num_classes": 3, "names": SCALE3_NAMES, "remap": None},
    # legacy LAMES tasks (v3)
    "10class": {"num_classes": 10, "names": LAMES_CLASSES, "remap": None},
    "3class": {"num_classes": 3, "names": THREE_CLASS_NAMES, "remap": THREE_CLASS_MAP},
}


@dataclass
class DataConfig:
    # ----- source selection -----
    # "mining" : v4 global mine-segmentation 6-band GeoTIFF chips + manifest (default)
    # "local"  : LAMES zips/folder on disk (v3) ;  "hf" : HuggingFace LAMES (v3)
    source: str = "mining"
    hf_name: str = "maduschek/LAMES"    # used when source == "hf"

    # ----- v4 mine-segmentation (GeoTIFF chips) -----
    chips_root: Optional[str] = "chips"        # dir with train/ val/ test/ subfolders
    manifest: Optional[str] = "chips/manifest.csv"  # per-chip scale/region/split table
    bands: int = 6                             # input channels (6 = B,G,R,NIR,SWIR1,SWIR2)
    band_order: List[str] = field(
        default_factory=lambda: ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]
    )
    reflectance_scale: float = 10000.0         # Sentinel-2 DN -> reflectance divisor
    holdout_regions: List[str] = field(default_factory=list)  # leave-one-region-out
    keep_empty_frac: float = 0.3               # fraction of mining-free chips to keep

    # ----- local source (Google Drive zips extracted to disk) -----

    # ----- local source (Google Drive zips extracted to disk) -----
    local_root: Optional[str] = "data/lames"   # where extracted image/mask files live
    drive_folder_url: Optional[str] = (
        "https://drive.google.com/drive/folders/1A27Fn7HIL8UPQlx-xUmN8Dpzxl4SQsLh"
    )
    zip_glob: str = "*.zip"             # which archives to extract from the source folder
    image_hint: Optional[List[str]] = None  # optional dir/name keywords forcing "image"
    mask_hint: Optional[List[str]] = None   # optional dir/name keywords forcing "mask"
    val_fraction: float = 0.1          # auto-split fractions when no train/val/test dirs
    test_fraction: float = 0.0

    # ----- common -----
    task: str = "10class"               # one of TASK_REGISTRY
    image_size: int = 256
    batch_size: int = 12
    num_workers: int = 2
    streaming: bool = False             # (hf only) stream instead of full download
    subset_train: Optional[int] = 4000  # cap #train patches for dev; None = full
    subset_val: Optional[int] = 1000
    subset_test: Optional[int] = None
    oversample_rare: bool = True        # oversample patches containing rare classes
    rare_class_ids: List[int] = field(default_factory=lambda: [3, 6, 8])
    edge_from_mask: bool = True         # derive Sobel edge target from the mask


@dataclass
class ModelConfig:
    name: str = "spearnet"              # spearnet | unet | attn_unet | deeplabv3p | unetpp
    backbone: str = "mobilenetv3_small_100"  # timm name; or efficientnet_lite0
    pretrained: bool = True
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 48, 32])
    # Spectral prior
    prior_type: str = "pisp"           # pisp (S2 spectral) | csp (RGB-only ablation)
    csp_mode: str = "gate"             # gate | concat | none  (integration mode)
    csp_alpha_init: float = 1.0        # initial gate strength (per-scale, learnable)
    use_edge_head: bool = True


@dataclass
class LossConfig:
    # Compound loss weights.  ce > 0 reproduces the plain-CrossEntropy ablation.
    w_ce: float = 0.0
    w_dice: float = 1.0
    w_tversky: float = 1.0
    w_boundary: float = 0.5
    w_edge: float = 0.5
    w_focal: float = 1.0               # focal term: per-pixel signal for rare classes
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3         # alpha < beta => penalise false negatives (recall)
    tversky_beta: float = 0.7
    present_only: bool = True          # average region losses over present classes only
    class_weights: Optional[List[float]] = None  # manual per-class weights for CE/focal
    auto_class_weights: bool = True    # estimate inverse-frequency weights from a sample
    auto_weight_samples: int = 500     # #train masks sampled to estimate frequencies
    ignore_index: int = -100


@dataclass
class OptimConfig:
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    amp: bool = True                   # mixed precision
    grad_clip: float = 1.0
    accumulate: int = 1


@dataclass
class RunConfig:
    seed: int = 42
    out_dir: str = "runs/spearnet"
    log_interval: int = 20
    val_interval: int = 1              # epochs
    save_best_metric: str = "miou"
    device: str = "cuda"               # falls back to cpu automatically
    resume: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ----- task helpers -----
    @property
    def num_classes(self) -> int:
        return TASK_REGISTRY[self.data.task]["num_classes"]

    @property
    def class_names(self) -> List[str]:
        return TASK_REGISTRY[self.data.task]["names"]

    @property
    def class_remap(self) -> Optional[Dict[int, int]]:
        return TASK_REGISTRY[self.data.task]["remap"]

    # ----- (de)serialization -----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        def build(dc_type, data):
            data = data or {}
            valid = {f.name for f in dataclasses.fields(dc_type)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
            return dc_type(**{k: v for k, v in data.items() if k in valid})

        return cls(
            data=build(DataConfig, raw.get("data")),
            model=build(ModelConfig, raw.get("model")),
            loss=build(LossConfig, raw.get("loss")),
            optim=build(OptimConfig, raw.get("optim")),
            run=build(RunConfig, raw.get("run")),
        )


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a config from YAML (or defaults) and apply dotted ``overrides``.

    ``overrides`` keys use dotted paths, e.g. ``{"optim.epochs": 5, "data.task": "binary"}``.
    """
    cfg = Config.load(path) if path else Config()
    if overrides:
        d = cfg.to_dict()
        for key, value in overrides.items():
            cur = d
            parts = key.split(".")
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = value
        cfg = Config.from_dict(d)
    return cfg
