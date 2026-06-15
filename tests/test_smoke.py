"""Smoke tests that exercise the full pipeline on random tensors — no GPU, no dataset.

Run: pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from spearnet.config import Config, load_config
from spearnet.data.priors import compute_csp_priors, N_CSP_PRIORS
from spearnet.data.transforms import sobel_edges, random_flip_rotate
from spearnet.data.dataset import remap_mask, LAMESDataset
from spearnet.losses import CompoundLoss
from spearnet.metrics import SegMetrics, area_stratified_recall
from spearnet.models import build_model
import numpy as np


def _cfg(**model_over):
    cfg = Config()
    cfg.data.image_size = 64
    cfg.data.batch_size = 2
    cfg.model.pretrained = False  # no network in CI
    for k, v in model_over.items():
        setattr(cfg.model, k, v)
    return cfg


def test_csp_priors_shape_and_range():
    x = torch.rand(2, 3, 32, 32)
    p = compute_csp_priors(x)
    assert p.shape == (2, N_CSP_PRIORS, 32, 32)
    assert torch.isfinite(p).all()


def test_sobel_edges():
    mask = torch.zeros(16, 16, dtype=torch.long)
    mask[4:12, 4:12] = 1
    e = sobel_edges(mask, num_classes=2)
    assert e.shape == (1, 16, 16)
    assert e.max() == 1.0 and e.min() == 0.0


def test_remap_3class_and_binary():
    m = np.arange(10).reshape(2, 5)
    from spearnet.config import THREE_CLASS_MAP, BINARY_MAP
    r3 = remap_mask(m, THREE_CLASS_MAP, 3)
    assert set(np.unique(r3)).issubset({0, 1, 2})
    rb = remap_mask(m, BINARY_MAP, 2)
    assert set(np.unique(rb)).issubset({0, 1})


def test_spearnet_forward_backward_gate():
    cfg = _cfg(csp_mode="gate", use_edge_head=True)
    model = build_model(cfg)
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out["logits"].shape == (2, cfg.num_classes, 64, 64)
    assert "edge" in out and out["edge"].shape == (2, 1, 64, 64)
    assert "attn" in out
    out["logits"].sum().backward()  # gradients flow


def test_spearnet_concat_and_none():
    for mode in ("concat", "none"):
        cfg = _cfg(csp_mode=mode, use_edge_head=False)
        model = build_model(cfg)
        out = model(torch.rand(2, 3, 64, 64))
        assert out["logits"].shape[1] == cfg.num_classes
        assert "edge" not in out


def test_compound_loss():
    cfg = _cfg(csp_mode="gate", use_edge_head=True)
    model = build_model(cfg)
    crit = CompoundLoss(cfg.num_classes)
    x = torch.rand(2, 3, 64, 64)
    batch = {
        "image": x,
        "mask": torch.randint(0, cfg.num_classes, (2, 64, 64)),
        "edge": torch.randint(0, 2, (2, 1, 64, 64)).float(),
    }
    out = model(x)
    loss, comps = crit(out, batch)
    assert torch.isfinite(loss)
    assert {"dice", "tversky", "boundary", "edge", "total"} <= set(comps)
    loss.backward()


def test_metrics():
    m = SegMetrics(3, ["a", "b", "c"])
    logits = torch.randn(2, 3, 16, 16)
    target = torch.randint(0, 3, (2, 16, 16))
    m.update(logits, target)
    res = m.compute()
    assert 0.0 <= res["miou"] <= 1.0
    assert len(res["per_class"]) == 3


def test_area_stratified_recall():
    preds = np.zeros((1, 16, 16), dtype=np.int64)
    targets = np.zeros((1, 16, 16), dtype=np.int64)
    targets[0, 2:5, 2:5] = 1
    preds[0, 2:5, 2:5] = 1
    res = area_stratified_recall(preds, targets, num_classes=2)
    assert res["small"] == 1.0


def test_dataset_getitem_with_fake_examples():
    from PIL import Image

    examples = []
    for _ in range(3):
        img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8), "RGB")
        msk = Image.fromarray(np.random.randint(0, 10, (64, 64)).astype(np.uint8), "L")
        examples.append({"image": img, "mask": msk})
    cfg = _cfg()
    ds = LAMESDataset(examples, cfg, train=True)
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["edge"].shape == (1, 64, 64)


def test_config_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "c.yaml"
    cfg.save(str(p))
    cfg2 = Config.load(str(p))
    assert cfg2.num_classes == cfg.num_classes


def test_override_loader(tmp_path):
    cfg = load_config(None, {"optim.epochs": 3, "data.task": "binary"})
    assert cfg.optim.epochs == 3
    assert cfg.data.task == "binary"
    assert cfg.num_classes == 2


# --------------------------------------------------------------------------------------
# Local (Google Drive) dataset auto-detection
# --------------------------------------------------------------------------------------
def _write_pair(img_path, mask_path, mask_vals=(0, 1, 8)):
    from PIL import Image
    import os as _os

    _os.makedirs(_os.path.dirname(img_path), exist_ok=True)
    _os.makedirs(_os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8), "RGB").save(img_path)
    m = np.zeros((32, 32), np.uint8)
    for i, v in enumerate(mask_vals):
        m[i * 4:(i + 1) * 4, :] = v
    Image.fromarray(m, "L").save(mask_path)


def test_discover_separate_folders(tmp_path):
    from spearnet.data.local import discover_pairs
    root = tmp_path / "ds"
    for i in range(6):
        _write_pair(str(root / "images" / f"tile_{i}.png"),
                    str(root / "masks" / f"tile_{i}.png"))
    pairs = discover_pairs(str(root))
    assert len(pairs) == 6
    for img, msk in pairs:
        assert "image" in img and "mask" in msk


def test_discover_suffix_same_folder(tmp_path):
    from spearnet.data.local import discover_pairs
    root = tmp_path / "ds2"
    for i in range(5):
        _write_pair(str(root / f"patch_{i}.png"),
                    str(root / f"patch_{i}_mask.png"))
    pairs = discover_pairs(str(root))
    assert len(pairs) == 5


def test_split_pairs_with_split_dirs(tmp_path):
    from spearnet.data.local import discover_pairs, split_pairs
    root = tmp_path / "ds3"
    for split, n in [("train", 8), ("val", 3)]:
        for i in range(n):
            _write_pair(str(root / split / "images" / f"t{i}.png"),
                        str(root / split / "masks" / f"t{i}.png"))
    pairs = discover_pairs(str(root))
    splits = split_pairs(pairs, val_fraction=0.1, test_fraction=0.0)
    assert len(splits["train"]) == 8 and len(splits["val"]) == 3


def test_extract_and_prepare(tmp_path):
    import zipfile
    from spearnet.data.local import prepare_local_dataset
    # Build a zip that contains images/ + masks/
    raw = tmp_path / "raw"
    for i in range(10):
        _write_pair(str(raw / "images" / f"x{i}.png"), str(raw / "masks" / f"x{i}.png"))
    zpath = tmp_path / "src" / "lames_part1.zip"
    os.makedirs(zpath.parent, exist_ok=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in raw.rglob("*.png"):
            zf.write(p, arcname=str(p.relative_to(raw)))

    splits = prepare_local_dataset(
        source_dir=str(zpath.parent), work_dir=str(tmp_path / "work"),
        val_fraction=0.2, test_fraction=0.0, seed=0,
    )
    total = sum(len(v) for v in splits.values())
    assert total == 10
    assert len(splits["val"]) == 2 and len(splits["train"]) == 8


def test_local_dataset_pathexamples(tmp_path):
    from spearnet.data.dataset import LAMESDataset
    root = tmp_path / "ds4"
    _write_pair(str(root / "images" / "a.png"), str(root / "masks" / "a.png"))
    cfg = _cfg()
    examples = [{"image": str(root / "images" / "a.png"),
                 "mask": str(root / "masks" / "a.png")}]
    ds = LAMESDataset(examples, cfg, train=False)
    s = ds[0]
    assert s["image"].shape == (3, 64, 64)
    assert s["mask"].shape == (64, 64)
