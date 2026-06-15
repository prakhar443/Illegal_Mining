# SPEAR-Net

**S**pectral-**P**rior **E**dge-**A**ware **R**ecall-optimized **Net**work — a lightweight,
color-prior-guided, recall-optimized network for fine-grained segmentation of mining
structures, with emphasis on **A**rtisanal & **S**mall-scale (illegal-prone) **M**ining.

> Zero-preprocessing pipeline on the ready-made **LAMES** image+mask dataset, single-image
> multi-class semantic segmentation, designed to train end-to-end on free Google Colab
> (T4, 15 GB).

---

## ✨ Highlights

| Contribution | What it is |
|---|---|
| **Color-Spectral Prior (CSP) gate** | Greenness (ExG), Brightness, iron-oxide Redness, and an RGB Vegetation Index computed on the fly and fused as a *learned spatial-attention prior* that steers the network toward physically-plausible mining surfaces and away from bare-rock/shadow false alarms. Explainable. |
| **Recall-weighted boundary stream** | An edge head with deep supervision + a compound loss (Dice + Tversky with β > α + boundary term + edge BCE) purpose-built for rare, small, fragmented structure classes. |
| **Lightweight backbone** | A ≤ ~6 M-parameter MobileNetV3-Small / EfficientNet-lite0 encoder with explicit params / GFLOPs / T4-latency reporting for edge & free-tier deployment. |
| **ASM-vs-LSM head** | An artisanal/small-scale (illegal-prone) vs large-scale (regulated) output of direct policy value. |

Every module is **independently ablatable**.

---

## 📦 Dataset — LAMES

**LAMES** — *Large-scale And Mining-site sEgmentation dataset*
(Hugging Face: [`maduschek/LAMES`](https://huggingface.co/datasets/maduschek/LAMES)).

```python
from datasets import load_dataset
ds = load_dataset("maduschek/LAMES")   # image (RGB) + mask (class-id) pairs
```

- Cloudless Sentinel-2 imagery over ~150 Chilean mining sites + Ghana ASM (galamsey),
  cut into 256×256 patches with masks pre-generated from GeoJSON.
- **No shapefiles, no Earth Engine, no rasterization.**
- License CC-BY-4.0.

### Class map (10-class mine-sector segmentation)

| id | class | id | class |
|----|-------|----|-------|
| 0 | other (background) | 5 | open pit |
| 1 | ASM site (artisanal/small-scale) | 6 | processing plant |
| 2 | LSM site (large-scale) | 7 | stockyard |
| 3 | heap leaching | 8 | tailings storage facility |
| 4 | mine facility | 9 | waste rock dump |

Supported tasks (set in config): `10class`, `3class` (`{other, ASM, LSM}`), `binary`
(`{background, mining}`).

---

## 🚀 Quickstart

### Google Colab (recommended)

Open [`notebooks/SPEARNet_LAMES_Colab.ipynb`](notebooks/SPEARNet_LAMES_Colab.ipynb)
in Colab, select a **T4 GPU** runtime, and *Run all*. The notebook clones this repo,
installs dependencies, streams a subset of LAMES, trains SPEAR-Net, and renders
metrics + CSP explainability overlays.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prakhar443/illegal_mining/blob/claude/busy-euler-kswww1/notebooks/SPEARNet_LAMES_Colab.ipynb)

### Local

```bash
git clone https://github.com/prakhar443/illegal_mining.git
cd illegal_mining
pip install -r requirements.txt
pip install -e .

# Smoke test (random tensors, no dataset needed) — verifies the full pipeline
python -m pytest tests/ -q

# Train (downloads / streams LAMES from Hugging Face)
python scripts/train.py --config configs/spearnet_10class.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --config configs/spearnet_10class.yaml \
    --checkpoint runs/spearnet_10class/best.pt

# Efficiency benchmark (params / GFLOPs / latency / peak VRAM)
python scripts/benchmark.py --config configs/spearnet_10class.yaml
```

---

## 🧱 Architecture

```
Input patch (RGB, 256x256x3)
        │
        ├──► Color-Spectral Prior  ── ExG, Brightness, Redness, RGB-VI
        │        1x1 + depthwise 3x3 -> sigmoid = attention map (per decoder scale)
        │
        ▼
Lightweight Encoder (MobileNetV3-S / EfficientNet-lite0, ImageNet-pretrained)  s2..s5
        │
        ▼   gated skips:  F = F_skip * (1 + alpha * attn)
Depthwise-Separable U-Net / FPN Decoder  (progressive upsample + fuse)
        │
        ├──► Segmentation head  (10-class | 3-class ASM/LSM/other | binary)
        └──► Edge/boundary head (deep supervision)

Compound loss = Dice + recall-weighted Tversky(β>α) + Boundary(BD) + BCE(edge)
```

See [`src/spearnet/models/spearnet.py`](src/spearnet/models/spearnet.py).

---

## 🔬 Color-Spectral Prior (CSP)

Computed on the fly from the RGB patch (no preprocessing):

| Prior | Formula | Captures |
|---|---|---|
| ExG | `2G − R − B` | residual vegetation / clearance contrast |
| Brightness | `(R+G+B)/3` | exposed ground, structures, processing plants |
| Redness | `(R−G)/(R+G)` | iron-oxide / reddish tailings & waste rock |
| RGB-VI | `(G²−R·B)/(G²+R·B)` | vegetation context |

Two integration modes (ablation): **`concat`** (priors appended as input channels — the
fast starter) and the stronger **`gate`** (priors → tiny conv → sigmoid attention map that
modulates each decoder skip). See [`src/spearnet/models/csp.py`](src/spearnet/models/csp.py).

---

## 🧪 Experiments

**Baselines** (all Colab-runnable, via `segmentation-models-pytorch`): plain U-Net,
Attention U-Net, DeepLabV3+ (MobileNet), U-Net++. Selectable with `model.name`.

**Ablations** (toggles in config):
1. Backbone + plain decoder, CrossEntropy (no prior, no recall loss).
2. + recall-weighted compound loss.
3. + CSP `concat`.
4. + CSP `gate` + edge head = **full SPEAR-Net**.
5. (optional) CSP-RGB vs spectral-PISP on a multiband subset.
6. Task variants: 10-class / 3-class ASM-LSM / binary.

**Metrics**: mIoU, per-class IoU (esp. rare classes), F1, precision, recall, plus
**area-stratified recall** (small / medium / large components), and params / GFLOPs /
T4 latency / peak VRAM.

---

## 🗂️ Repository layout

```
.
├── configs/                  # YAML experiment configs (one per task/ablation)
├── notebooks/
│   └── SPEARNet_LAMES_Colab.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── src/spearnet/
│   ├── config.py             # dataclass config + YAML loader
│   ├── data/                 # LAMES dataset, CSP priors, transforms
│   ├── models/               # SPEAR-Net, CSP gate, encoder, decoder, heads, baselines
│   ├── losses/               # Dice + Tversky + Boundary + edge BCE compound loss
│   ├── metrics/              # mIoU, per-class IoU, area-stratified recall
│   ├── engine/               # trainer + evaluator
│   └── utils/                # seeding, efficiency, class-remap, visualization
└── tests/                    # smoke tests (run with random tensors, no GPU/dataset)
```

---

## 📜 Citation

If you use this code, please cite the LAMES dataset and (once published) the SPEAR-Net paper.

```bibtex
@misc{spearnet,
  title  = {SPEAR-Net: A Lightweight, Color-Prior-Guided, Recall-Optimized Network
            for Fine-Grained Segmentation of Mining Structures},
  year   = {2026}
}
```

LAMES dataset: Mineral Resources Engineering, RWTH Aachen + Chair of Data Science in
Earth Observation, TU Munich; incorporates Maus et al. global mine-polygon annotations.

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The LAMES dataset is
CC-BY-4.0.
