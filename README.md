# SPEAR-Net (v4)

**S**pectral-**P**rior **E**dge-**A**ware **R**ecall-optimized **Net**work — a lightweight,
**spectral-prior-guided**, recall-optimized network for detecting **A**rtisanal &
**S**mall-scale (illegal-prone) **M**ining in **global Sentinel-2 imagery**.

> Trains end-to-end on free Google Colab (T4, 15 GB) from a **35.6 MB verified annotation
> set** plus an on-demand **subset** of 6-band Sentinel-2 fetched from Microsoft Planetary
> Computer — no 20 GB download, and a real, verified artisanal-vs-industrial label.

---

## ✨ Highlights

| Contribution | What it is |
|---|---|
| **PISP gate** (Physically-Informed Spectral Prior) | NDVI (vegetation loss), MNDWI (mining ponds), NDTI (turbidity), BSI (bare soil/tailings) computed on the fly from Sentinel-2 bands and fused as a *learned spatial-attention prior* that steers the network toward physically-plausible mining and suppresses bare-rock/water false alarms. Explainable. |
| **Recall-weighted boundary stream** | An edge head with deep supervision + a compound loss (present-class Dice + Tversky β>α + Focal + boundary + edge BCE) with inverse-frequency class weights, built for small, fragmented artisanal sites — and engineered to **not collapse to background** under heavy imbalance. |
| **Lightweight backbone** | A ≤ ~6 M-parameter MobileNetV3-Small / EfficientNet-lite0 encoder (stem adapted 3→6 bands), with explicit params / GFLOPs / T4-latency reporting for edge & free-tier deployment. |
| **Artisanal-vs-industrial head** | A small-scale (illegal-prone) vs industrial (regulated) output of direct policy value, on **verified** labels. |
| **Cross-region generalization** | A geographic leave-one-region-out protocol (train on some continents, test on held-out ones) enabled by the global dataset. |

Every module is **independently ablatable**.

---

## 📦 Dataset — global mine-segmentation

`SimonJasansky/mine-segmentation` (Zenodo [10.5281/zenodo.14195737](https://doi.org/10.5281/zenodo.14195737),
Maastricht University). 1,210 mining sites worldwide (1,514 annotated tiles; confirm with `scripts/diagnostics.py`); masks from Maus et al. (2022) + Tang
et al. (2023), **manually re-validated** (accuracy 99.78, precision 99.22, recall 95.71).
License CC-BY-SA-4.0.

- **Annotations**: a 35.6 MB GeoPackage — per-site Sentinel-2 STAC id (`s2_tile_id`), mask
  polygons, train/val/test `split`, and metadata: `minetype1` (surface/placer/underground/
  brine) and **`minetype2` = Artisanal vs Industrial** (the headline scale label;
  Artisanal ≈ 329 tiles, Industrial ≈ 1,185).
- **Imagery**: fetched on demand from **Microsoft Planetary Computer** (Sentinel-2 L2A,
  free, no credentials) — 6 bands **B, G, R, NIR, SWIR1, SWIR2** — and chipped 2048→256.
  You fetch only a subset → a few GB, cached once to Drive.

### Fetch & store (one-time)

```python
from spearnet.data.fetch import fetch_dataset
# metadata only — prints split / minetype / artisanal-vs-industrial counts
fetch_dataset(fetch_imagery=False)
# fetch a subset of 6-band S2 -> uint16 GeoTIFF chips + manifest.csv (scale + region)
fetch_dataset(out_dir="chips", fetch_imagery=True, scale_filter="artisanal", subset_per_split=None)
fetch_dataset(out_dir="chips", fetch_imagery=True, scale_filter="industrial", subset_per_split=80)
```

The pipeline reads chips via `chips/manifest.csv`, which carries each chip's verified
`scale` and `region`. Tasks (set `data.task`):

| task | classes |
|------|---------|
| `binary` | `{background, mining}` — robust detector |
| `scale3` | `{background, artisanal, industrial}` — the headline (mining pixels relabelled by the chip's verified scale) |

> Legacy v3 LAMES sources (`source: local` Drive zips, or `source: hf`) are still supported
> for the RGB-only ablation; see git history / configs `spearnet_*.yaml`.

---

## 🚀 Quickstart

### Google Colab (recommended)

Open [`notebooks/SPEARNet_LAMES_Colab.ipynb`](notebooks/SPEARNet_LAMES_Colab.ipynb)
in Colab, select a **T4 GPU** runtime, and *Run all*. The notebook clones this repo,
installs dependencies, streams a subset of LAMES, trains SPEAR-Net, and renders
metrics + CSP explainability overlays.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prakhar443/Illegal_Mining/blob/spearnet-colab/notebooks/SPEARNet_LAMES_Colab.ipynb)

> **Working in a team?** See [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md) — interns fork the
> repo (your original stays protected, changes only via PR review) and each keeps the
> dataset on their **own** Google Drive (fetch once, share a link, everyone restores).

### Local

```bash
git clone https://github.com/prakhar443/Illegal_Mining.git
cd Illegal_Mining
pip install -r requirements.txt
pip install -e .

# Smoke test (random tensors, no dataset needed) — verifies the full pipeline
python -m pytest tests/ -q

# 1) Inspect annotations (35.6 MB) — prints artisanal/industrial counts
python -c "from spearnet.data.fetch import fetch_dataset; fetch_dataset(fetch_imagery=False)"

# 2) Fetch a 6-band Sentinel-2 subset -> chips + manifest (cached once)
python scripts/fetch_data.py --scale artisanal --subset 0   --out chips   # all artisanal
python scripts/fetch_data.py --scale industrial --subset 80 --out chips   # capped industrial

# 3) Train (binary detector or artisanal/industrial)
python scripts/train.py --config configs/v4_scale3.yaml

# Evaluate / benchmark
python scripts/evaluate.py --config configs/v4_scale3.yaml --checkpoint runs/v4_scale3/best.pt
python scripts/benchmark.py --config configs/v4_scale3.yaml
```

---

## 🧱 Architecture

```
Input patch (S2: B,G,R,NIR,SWIR1,SWIR2 — 256x256x6)
        │
        ├──► PISP Prior  ── NDVI, MNDWI, NDTI, BSI
        │        1x1 + depthwise 3x3 -> sigmoid = attention map (per decoder scale)
        │
        ▼
Lightweight Encoder (MobileNetV3-S / EfficientNet-lite0, stem 3→6 bands)  s2..s5
        │
        ▼   gated skips:  F = F_skip * (1 + alpha * attn)
Depthwise-Separable U-Net / FPN Decoder  (progressive upsample + fuse)
        │
        ├──► Segmentation head  (binary mining | 3-class artisanal/industrial/bg)
        └──► Edge/boundary head (deep supervision)

Compound loss = present-class Dice + recall-weighted Tversky(β>α) + Focal
                + Boundary(BD) + BCE(edge)   [+ inverse-frequency class weights]
```

See [`src/spearnet/models/spearnet.py`](src/spearnet/models/spearnet.py).

---

## 🔬 Physically-Informed Spectral Prior (PISP)

Computed on the fly from the Sentinel-2 bands (`prior_type: pisp`):

| Prior | Formula | Captures |
|---|---|---|
| NDVI | `(NIR−R)/(NIR+R)` | forest/scrub → cleared-land loss |
| MNDWI | `(G−SWIR1)/(G+SWIR1)` | diagnostic mining ponds / lagoons |
| NDTI | `(R−G)/(R+G)` | turbid / muddy mining water |
| BSI | `((SWIR1+R)−(NIR+B))/((SWIR1+R)+(NIR+B))` | bare soil / tailings |

Two integration modes: **`concat`** (priors as extra input channels) and the recall-oriented
**`gate`** (priors → tiny conv → sigmoid attention modulating each decoder skip; comparable
mIoU to `concat` but higher recall — see the paper's ablation). Set `prior_type: csp`
(+ `bands: 3`) for the **RGB-only ablation** that isolates the SWIR/NIR
contribution. See [`src/spearnet/models/csp.py`](src/spearnet/models/csp.py) and
[`src/spearnet/data/priors.py`](src/spearnet/data/priors.py).

---

## 🧪 Experiments

**Baselines** (Colab-runnable, via `segmentation-models-pytorch`): plain U-Net, Attention
U-Net, DeepLabV3+, U-Net++. Selectable with `model.name`.

**Ablations**:
1. Backbone + plain decoder, CrossEntropy (no prior, no recall loss).
2. + recall-weighted compound loss.
3. + PISP `concat`.
4. + PISP `gate` + edge head = **full SPEAR-Net**.
5. RGB-only prior (`prior_type: csp`, 3 bands) vs full spectral PISP (6 bands).
6. Tasks: `binary` / `scale3` (artisanal-vs-industrial).

**Generalization (headline)**: geographic **leave-one-region-out** — set
`data.holdout_regions` (e.g. `[Asia, Oceania]`); training excludes them and an `ood` split
is exposed for out-of-region testing. Report in-region vs out-of-region mIoU drop.

**Metrics**: mIoU, per-class IoU, F1, precision, recall, **area-stratified recall**
(small/medium/large), and params / GFLOPs / T4 latency / peak VRAM.

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

If you use this code, please cite the dataset and (once published) the SPEAR-Net paper.

```bibtex
@misc{spearnet,
  title  = {Distinguishing Artisanal from Industrial Mining in Sentinel-2 Imagery:
            A Spectral-Prior, Recall-Oriented Segmentation Approach},
  year   = {2026}
}
```

Dataset: Jasansky, Maus, Popa \& Wilbik (Maastricht University), *Global ML-ready dataset
for mining areas in satellite images*, Zenodo, 2024, DOI
[10.5281/zenodo.14195737](https://doi.org/10.5281/zenodo.14195737); masks derived from
Maus et al. (2022) and Tang et al. (2023).

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The mining annotation
dataset is **CC-BY-SA-4.0** (copyleft — derivatives must share alike).
