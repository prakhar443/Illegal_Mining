#!/usr/bin/env python
"""Generate the paper's data figures from existing results (no re-training).

Produces (PNG @300dpi + vector PDF for the manuscript):
  fig_dataset_map    - world map of the 1,514 dataset sites, artisanal vs industrial
  fig_efficiency     - mIoU vs parameters (log-x): SPEAR-Net family vs baselines
  fig_ablation       - grouped bars: mIoU / artisanal IoU / industrial IoU per variant
  fig_area_recall    - area-stratified recall of SPEAR-Net (small/medium/large)

Colors are the Okabe-Ito colorblind-safe palette, validated for CVD separation and
contrast; entity assignment is fixed across figures (artisanal=vermillion,
industrial=blue). Identity is never color-alone: the scatter adds marker shapes and
direct labels, bars add a legend and axis labels.

Usage (Colab or local):
  python scripts/make_figures.py \
      --results /path/comparison_full_scale3.json \
      --gpkg mine_data/mining_area_data.gpkg \
      --area 0.441 0.796 0.922 --area-counts 68 49 51 \
      --out figures
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fixed, validated palette (Okabe-Ito subset) -------------------------------------
C_ARTISANAL = "#D55E00"   # vermillion  (entity color, all figures)
C_INDUSTRIAL = "#0072B2"  # blue        (entity color, all figures)
C_MIOU = "#009E73"        # bluish green (mIoU series / SPEAR-Net group)
C_BASELINE = "#E69F00"    # orange      (baseline group; relief via labels+shape)
C_LAND = "#E8E6E1"
C_GRID = "#B9B4A7"
INK = "#1F1D1A"

NE_LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_110m_land.geojson")


def _style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "axes.edgecolor": C_GRID, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": C_GRID, "grid.alpha": 0.35,
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False,
    })


def _save(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {name}.png/.pdf -> {out_dir}")


# --------------------------------------------------------------------------------------
def fig_dataset_map(gpkg: str, out_dir: str, scale_col: str = "minetype2"):
    """World map of site centroids, colored by verified scale label."""
    import geopandas as gpd

    tiles = gpd.read_file(gpkg, layer="tiles").to_crs(epsg=4326)
    cent = tiles.geometry.centroid
    scale = tiles[scale_col].str.lower().fillna("industrial")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.grid(False)

    # Land backdrop (Natural Earth 110m); fall back to a bare graticule offline.
    try:
        tmp = "/tmp/ne_110m_land.geojson"
        if not os.path.exists(tmp):
            urllib.request.urlretrieve(NE_LAND_URL, tmp)
        gpd.read_file(tmp).plot(ax=ax, color=C_LAND, edgecolor="none", zorder=0)
    except Exception as e:  # noqa: BLE001
        print(f"[fig] land backdrop unavailable ({e}); plotting graticule only")
        ax.grid(True)

    for label, color, marker in (("industrial", C_INDUSTRIAL, "o"),
                                 ("artisanal", C_ARTISANAL, "^")):
        sel = scale == label
        ax.scatter(cent[sel].x, cent[sel].y, s=9, c=color, marker=marker,
                   linewidths=0.3, edgecolors="white",
                   label=f"{label} (n={int(sel.sum())})", zorder=2)

    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.legend(loc="lower left", markerscale=1.6)
    _save(fig, out_dir, "fig_dataset_map")


# --------------------------------------------------------------------------------------
def _load_rows(results: str):
    rows = [r for r in json.load(open(results)) if "mIoU" in r]
    return sorted(rows, key=lambda r: r["name"])


def fig_efficiency(results: str, out_dir: str):
    """mIoU vs parameters (log-x). The paper's headline figure."""
    rows = _load_rows(results)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    # Per-point label placement (points cluster at shared x; avoid collisions).
    offs = {"A1": (7, -3, "left"), "A2": (7, -3, "left"), "A3": (7, 1, "left"),
            "A4": (7, -7, "left"), "A5": (7, -3, "left"),
            "B1": (-7, 2, "right"), "B2": (-7, -9, "right"), "B3": (7, -9, "left")}

    for r in rows:
        is_spear = r.get("model") == "spearnet"
        color = C_MIOU if is_spear else C_BASELINE
        marker = "o" if is_spear else "s"
        ax.scatter(r["params_M"], r["mIoU"], s=46, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=3)
        short = r["name"].split(" ")[0]           # A1..A5 / B1..B3
        dx, dy, ha = offs.get(short, (0, 6, "center"))
        ax.annotate(short, (r["params_M"], r["mIoU"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8, color=INK)

    ax.set_xscale("log")
    ax.margins(y=0.12)
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("mIoU")
    handles = [
        plt.Line2D([], [], color=C_MIOU, marker="o", ls="", markeredgecolor="white",
                   label="SPEAR-Net variants (A1–A5)"),
        plt.Line2D([], [], color=C_BASELINE, marker="s", ls="", markeredgecolor="white",
                   label="Baselines (B1–B3)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _save(fig, out_dir, "fig_efficiency")


def fig_ablation(results: str, out_dir: str):
    """Grouped bars over the SPEAR-Net variants: mIoU + the two mining-class IoUs.

    Background IoU (~0.90 for every variant) is omitted so the axis resolves the
    differences that matter; it is reported in the table instead.
    """
    rows = [r for r in _load_rows(results) if r.get("model") == "spearnet"]
    names = [r["name"].split(" ")[0] for r in rows]           # A1..A5
    series = [
        ("mIoU", C_MIOU, [r["mIoU"] for r in rows]),
        ("artisanal IoU", C_ARTISANAL, [r.get("IoU_artisanal") for r in rows]),
        ("industrial IoU", C_INDUSTRIAL, [r.get("IoU_industrial") for r in rows]),
    ]

    x = range(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for i, (label, color, vals) in enumerate(series):
        ax.bar([xi + (i - 1) * w for xi in x], vals, width=w, color=color,
               edgecolor="white", linewidth=1.0, label=label)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("IoU")
    ax.set_xlabel("SPEAR-Net variant")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    _save(fig, out_dir, "fig_ablation")


def fig_area_recall(area, counts, out_dir: str):
    """Component-level recall by ground-truth component area (single series)."""
    cats = ["small\n(<32² px)", "medium", "large\n(≥96² px)"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bars = ax.bar(cats, area, width=0.55, color=C_INDUSTRIAL,
                  edgecolor="white", linewidth=1.0)
    for b, v, n in zip(bars, area, counts):
        ax.annotate(f"{v:.2f}\n(n={n})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Component recall")
    _save(fig, out_dir, "fig_area_recall")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="comparison_full_*.json")
    ap.add_argument("--gpkg", default=None, help="annotation GeoPackage (dataset map)")
    ap.add_argument("--scale-col", default="minetype2")
    ap.add_argument("--area", type=float, nargs=3, default=[0.441, 0.796, 0.922],
                    help="area-stratified recall: small medium large")
    ap.add_argument("--area-counts", type=int, nargs=3, default=[68, 49, 51])
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    _style()
    if args.gpkg and os.path.exists(args.gpkg):
        fig_dataset_map(args.gpkg, args.out, args.scale_col)
    if args.results and os.path.exists(args.results):
        fig_efficiency(args.results, args.out)
        fig_ablation(args.results, args.out)
    fig_area_recall(args.area, args.area_counts, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
