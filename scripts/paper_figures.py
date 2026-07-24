#!/usr/bin/env python
"""Regenerate the paper's data figures from the verified TEST-split numbers.

These figures are deterministic functions of the numbers already reported in the
manuscript tables (Table 2/3/4/6 and the gate confusion matrix), so they can be
rebuilt exactly without a GPU, a checkpoint or the dataset. Outputs vector PDF +
300-dpi PNG into --out (default: figures).

    python scripts/paper_figures.py --out figures

Produces: fig_area_recall, fig_efficiency, fig_ablation, fig_latency, fig_confusion.
"""
from __future__ import annotations
import argparse, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7")

# name, mIoU, fg, R_art, R_ind, small, med, large, params_M, gflops, GPUms, CPUms, family
ROWS = [
 ("A1 Backbone+CE",      0.584,0.422,0.573,0.586,0.366,0.644,0.838,1.06,0.53,None,None,"spear"),
 ("A2 +recall loss",     0.598,0.442,0.605,0.627,0.463,0.695,0.838,1.06,0.53,None,None,"spear"),
 ("A3 +PISP concat",     0.590,0.432,0.611,0.672,0.610,0.780,0.784,1.06,0.53,None,None,"spear"),
 ("A4 SPEAR-Net (gate)", 0.572,0.409,0.748,0.662,0.537,0.729,0.838,1.06,0.60,7.9,42.7,"spear"),
 ("A5 RGB prior",        0.620,0.471,0.681,0.633,0.439,0.678,0.838,1.06,0.53,None,None,"spear"),
 ("U-Net",               0.587,0.432,0.660,0.718,0.415,0.729,0.865,24.45,16.06,7.3,240.3,"base"),
 ("DeepLabV3+",          0.562,0.395,0.662,0.644,0.488,0.729,0.865,22.45,16.14,6.5,203.8,"base"),
 ("U-Net++",             0.604,0.452,0.647,0.678,0.512,0.712,0.865,26.09,37.27,11.6,518.1,"base"),
]
N_BINS = {"small": 41, "medium": 59, "large": 37}   # ground-truth component counts
CM = np.array([[165081346, 4345110, 8922339],       # gate confusion (rows=GT, cols=pred)
               [1517953, 4687826, 61924],
               [4264938, 1855928, 11965484]], float)


def _style():
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "figure.dpi": 120,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def area_recall(out):
    labels = [r[0] for r in ROWS]
    bins = ["small", "medium", "large"]
    bcol = {"small": OK["blue"], "medium": OK["orange"], "large": OK["green"]}
    idx = {"small": 5, "medium": 6, "large": 7}
    x = np.arange(len(ROWS)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, b in enumerate(bins):
        p = np.array([r[idx[b]] for r in ROWS]); lo = []; hi = []
        for pi in p:
            a, c = _wilson(pi, N_BINS[b]); lo.append(pi - a); hi.append(c - pi)
        ax.bar(x + (i - 1) * w, p, w, label=f"{b.capitalize()} (n={N_BINS[b]})",
               color=bcol[b], yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8, alpha=0.7))
    ax.axvline(4.5, color="0.6", ls="--", lw=0.8)
    ax.text(2.0, 0.96, "SPEAR-Net family (1.06 M)", ha="center", fontsize=8, color="0.3")
    ax.text(6.5, 0.96, "Baselines (22-26 M)", ha="center", fontsize=8, color="0.3")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Component recall"); ax.set_ylim(0, 1.02)
    ax.set_title("Area-stratified recall on the test split (95% Wilson CI)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    _save(fig, out, "fig_area_recall")


def efficiency(out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in ROWS:
        ax.scatter(r[8], r[1], s=70, marker=("o" if r[12] == "spear" else "s"),
                   color=(OK["blue"] if r[12] == "spear" else OK["verm"]),
                   edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(r[0].split(" ", 1)[-1], (r[8], r[1]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("mIoU (test)"); ax.set_ylim(0.54, 0.63)
    ax.scatter([], [], marker="o", color=OK["blue"], edgecolor="k", label="SPEAR-Net family")
    ax.scatter([], [], marker="s", color=OK["verm"], edgecolor="k", label="Baselines")
    ax.legend(frameon=False, loc="lower right"); ax.set_title("Accuracy vs. model size")
    _save(fig, out, "fig_efficiency")


def ablation(out):
    abl = [r for r in ROWS if r[0][:2] in ("A1", "A2", "A3", "A4", "A5")]
    metrics = [("mIoU", 1, OK["blue"]), ("Artisanal recall", 3, OK["orange"]),
               ("Small-object recall", 5, OK["green"])]
    x = np.arange(len(abl)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (nm, j, c) in enumerate(metrics):
        ax.bar(x + (i - 1) * w, [r[j] for r in abl], w, label=nm, color=c)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in abl], rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 0.9)
    ax.set_title("Component analysis (ablation, test split)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    _save(fig, out, "fig_ablation")


def latency(out):
    M = [r for r in ROWS if r[10] is not None]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, j, xl, logx, ttl in [(a1, 10, "GPU latency (ms/chip)", False, "GPU (T4)"),
                                 (a2, 11, "CPU latency (ms/chip, log)", True, "CPU (single-thread)")]:
        for r in M:
            ax.scatter(r[j], r[1], s=80, marker=("o" if r[12] == "spear" else "s"),
                       color=(OK["blue"] if r[12] == "spear" else OK["verm"]),
                       edgecolor="k", lw=0.5, zorder=3)
            ax.annotate(r[0].split(" ", 1)[-1] if r[12] == "spear" else r[0],
                        (r[j], r[1]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylim(0.55, 0.62); ax.set_title(ttl, fontsize=10)
    a1.set_ylabel("mIoU (test)")
    a1.scatter([], [], marker="o", color=OK["blue"], edgecolor="k", label="SPEAR-Net")
    a1.scatter([], [], marker="s", color=OK["verm"], edgecolor="k", label="Baselines")
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Accuracy vs. inference latency", y=1.0, fontsize=11); fig.tight_layout()
    _save(fig, out, "fig_latency")


def confusion(out):
    cls = ["Background", "Artisanal", "Industrial"]
    rown = CM / CM.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(cls); ax.set_yticklabels(cls)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("SPEAR-Net (gate): confusion matrix, test split\n"
                 "(row-normalized; cell = share of true-class pixels)")
    human = lambda v: f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    for i in range(3):
        for j in range(3):
            f = rown[i, j]
            ax.text(j, i - 0.10, f"{f*100:.1f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=("white" if f > 0.5 else "black"))
            ax.text(j, i + 0.18, human(CM[i, j]), ha="center", va="center",
                    fontsize=8, color=("white" if f > 0.5 else "0.35"))
    for (i, j) in [(1, 0), (1, 2)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#D55E00", lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Row-normalized fraction")
    fig.tight_layout()
    _save(fig, out, "fig_confusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    _style(); os.makedirs(args.out, exist_ok=True)
    area_recall(args.out); efficiency(args.out); ablation(args.out)
    latency(args.out); confusion(args.out)
    print(f"wrote fig_area_recall, fig_efficiency, fig_ablation, fig_latency, fig_confusion -> {args.out}")


if __name__ == "__main__":
    main()
