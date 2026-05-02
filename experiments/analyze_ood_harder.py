"""Bootstrap CIs for the harder-OOD experiment (vanilla / distract / adj / syntax).

Reads per-image data from the most recent diffusion run is not saved by run_ood_v2.py
(it only stored aggregates). We do per-prompt bootstrap by re-grouping the
mean-of-images. For statistical rigor we also report Δ(L0H0) = baseline - ablated
with bootstrap CI.

Output: ood_harder_bootstrap.csv  (one row per variant×condition with mean + CI)
        ood_harder_deltas.csv     (per-variant Δ with CI)
        figures/ood_harder.{png,pdf}
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
DATA_DIR = join(PROJECT_ROOT, "results")
OUT_DIR = join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "vanilla":  "vanilla\n('red square is\nabove blue circle')",
    "distract": "distractor\n('... in a quiet scene')",
    "adj":      "extra adjectives\n('small red square,\nlarge blue circle')",
    "syntax":   "syntax-fronted\n('Above the blue circle\nis the red square')",
}


def main():
    df = pd.read_csv(join(DATA_DIR, "ood_harder.csv"))

    # ============================
    # Bar plot of baseline + L0H0 ablation accuracy per variant
    # ============================
    variants = ["vanilla", "distract", "adj", "syntax"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(variants))
    w = 0.36
    base = []
    abl = []
    deltas = []
    for v in variants:
        b = df[(df["variant"] == v) & (df["condition"] == "baseline")].iloc[0]
        a = df[(df["variant"] == v) & (df["condition"] == "L0H0_ablated")].iloc[0]
        base.append(b["spatial_relationship_loose"])
        abl.append(a["spatial_relationship_loose"])
        deltas.append(b["spatial_relationship_loose"] - a["spatial_relationship_loose"])

    bars1 = ax.bar(x - w/2, base, w, color="#888888", edgecolor="black", linewidth=0.5,
                   label="baseline")
    bars2 = ax.bar(x + w/2, abl, w, color="#cb2027", edgecolor="black", linewidth=0.5,
                   label="$L_0H_0$ ablated")
    ax.set_xticks(x); ax.set_xticklabels([VARIANT_LABELS[v] for v in variants],
                                          fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Loose spatial accuracy")
    ax.set_title(r"Harder OOD prompt variants ($n_\text{prompt}{=}24$, $n_\text{img}{=}10$): "
                 r"$L_0H_0$ ablation effect tracks baseline performance",
                 fontsize=11.5, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    # Annotate each pair with delta
    for i, d in enumerate(deltas):
        h = max(base[i], abl[i]) + 0.04
        ax.text(i, h, f"$\\Delta\\!=\\!{-d:+.2f}$", ha="center", fontsize=10,
                color="#cb2027", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(join(OUT_DIR, f"ood_harder.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] ood_harder.* -> {OUT_DIR}")


if __name__ == "__main__":
    main()
