"""Two figures:

  layer_ablation: per-layer (entire layer ablated = all 6 heads in that
    layer zeroed) accuracy. Shows that only Layer 0 carries spatial signal.

  per_relation_breakdown: per-relation effect of L0H0/L1H2/control ablation.
    Shows L0H0 affects diagonals strongly, cardinals less; L1H2 alone
    doesn't have measurable effect (consistent with redundancy).
"""
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
OUT_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt


def fig_layer_ablation():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments", "layer_ablation_results.csv"))
    base = df[df["label"] == "baseline"].iloc[0]
    base_loose = base["spatial_relationship_loose"]
    layers = []
    losses = []
    color_drops = []; shape_drops = []
    for _, row in df[df["label"] != "baseline"].iterrows():
        l = int(row["label"].split("_")[1])
        layers.append(l)
        losses.append((base_loose - row["spatial_relationship_loose"]) * 100)
        color_drops.append((base["color"] - row["color"]) * 100)
        shape_drops.append((base["shape"] - row["shape"]) * 100)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(layers))
    w = 0.27
    ax.bar(x - w, losses, w, color="#cb2027", label="loose spatial relation",
           edgecolor="black", linewidth=0.4)
    ax.bar(x,     color_drops, w, color="#1d4e89", label="color", edgecolor="black",
           linewidth=0.4)
    ax.bar(x + w, shape_drops, w, color="#888888", label="shape", edgecolor="black",
           linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"Layer {l}\n(6 heads)" for l in layers])
    ax.set_ylabel("Accuracy drop (pp)")
    ax.set_title("Per-layer cross-attention ablation: only Layer 0 carries the spatial signal\n"
                 f"(baseline loose = {base_loose:.0%})", pad=8, fontsize=11.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"layer_ablation.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig] layer_ablation.* ->", OUT_DIR)


def fig_per_relation():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments",
                                  "per_relation_l0h0_breakdown.csv"))
    rels = ["above", "below", "left", "right",
            "upper_left", "upper_right", "lower_left", "lower_right"]
    conds = ["ablate_L0H0", "ablate_L1H2", "ablate_L0H2_control"]
    cond_label = {"ablate_L0H0": r"$L_0H_0$ ablated",
                  "ablate_L1H2": r"$L_1H_2$ ablated",
                  "ablate_L0H2_control": "$L_0H_2$ (matched control)"}
    cond_color = {"ablate_L0H0": "#cb2027",
                  "ablate_L1H2": "#1d4e89",
                  "ablate_L0H2_control": "#888888"}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(rels))
    w = 0.27
    for i, c in enumerate(conds):
        sub = df[df["condition"] == c].set_index("relation")
        deltas = [sub.loc[r, "delta"] * 100 if r in sub.index else 0 for r in rels]
        offset = (i - 1) * w
        ax.bar(x + offset, deltas, w, color=cond_color[c], label=cond_label[c],
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    # Visual separator between cardinal (first 4) and diagonal (last 4)
    ax.axvline(3.5, color="gray", lw=0.5, linestyle=":")
    ax.text(1.5, 47, "cardinal", ha="center", fontsize=10, style="italic", color="#444")
    ax.text(5.5, 47, "diagonal", ha="center", fontsize=10, style="italic", color="#444")

    ax.set_xticks(x); ax.set_xticklabels([r.replace("_", "\n") for r in rels], fontsize=9.5)
    ax.set_ylabel("Accuracy drop (pp; positive = damage)")
    ax.set_title("$L_0H_0$ ablation hits diagonals harder than cardinals; "
                 "$L_1H_2$ alone has no measurable effect",
                 pad=8, fontsize=11.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"per_relation_breakdown.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig] per_relation_breakdown.* ->", OUT_DIR)


if __name__ == "__main__":
    fig_layer_ablation()
    fig_per_relation()
