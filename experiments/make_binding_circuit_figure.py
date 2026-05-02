"""Two-panel figure that visualises the downstream binding circuit.

  (a) Layer-pair interaction heatmap: rows = layer X (1-5), columns =
      metric (spatial, color, shape); cell = I(layerX | layer0). Layer 2
      stands out as the strong super-additive on color/shape but mild on
      spatial. (Original "object-binding stage" hypothesis, rediscovered.)

  (b) Triple-ablation candidate ranking: bars showing color/shape
      super-additive interactions of each consensus or layer-2 head when
      ablated on top of the source pair {L0H0, L1H2}.
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
OUT_DIR = join(PROJECT_ROOT, "figures")
DATA_DIR = join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def panel_layer_pair(ax):
    df = pd.read_csv(join(DATA_DIR, "layer_pair_interactions.csv"))
    df = df.sort_values("layer")
    layers = df["layer"].values
    metrics = ["spatial", "color", "shape"]
    grid = np.zeros((len(layers), 3))
    for i, _ in enumerate(layers):
        grid[i, 0] = df.iloc[i]["interaction_spatial_relationship_loose"] * 100
        grid[i, 1] = df.iloc[i]["interaction_color"] * 100
        grid[i, 2] = df.iloc[i]["interaction_shape"] * 100

    vmax = float(np.abs(grid).max())
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["spatial\n(loose)", "color", "shape"])
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"Layer {L}" for L in layers])
    ax.set_xlabel("Metric"); ax.set_ylabel("Co-ablated layer X")
    for i in range(len(layers)):
        for j in range(3):
            v = grid[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    # highlight layer 2 row
    rect = mpatches.Rectangle((-0.5, 0.5), 3, 1,  # row 1 = layer 2 (since layers 1..5 zero-indexed)
                               fill=False, edgecolor="lime", linewidth=3)
    ax.add_patch(rect)
    ax.set_title("(a) $I(\\text{layer X} \\mid \\text{layer 0}) = \\Delta(\\text{both}) - \\Delta(\\text{layer 0}) - \\Delta(\\text{layer X})$,\n"
                 "in pp; positive = super-additive (shared circuit)",
                 fontsize=11, pad=10)
    plt.colorbar(im, ax=ax, label="interaction (pp)", fraction=0.05, pad=0.04)


def panel_triple(ax):
    df = pd.read_csv(join(DATA_DIR, "triple_ablation_interactions.csv"))
    df["binding_inter"] = (df["interaction_color"] + df["interaction_shape"]) / 2 * 100
    df = df.sort_values("binding_inter", ascending=False)
    cands = df["candidate"].tolist()
    color_inter = df["interaction_color"].values * 100
    shape_inter = df["interaction_shape"].values * 100
    spatial_inter = df["interaction_spatial_relationship_loose"].values * 100

    x = np.arange(len(cands))
    w = 0.27
    ax.bar(x - w, spatial_inter, w, color="#cb2027", label="spatial loose",
           edgecolor="black", linewidth=0.4)
    ax.bar(x,     color_inter,   w, color="#1d4e89", label="color",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w, shape_inter,   w, color="#888888", label="shape",
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cands, fontsize=10)
    ax.set_ylabel("Triple-ablation interaction (pp)")
    ax.set_title(r"(b) $I(h \mid \text{src}=\{L_0H_0, L_1H_2\})$ for triple ablation;"+
                 "\nseveral candidates super-additive on BINDING but only weakly on SPATIAL",
                 fontsize=11, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1, 1.4]})
    panel_layer_pair(axes[0])
    panel_triple(axes[1])
    fig.suptitle("Downstream object-binding stage: Layer 2 (and L0H5) super-additive with the source pair on color/shape but not spatial",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(join(OUT_DIR, f"binding_circuit.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] binding_circuit.* -> {OUT_DIR}")


if __name__ == "__main__":
    main()
