"""Standalone heatmap: layer-pair interaction I(L | layer 0) on each metric.

Output: layer_pair_heatmap.pdf/png.  Used as panel (b) of the circuit-diagram
figure (panel (a) is now an inline TikZ schematic).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PAPER_DIR, "figures")
DATA_DIR = os.path.join(PAPER_DIR, "results")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "layer_pair_interactions.csv")).sort_values("layer")
    layers = df["layer"].values
    grid = np.zeros((len(layers), 3))
    for i in range(len(layers)):
        grid[i, 0] = df.iloc[i]["interaction_spatial_relationship_loose"] * 100
        grid[i, 1] = df.iloc[i]["interaction_color"] * 100
        grid[i, 2] = df.iloc[i]["interaction_shape"] * 100

    vmax = float(np.abs(grid).max())
    fig, ax = plt.subplots(figsize=(4.0, 4.5))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["spatial", "colour", "shape"], fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{int(L)}" for L in layers], fontsize=10)

    for i in range(len(layers)):
        for j in range(3):
            v = grid[i, j]
            color = "white" if abs(v) > vmax * 0.5 else "#222"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    rect = mpatches.Rectangle((-0.5, 0.5), 3, 1,
                               fill=False, edgecolor="#2e7d32", linewidth=2.5)
    ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("interaction (pp)", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"layer_pair_heatmap.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved layer_pair_heatmap.png/.pdf to {OUT_DIR}")


if __name__ == "__main__":
    main()
