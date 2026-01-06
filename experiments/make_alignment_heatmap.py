"""Replicate the original paper's 'all_heads_align_score_synopsis' figure
with prettier formatting. Three side-by-side heatmaps: |cos|, projection,
energy, all 6x6 (layer x head). L0H0 and L1H2 highlighted.

Source: experiments/verify_top_heads_align.csv (real alignment data at
final checkpoint).
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR     = join(PROJECT_ROOT, "paper_original_completion", "figures")
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


df = pd.read_csv(join(PROJECT_ROOT, "experiments", "verify_top_heads_align.csv"))
df["abs_projection"] = df["projection"].abs()
n_layers = int(df["layer"].max()) + 1
n_heads  = int(df["head"].max()) + 1

def make_grid(col):
    g = np.zeros((n_layers, n_heads))
    for _, r in df.iterrows():
        g[int(r["layer"]), int(r["head"])] = r[col]
    return g

cos_grid = make_grid("abs_cosine")
proj_grid = make_grid("abs_projection")
energy_grid = make_grid("energy")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
metrics = [
    (cos_grid, r"$|\cos|$", "coolwarm", 0, 1, ".2f"),
    (proj_grid, r"projection magnitude", "coolwarm", 0, proj_grid.max() * 1.05, ".1f"),
    (energy_grid, r"energy", "Reds",      0, energy_grid.max() * 1.05, ".0f"),
]
for ax, (grid, title, cmap, vmin, vmax, fmt) in zip(axes, metrics):
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for li in range(n_layers):
        for hi in range(n_heads):
            v = grid[li, hi]
            txt_color = "white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else "black"
            ax.text(hi, li, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txt_color)
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=13, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # Highlight L0H0 and L1H2
    for layer, head in [(0, 0), (1, 2)]:
        rect = mpatches.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                  fill=False, edgecolor="lime", linewidth=3)
        ax.add_patch(rect)

fig.suptitle("Cross-attention head alignment to spatial-relation factor (final checkpoint)\n"
             "$L_0H_0$ and $L_1H_2$ highlighted as the top-2 by projection magnitude",
             fontsize=14, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(join(OUT_DIR, f"alignment_heatmap.{ext}"), bbox_inches="tight", dpi=300)
print(f"[fig] alignment_heatmap.* -> {OUT_DIR}")
