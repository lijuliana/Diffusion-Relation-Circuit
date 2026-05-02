"""Phase A2 figure: emergence of spatial-relation alignment for ALL 36 heads
across 11 training checkpoints.

Two panels:
 (a) heatmap: 36 heads (rows, sorted by final-checkpoint |projection|) ×
     11 epochs (cols), color = |cos|.
 (b) curves for the top-6 heads + 5 random non-spatial controls, x=epoch,
     y=|projection|.

Source: results/emergence_all_heads.csv.
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
import matplotlib.patches as mpatches


df = pd.read_csv(join(PROJECT_ROOT, "results",
                      "emergence_all_heads.csv"))
df["label"] = df.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)
epochs = sorted(df["epoch"].unique())
labels = [f"L{l}H{h}" for l in range(6) for h in range(6)]

# Order rows by final-checkpoint abs_projection (descending)
final = df[df["epoch"] == max(epochs)].set_index("label")
ordered_labels = final.sort_values("abs_projection", ascending=False).index.tolist()

# Build (n_heads, n_epochs) matrix of |cos|
mat_cos = np.zeros((len(ordered_labels), len(epochs)))
mat_proj = np.zeros((len(ordered_labels), len(epochs)))
for i, lab in enumerate(ordered_labels):
    for j, e in enumerate(epochs):
        sub = df[(df["label"] == lab) & (df["epoch"] == e)]
        if len(sub):
            mat_cos[i, j] = float(sub.iloc[0]["abs_cosine"])
            mat_proj[i, j] = float(sub.iloc[0]["abs_projection"])

# ---------------- Figure ----------------
fig = plt.figure(figsize=(13.5, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.25)

ax0 = fig.add_subplot(gs[0, 0])
# Use |p| (the discriminative metric) on a cube-root scale: keeps the
# purple→yellow gradient but lifts low values out of pitch-black so the
# heatmap shows structure across the whole training range.
mat_cbrt = np.cbrt(mat_proj)
vmax = float(mat_cbrt.max())
# Use plasma (purple→pink→yellow); vmin offset below 0 so the lowest data
# points map to visible purple rather than black.
im = ax0.imshow(mat_cbrt, aspect="auto", cmap="plasma",
                vmin=-0.3, vmax=vmax)
ax0.set_xticks(range(len(epochs)))
ax0.set_xticklabels([str(e) for e in epochs], rotation=0)
ax0.set_yticks(range(len(ordered_labels)))
ax0.set_yticklabels(ordered_labels, fontsize=8.5)
ax0.set_xlabel("Training epoch")
ax0.set_ylabel("Head (sorted by final $|p|$ desc.)")
ax0.set_title(r"(a) Per-head emergence: projection magnitude $|p|$",
              fontsize=12, pad=8)
cbar = fig.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)
real_ticks = [0.1, 1, 5, 15, 30, 50]
cbar.set_ticks([np.cbrt(t) for t in real_ticks])
cbar.set_ticklabels([f"{t:g}" for t in real_ticks])
cbar.set_label(r"$|p|$", fontsize=11)
# Highlight L0H0/L1H2 rows
for highlight in ["L0H0", "L1H2"]:
    if highlight in ordered_labels:
        r = ordered_labels.index(highlight)
        rect = mpatches.Rectangle((-0.5, r - 0.5), len(epochs), 1,
                                  fill=False, edgecolor="lime", linewidth=2.0)
        ax0.add_patch(rect)

# ----- Panel (b) curves -----
ax1 = fig.add_subplot(gs[0, 1])
top_heads = ["L0H0", "L1H2", "L0H4", "L0H5", "L1H0", "L1H3"]
for lab in top_heads:
    if lab not in ordered_labels: continue
    i = ordered_labels.index(lab)
    is_top = lab in ("L0H0", "L1H2")
    ax1.plot(epochs, mat_proj[i],
             marker="o" if is_top else "s",
             lw=2.4 if is_top else 1.4,
             color={"L0H0": "#cb2027", "L1H2": "#1d4e89"}.get(lab, None),
             alpha=1.0 if is_top else 0.6,
             label=lab,
             markersize=8 if is_top else 5)

# Draw 4 random non-spatial controls as gray
np.random.seed(42)
controls = [lab for lab in ordered_labels if lab not in top_heads][-12:]
controls = list(np.random.choice(controls, size=4, replace=False))
for lab in controls:
    i = ordered_labels.index(lab)
    ax1.plot(epochs, mat_proj[i], color="#888", lw=0.9, alpha=0.55)

# Phase-transition shading
ax1.axvspan(600, 800, color="#fcdcdc", alpha=0.6,
            label="phase transition", zorder=0)

ax1.set_xlabel("Training epoch")
ax1.set_ylabel(r"$|$projection$|$ to spatial-relation factor")
ax1.set_xscale("log")
ax1.set_xticks([100, 250, 500, 800, 1000, 2000, 4000])
ax1.set_xticklabels(["100", "250", "500", "800", "1k", "2k", "4k"])
ax1.set_title("(b) Phase-transition emergence at epoch 600–800",
              fontsize=12, pad=8)
ax1.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95, ncol=2)

fig.suptitle("Spatial-relation alignment across training: $L_0H_0$ and "
             "$L_1H_2$ emerge sharply",
             fontsize=14, y=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(join(OUT_DIR, f"emergence.{ext}"), bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"[fig] emergence.* -> {OUT_DIR}")
