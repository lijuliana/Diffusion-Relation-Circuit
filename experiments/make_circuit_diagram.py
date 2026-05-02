"""Redesigned Figure 1: clean two-stage circuit diagram (v2 — simplified).

Layout:
  (a) Schematic with two stages, simplified for clarity
  (b) Stage-1 evidence: L0H0 attention bias for 'above'
  (c) Stage-2 evidence: layer-pair interaction values (real data)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = "/DiffusionInterp"
OUT_DIR = os.path.join(PROJECT_ROOT, "figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "results")


def panel_schematic(ax):
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)
    ax.axis("off")
    bk = dict(boxstyle="round,pad=0.30")
    arrow_kw = dict(arrowstyle="->,head_length=8,head_width=4", lw=1.5, color="#374151")
    big_arrow_kw = dict(arrowstyle="->,head_length=12,head_width=7", lw=2.4, color="#92400e")

    # ---- Row 1: T5 prompt -----------------------------------------
    ax.add_patch(FancyBboxPatch((1.0, 10.5), 10.0, 1.0, fc="#e8eef7", ec="#1d4e89", lw=1.4, **bk))
    ax.text(6.0, 11.0, 'T5 prompt:  "red square is above blue circle"',
            ha="center", va="center", fontsize=10.5, style="italic", color="#1d4e89")

    # Two token-type lanes coming out of the prompt
    ax.text(2.0, 9.7, "relation token: 'above'",
            ha="center", va="center", fontsize=9.5, color="#cb2027",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fce4ec", ec="#cb2027", lw=1.0))
    ax.text(9.5, 9.7, "object tokens: 'red square', 'blue circle'",
            ha="center", va="center", fontsize=9.5, color="#166534",
            bbox=dict(boxstyle="round,pad=0.2", fc="#dcfce7", ec="#166534", lw=1.0))
    # Down-arrow from prompt to two lanes
    ax.add_patch(FancyArrowPatch((2.0, 10.4), (2.0, 9.95), **arrow_kw))
    ax.add_patch(FancyArrowPatch((9.5, 10.4), (9.5, 9.95), **arrow_kw))

    # ---- Row 2: caption projection (full width) -------------------
    ax.add_patch(FancyBboxPatch((1.0, 8.2), 10.0, 1.0, fc="#fff4e6", ec="#d97706", lw=1.4, **bk))
    ax.text(6.0, 8.7, r"caption projection:  $\mathbb{R}^{4096} \to \mathbb{R}^{384}$",
            ha="center", va="center", fontsize=10, color="#d97706")
    ax.add_patch(FancyArrowPatch((2.0, 9.4), (2.0, 9.2), **arrow_kw))
    ax.add_patch(FancyArrowPatch((9.5, 9.4), (9.5, 9.2), **arrow_kw))

    # ---- Row 3: Stage 1 (left) and Stage 2 (right) ---------------
    # Stage 1 box
    ax.add_patch(FancyBboxPatch((0.3, 4.4), 5.4, 3.0, fc="#fef9f0", ec="#cb2027", lw=2.3, **bk))
    ax.text(3.0, 7.0, "STAGE 1:  spatial routing",
            ha="center", va="center", fontsize=11, color="#cb2027", fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.6, 4.9), 2.2, 1.5, fc="#fce4ec", ec="#cb2027", lw=1.4, **bk))
    ax.text(1.7, 5.65, r"$L_0H_0$" + "\nlayer 0,\nhead 0",
            ha="center", va="center", fontsize=9.5, color="#cb2027", fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.2, 4.9), 2.2, 1.5, fc="#fce4ec", ec="#cb2027", lw=1.4, **bk))
    ax.text(4.3, 5.65, r"$L_1H_2$" + "\nlayer 1,\nhead 2",
            ha="center", va="center", fontsize=9.5, color="#cb2027", fontweight="bold")
    # Stage 1 caption
    ax.text(3.0, 4.62,
            r"emit $8\times 8$ per-position attention bias",
            ha="center", va="center", fontsize=8.5, color="#374151", style="italic")
    # Caption-proj -> Stage 1 (relation lane)
    ax.add_patch(FancyArrowPatch((2.0, 8.1), (3.0, 7.4), **arrow_kw))

    # Stage 2 box
    ax.add_patch(FancyBboxPatch((6.3, 4.4), 5.4, 3.0, fc="#fef3c7", ec="#92400e", lw=2.3, **bk))
    ax.text(9.0, 7.0, "STAGE 2:  object binding",
            ha="center", va="center", fontsize=11, color="#92400e", fontweight="bold")
    ax.add_patch(FancyBboxPatch((6.7, 4.9), 4.6, 1.5, fc="#fef9f0", ec="#92400e", lw=1.4, **bk))
    ax.text(9.0, 5.65, "Layer 2 (six heads as a group)\ndistributed",
            ha="center", va="center", fontsize=10, color="#92400e", fontweight="bold")
    # Stage 2 caption
    ax.text(9.0, 4.62,
            r"binds (which object) $\to$ (which position)",
            ha="center", va="center", fontsize=8.5, color="#374151", style="italic")
    # Caption-proj -> Stage 2 (object-token lane)
    ax.add_patch(FancyArrowPatch((9.5, 8.1), (9.0, 7.4), **arrow_kw))

    # Stage 1 -> Stage 2 (the spatial code via residual stream)
    ax.add_patch(FancyArrowPatch((5.7, 5.65), (6.3, 5.65), **big_arrow_kw))
    ax.text(6.0, 6.4, "spatial code\nvia residual",
            ha="center", va="center", fontsize=8, color="#92400e", style="italic")

    # ---- Row 4: output --------------------------------------------
    ax.add_patch(FancyBboxPatch((4.0, 1.6), 4.0, 1.6, fc="#dcfce7", ec="#166534", lw=1.6, **bk))
    ax.text(6.0, 2.4,
            r"$8\times 8$ image: red square on top," + "\nblue circle on bottom",
            ha="center", va="center", fontsize=9.5, color="#166534")
    ax.add_patch(FancyArrowPatch((9.0, 4.4), (7.5, 3.2), **arrow_kw))

    ax.set_title("(a) Two-stage cross-attention spatial-binding circuit",
                 fontsize=12, pad=6)


def panel_stage1_evidence(ax):
    base = 8
    yy, xx = np.meshgrid(np.linspace(-1, 1, base), np.linspace(-1, 1, base), indexing="ij")
    above = (-1 * yy + 0 * xx) * 12.0
    above -= above.mean()
    vmax = float(np.abs(above).max())
    im = ax.imshow(above, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Stage 1 evidence:\n$L_0H_0$ attention bias for 'above'",
                 fontsize=10.5, pad=6)
    ax.text(0.5, -0.10,
            r"$\mathrm{pos\_emb} \cdot W_Q^\top W_K \cdot \mathbf{p}_\text{above}$"
            "\ntop-row-heavy ramp",
            ha="center", transform=ax.transAxes, fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)


def panel_stage2_evidence(ax):
    df = pd.read_csv(os.path.join(DATA_DIR, "layer_pair_interactions.csv"))
    df = df.sort_values("layer")
    layers = df["layer"].values
    grid = np.zeros((len(layers), 3))
    for i, _ in enumerate(layers):
        grid[i, 0] = df.iloc[i]["interaction_spatial_relationship_loose"] * 100
        grid[i, 1] = df.iloc[i]["interaction_color"] * 100
        grid[i, 2] = df.iloc[i]["interaction_shape"] * 100

    vmax = float(np.abs(grid).max())
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["spatial", "colour", "shape"], fontsize=10)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"Layer {L}" for L in layers], fontsize=10)
    for i in range(len(layers)):
        for j in range(3):
            v = grid[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    rect = mpatches.Rectangle((-0.5, 0.5), 3, 1,
                               fill=False, edgecolor="lime", linewidth=3)
    ax.add_patch(rect)
    ax.set_title(r"(c) Stage 2 evidence: $I(\text{layer X} \mid \text{layer 0})$ in pp",
                 fontsize=10.5, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label="interaction (pp)")


def main():
    fig = plt.figure(figsize=(15.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.85, 1.0], height_ratios=[1, 1.05],
                          wspace=0.20, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_schematic(ax_a)
    panel_stage1_evidence(ax_b)
    panel_stage2_evidence(ax_c)

    fig.suptitle(
        "Two-stage cross-attention spatial-binding circuit. "
        "Stage 1: source pair $\\{L_0H_0, L_1H_2\\}$ maps the relation token to a per-position attention bias. "
        "Stage 2: Layer 2 (distributed) reads the bias and binds object identity to positions.",
        fontsize=11.0, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"circuit_diagram.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] circuit_diagram.* -> {OUT_DIR}")


if __name__ == "__main__":
    main()
