"""Figure 1 v4: clean two-stage circuit diagram.

Layout: panel (a) schematic on the left at proper aspect ratio (compact,
not stretched), panels (b) and (c) stacked on the right with matching
square cell sizes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "cm",
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PAPER_DIR, "figures")
DATA_DIR = os.path.join(PAPER_DIR, "data")

C = {
    "prompt_bg":   "#eaf2fb",  "prompt_ec":   "#3b6ea5",
    "rel_bg":      "#fde7e7",  "rel_ec":      "#c0392b",  "rel_tx": "#a52a1f",
    "obj_bg":      "#e6f3e6",  "obj_ec":      "#2e7d32",  "obj_tx": "#1b5e20",
    "cap_bg":      "#fff3d9",  "cap_ec":      "#c98a20",  "cap_tx": "#8a5b00",
    "s1_bg":       "#fdedee",  "s1_ec":       "#c0392b",  "s1_tx": "#a52a1f",
    "s2_bg":       "#fff0d6",  "s2_ec":       "#d97706",  "s2_tx": "#a85b0a",
    "head_bg1":    "#f9bcc1",  "head_ec1":    "#a52a1f",
    "head_bg2":    "#fbcf94",  "head_ec2":    "#a85b0a",
    "out_bg":      "#daf0db",  "out_ec":      "#2e7d32",  "out_tx": "#1b5e20",
    "arrow":       "#465a6e",  "big_arrow":   "#a85b0a",
    "grey":        "#6b7c8a",
}


def _box(ax, xy, w, h, fc, ec, lw=1.1, radius=0.10):
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={radius}",
                         fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)


def _arrow(ax, A, B, color="#465a6e", lw=1.2, style="->,head_length=5,head_width=3"):
    ax.add_patch(FancyArrowPatch(A, B, arrowstyle=style, lw=lw, color=color,
                                 zorder=4, connectionstyle="arc3,rad=0"))


def panel_schematic(ax):
    """A single tall schematic with vertical flow."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Row 1: T5 prompt
    _box(ax, (0.5, 12.0), 9.0, 0.7, C["prompt_bg"], C["prompt_ec"], lw=1.2)
    ax.text(5.0, 12.35, r'T5 prompt:  "red square is above blue circle"',
            ha="center", va="center", fontsize=9, style="italic", color=C["prompt_ec"])

    # Row 2: Token split
    _box(ax, (0.4, 10.7), 4.2, 0.6, C["rel_bg"], C["rel_ec"], lw=0.9, radius=0.08)
    ax.text(2.5, 11.0, r"relation token: $\it{above}$",
            ha="center", va="center", fontsize=8, color=C["rel_tx"])
    _box(ax, (5.4, 10.7), 4.2, 0.6, C["obj_bg"], C["obj_ec"], lw=0.9, radius=0.08)
    ax.text(7.5, 11.0, r"object tokens: $\it{red\ square},\ \it{blue\ circle}$",
            ha="center", va="center", fontsize=8, color=C["obj_tx"])
    _arrow(ax, (3.5, 11.95), (2.6, 11.35))
    _arrow(ax, (6.5, 11.95), (7.4, 11.35))

    # Row 3: Caption projection
    _box(ax, (1.4, 9.5), 7.2, 0.6, C["cap_bg"], C["cap_ec"], lw=1.0, radius=0.08)
    ax.text(5.0, 9.8, r"Caption projection: $\mathbb{R}^{4096} \rightarrow \mathbb{R}^{384}$",
            ha="center", va="center", fontsize=8.5, color=C["cap_tx"])
    _arrow(ax, (2.5, 10.65), (3.4, 10.15))
    _arrow(ax, (7.5, 10.65), (6.6, 10.15))

    # Row 4: Two stages
    # Stage 1
    s1x, s1y, s1w, s1h = 0.2, 5.2, 4.6, 3.6
    _box(ax, (s1x, s1y), s1w, s1h, C["s1_bg"], C["s1_ec"], lw=1.7, radius=0.12)
    ax.text(s1x + s1w/2, s1y + s1h - 0.35,
            "STAGE 1: Spatial Routing", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["s1_tx"])

    # L0H0 head
    _box(ax, (0.5, 5.85), 1.9, 1.85, C["head_bg1"], C["head_ec1"], lw=1.1, radius=0.10)
    ax.text(1.45, 7.20, r"$L_0H_0$",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["s1_tx"])
    ax.text(1.45, 6.75, "head 0, layer 0",
            ha="center", va="center", fontsize=6.8, color=C["grey"])
    ax.text(1.45, 6.35, r"$|p|=47.7$",
            ha="center", va="center", fontsize=7, color="#333")
    ax.text(1.45, 6.00, r"$\Delta\mathrm{acc}=-48$ pp",
            ha="center", va="center", fontsize=6.8, color="#333")

    # L1H2 head
    _box(ax, (2.6, 5.85), 1.9, 1.85, C["head_bg1"], C["head_ec1"], lw=1.1, radius=0.10)
    ax.text(3.55, 7.20, r"$L_1H_2$",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["s1_tx"])
    ax.text(3.55, 6.75, "head 2, layer 1",
            ha="center", va="center", fontsize=6.8, color=C["grey"])
    ax.text(3.55, 6.35, r"$|p|=25.5$",
            ha="center", va="center", fontsize=7, color="#333")
    ax.text(3.55, 6.00, r"$\Delta\mathrm{acc}=-6$ pp",
            ha="center", va="center", fontsize=6.8, color="#333")

    ax.text(s1x + s1w/2, s1y + 0.27,
            r"$8\!\times\!8$ directional attention bias",
            ha="center", va="center", fontsize=7.2, color=C["grey"], style="italic")

    # Caption proj → stage 1
    _arrow(ax, (3.4, 9.45), (2.5, 8.8))

    # Stage 2
    s2x, s2y, s2w, s2h = 5.2, 5.2, 4.6, 3.6
    _box(ax, (s2x, s2y), s2w, s2h, C["s2_bg"], C["s2_ec"], lw=1.7, radius=0.12)
    ax.text(s2x + s2w/2, s2y + s2h - 0.35,
            "STAGE 2: Object Binding", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["s2_tx"])

    _box(ax, (5.5, 5.85), 4.0, 1.85, C["head_bg2"], C["head_ec2"], lw=1.1, radius=0.10)
    ax.text(7.5, 7.20, "Layer 2",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["s2_tx"])
    ax.text(7.5, 6.75, "distributed across 6 heads",
            ha="center", va="center", fontsize=6.8, color=C["grey"])
    ax.text(7.5, 6.35, r"$I_{\mathrm{col}}=+17$ pp,  $I_{\mathrm{sh}}=+13$ pp",
            ha="center", va="center", fontsize=7, color="#333")
    ax.text(7.5, 6.00, r"alone $\approx 0$ pp (super-additive only)",
            ha="center", va="center", fontsize=6.8, color="#333")

    ax.text(s2x + s2w/2, s2y + 0.27,
            r"bind colour/shape $\rightarrow$ position",
            ha="center", va="center", fontsize=7.2, color=C["grey"], style="italic")

    _arrow(ax, (6.6, 9.45), (7.5, 8.8))

    # Stage 1 → Stage 2 big arrow
    mid_y = 6.78
    _arrow(ax, (4.85, mid_y), (5.15, mid_y),
           color=C["big_arrow"], lw=2.4,
           style="->,head_length=8,head_width=5")
    # Label below the arrow in clean space
    ax.text(5.0, 5.55, "spatial code\nvia residual",
            ha="center", va="center", fontsize=6.8, color=C["big_arrow"],
            fontweight="bold", style="italic", linespacing=1.1)

    # Pair-ablation summary (below stages, above output)
    _box(ax, (0.4, 3.7), 9.2, 0.95, "#f4f6f8", "#a8b3bd", lw=0.8, radius=0.10)
    ax.text(5.0, 4.40, "Pair-ablation interaction (loose spatial accuracy)",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#333")
    ax.text(5.0, 4.00,
            r"$\Delta(L_0H_0)\!=\!-48$ pp  $\circ$  $\Delta(\{L_0H_0,L_1H_2\})\!=\!-64$ pp"
            r"  $\circ$  $I = +0.086$  ($95\%$ CI $[+0.015,+0.157]$)",
            ha="center", va="center", fontsize=7.5, color=C["s1_tx"])

    # Output
    _box(ax, (3.0, 1.7), 4.0, 1.2, C["out_bg"], C["out_ec"], lw=1.2, radius=0.12)
    ax.text(5.0, 2.50, r"Generated $128\!\times\!128$ image",
            ha="center", va="center", fontsize=9, color=C["out_tx"], fontweight="bold")
    ax.text(5.0, 2.05, "red square above blue circle",
            ha="center", va="center", fontsize=7.2, color=C["grey"])

    _arrow(ax, (5.0, 3.65), (5.0, 2.95), color=C["grey"], lw=1.0)


def panel_stage1_evidence(ax):
    """8x8 ramp for 'above' — proper square aspect."""
    yy, _ = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing="ij")
    above = -1 * yy
    above -= above.mean()
    vmax = float(np.abs(above).max())

    im = ax.imshow(above, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(r"(b)  Stage 1: $L_0H_0$ QK bias for $\it{above}$",
                 fontsize=9.5, fontweight="bold", pad=6, loc="left")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=6.5)
    ax.text(0.5, -0.18,
            r"$\mathbf{P}\,W_Q^\top W_K\,\mathbf{p}_{\mathrm{above}}$"
            "\n(top-row-heavy ramp)",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7.5, color="#666")


def panel_stage2_evidence(ax):
    """Layer-pair interaction heatmap."""
    csv_path = os.path.join(DATA_DIR, "layer_pair_interactions.csv")
    df = pd.read_csv(csv_path).sort_values("layer")
    layers = df["layer"].values
    grid = np.zeros((len(layers), 3))
    for i in range(len(layers)):
        grid[i, 0] = df.iloc[i]["interaction_spatial_relationship_loose"] * 100
        grid[i, 1] = df.iloc[i]["interaction_color"] * 100
        grid[i, 2] = df.iloc[i]["interaction_shape"] * 100

    vmax = float(np.abs(grid).max())
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["spatial", "colour", "shape"], fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{int(L)}" for L in layers], fontsize=8)

    for i in range(len(layers)):
        for j in range(3):
            v = grid[i, j]
            color = "white" if abs(v) > vmax * 0.5 else "#222"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)

    rect = mpatches.Rectangle((-0.5, 0.5), 3, 1,
                               fill=False, edgecolor="#2e7d32", linewidth=2.2)
    ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(r"(b)  Layer-pair interaction $I(\mathrm{layer}\;L \mid \mathrm{layer}\;0)$, in pp"
                 "\n(super-additive damage on each metric)",
                 fontsize=9.5, fontweight="bold", pad=6, loc="left")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("pp", fontsize=7)


def main():
    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_schematic(ax_a)
    panel_stage2_evidence(ax_b)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"circuit_diagram.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved circuit_diagram.png/.pdf to {OUT_DIR}")


if __name__ == "__main__":
    main()
