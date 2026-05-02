"""Two figures:

A3: 6x6 single-head zero-ablation heatmap (held-out pairs, delta loose acc).
A4: cumulative multi-head ablation curve.
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


def fig_single_head():
    df = pd.read_csv(join(PROJECT_ROOT, "experiments", "all_heads_single_ablation_heldout.csv"))
    base = df[df["label"] == "baseline"].iloc[0]
    base_loose = base["spatial_loose"]
    df = df[df["label"] != "baseline"].copy()

    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    grid = np.zeros((n_layers, n_heads))
    for _, r in df.iterrows():
        grid[int(r["layer"]), int(r["head"])] = r["delta_loose"] * 100  # percentage points

    fig, ax = plt.subplots(figsize=(7.4, 6))
    vmax = float(np.max(np.abs(grid)))
    im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=vmax, aspect="equal")
    for li in range(n_layers):
        for hi in range(n_heads):
            v = grid[li, hi]
            txt_color = "white" if v / (vmax + 1e-9) > 0.55 else "black"
            ax.text(hi, li, f"{v:+.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Single-head zero-ablation: $\\Delta$ spatial accuracy (loose, held-out pairs)\n"
                 f"baseline = {base_loose:.0%}; values are $-\\Delta$ in pp ($+$ = damage)",
                 fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, label="accuracy drop (pp)")
    # Highlight L0H0, L1H2
    for layer, head in [(0, 0), (1, 2)]:
        rect = mpatches.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                  fill=False, edgecolor="lime", linewidth=3)
        ax.add_patch(rect)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(join(OUT_DIR, f"single_head_ablation.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig] single_head_ablation.* ->", OUT_DIR)


def fig_multi_head():
    # Use the unified 30-prompt set values (consistent with Table 3 and pair-ablation)
    df = pd.read_csv(join(PROJECT_ROOT, "results",
                          "unified_ablation_30prompt.csv"))
    rc_mean = df[df["label"].str.startswith("random_ctrl_")]["spatial_relationship_loose"].mean()
    pretty = {
        "baseline":        ("baseline", 0),
        "L0H0":            (r"$L_0H_0$", 1),
        "L0H0_L1H2":       (r"$+L_1H_2$", 2),
        "top4_consensus":  (r"$+L_0H_5,L_1H_3$", 4),
        "top6_consensus":  (r"$+L_2H_1,L_2H_2$", 6),
        "all_36":          ("all 36 heads", 36),
    }
    rows = []
    for lab, (name, n) in pretty.items():
        sub = df[df["label"] == lab]
        if len(sub):
            rows.append({"label": lab, "name": name, "n": n,
                         "loose": float(sub["spatial_relationship_loose"].iloc[0]) * 100,
                         "strict": float(sub["spatial_relationship"].iloc[0]) * 100})
    p = pd.DataFrame(rows).sort_values("n")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(p["n"], p["loose"], marker="o", lw=2.4, color="#cb2027",
            label="loose (correct relation $\\pm$ tolerance)", markersize=9)
    ax.plot(p["n"], p["strict"], marker="s", lw=2.0, color="#1d4e89", linestyle="--",
            label="strict", markersize=8)
    ax.axhline(rc_mean * 100, color="#888", linestyle=":", lw=1.2,
               label=f"random ctrl (mean of 5)")
    ax.axhline(100/8, color="#aaa", linestyle=":", lw=1.0,
               label="chance (1/8)")
    for _, r in p.iterrows():
        ax.annotate(r["name"], xy=(r["n"], r["loose"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, color="#444")
    ax.set_xticks([0, 1, 2, 4, 6, 36])
    ax.set_xlabel("Heads ablated (cumulative)")
    ax.set_ylabel("Loose spatial-relation accuracy (%)")
    ax.set_title("Cumulative multi-head zero-ablation collapses spatial behaviour\n"
                 r"$L_0H_0$ alone = 48 pp drop; $\{L_0H_0,L_1H_2\}$ = 64 pp; +4 more heads = 7 pp",
                 fontsize=11.5, pad=10)
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlim(-0.5, 50)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(join(OUT_DIR, f"multi_head_ablation.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig] multi_head_ablation.* ->", OUT_DIR)


if __name__ == "__main__":
    fig_single_head()
    fig_multi_head()
