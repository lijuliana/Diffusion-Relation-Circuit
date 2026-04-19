"""Reformat qualitative ablation grid for publication quality.

Loads the 5 sub-panels from the notebook output and re-renders them
with clean column headers, color-coded borders, and tight margins.
"""
import os
import numpy as np
from PIL import Image
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PAPER_DIR, "figures")
SRC_PATH = "/tmp/notebook_qual_cell66_out1.png"

# Panel column ranges in source image
PANEL_COLS = [
    (9, 441),
    (456, 888),
    (903, 1336),
    (1346, 1782),
    (1791, 2234),
]
# Image grid only, excluding original axis titles
BODY_ROWS = (110, 520)

LABELS = [
    ("Baseline", ""),
    ("Ablate 1 head", r"$L_1H_2$"),
    ("Ablate 2 heads", r"$L_1H_2 + L_0H_0$"),
    ("Ablate 4 heads", r"$+\, L_2H_2 + L_3H_4$"),
    ("Ablate all 36 heads", "(every cross-attention head)"),
]
PANEL_COLORS = ["#2e7d32", "#558b2f", "#f57c00", "#d84315", "#b71c1c"]


def main():
    src = np.array(Image.open(SRC_PATH).convert("RGB"))
    panels = []
    for c0, c1 in PANEL_COLS:
        crop = src[BODY_ROWS[0]:BODY_ROWS[1], c0:c1]
        panels.append(crop)

    # figsize tuned: 5 panels of ~430x410 each, tight margins
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 3.0),
                             gridspec_kw={"wspace": 0.06})
    for ax, panel, (title, sub), color in zip(axes, panels, LABELS, PANEL_COLORS):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)
        full_title = f"{title}\n{sub}" if sub else f"{title}\n"
        ax.set_title(full_title, fontsize=10.5, fontweight="bold", color=color, pad=4,
                     linespacing=1.25)

    # Tight layout — let LaTeX caption handle the descriptive text
    fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.02, wspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"qualitative_ablation_grid.{ext}"),
                    bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    print(f"Saved qualitative_ablation_grid.png/.pdf to {OUT_DIR}")


if __name__ == "__main__":
    main()
