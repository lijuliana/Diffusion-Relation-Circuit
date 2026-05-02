"""OOD ablation effect with prompt-bootstrap CIs.

For each held-out set (unseen color x shape, synonym-paraphrased), show
loose spatial accuracy under each ablation condition with 95% CIs.
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


def main():
    df = pd.read_csv(join(DATA_DIR, "ood_bootstrap.csv"))
    sets = ["unseen_pair", "synonym"]
    set_label = {"unseen_pair": "unseen color × shape\n(n=12 prompts)",
                 "synonym": "synonym-paraphrased relations\n(n=7 prompts)"}
    conditions = ["baseline", "ablate_L0H0", "ablate_L1H2", "ablate_L0H2_control"]
    cond_label = {
        "baseline":             "baseline",
        "ablate_L0H0":          "$L_0H_0$ ablated",
        "ablate_L1H2":          "$L_1H_2$ ablated",
        "ablate_L0H2_control":  "$L_0H_2$ control",
    }
    cond_color = {
        "baseline":             "#888888",
        "ablate_L0H0":          "#cb2027",
        "ablate_L1H2":          "#1d4e89",
        "ablate_L0H2_control":  "#bbbbbb",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, s in zip(axes, sets):
        sub = df[df["set"] == s]
        ys = []
        ylo = []
        yhi = []
        for c in conditions:
            row = sub[sub["condition"] == c].iloc[0]
            ys.append(row["mean"]); ylo.append(row["mean"] - row["ci_lo"])
            yhi.append(row["ci_hi"] - row["mean"])
        x = np.arange(len(conditions))
        bars = ax.bar(x, ys, yerr=[ylo, yhi], capsize=5,
                      color=[cond_color[c] for c in conditions],
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([cond_label[c] for c in conditions],
                                              rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(set_label[s], fontsize=11.5, pad=8)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if s == "unseen_pair":
            ax.set_ylabel("Loose spatial accuracy")
    fig.suptitle("Out-of-distribution generalisation: $L_0H_0$ ablation effect with $95\\%$ prompt-bootstrap CIs",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(join(OUT_DIR, f"ood_bootstrap.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] ood_bootstrap.* -> {OUT_DIR}")


if __name__ == "__main__":
    main()
