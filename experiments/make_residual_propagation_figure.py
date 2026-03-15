"""Layer-resolved residual stream propagation figure.

For each layer 0..5, compute the spatial-signal energy in the residual
stream as the projection of the per-position residual onto the ramp
direction matching the prompt's ground-truth relation, MINUS the mean
projection onto the other 7 ramps. This isolates relation-specific
spatial code from generic energy.

Two curves:
  baseline: residual at layer L
  L0H0-ablated: residual at layer L when L0H0 is zeroed

Bootstrap CIs over prompts.
"""
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_original_completion", "figures")
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
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments",
                                   "residual_per_layer_projections.csv"))
    # The 'diff_baseline_minus_ablated' rows have the L0H0 ablation difference;
    # 'baseline' rows have the untouched residual projections.
    # For each (layer, prompt) we need: "matched-ramp" projection - "mean-mismatched-ramp" projection.

    # Map left_of -> left for legacy
    df["ramp"] = df["ramp"].replace({"left_of": "left", "right_of": "right"})

    rows = []
    for cond, sub in df.groupby("condition"):
        for (layer, prompt, relation), grp in sub.groupby(["layer", "prompt", "relation"]):
            matched = grp[grp["ramp"] == relation]
            mis = grp[grp["ramp"] != relation]
            if matched.empty or mis.empty: continue
            m = float(matched.iloc[0]["best_channel_projection"])
            mu = float(mis["best_channel_projection"].abs().mean())
            rows.append(dict(
                condition=cond, layer=int(layer), prompt=prompt, relation=relation,
                matched_proj=m, mis_proj=mu,
                signal=abs(m) - mu,
            ))
    out = pd.DataFrame(rows)
    print("rows per condition x layer:")
    print(out.groupby(["condition", "layer"]).size())

    # Aggregate: mean signal per (condition, layer) with 1000-bootstrap CI over prompts
    n_layers = 6
    summary = []
    rng = np.random.default_rng(43)
    for cond in out["condition"].unique():
        for L in range(n_layers):
            vals = out[(out["condition"] == cond) & (out["layer"] == L)]["signal"].values
            if len(vals) == 0: continue
            iters = []
            for _ in range(1000):
                ix = rng.integers(0, len(vals), size=len(vals))
                iters.append(np.mean(vals[ix]))
            lo, hi = np.percentile(iters, [2.5, 97.5])
            summary.append(dict(condition=cond, layer=L, mean=np.mean(vals),
                                ci_lo=lo, ci_hi=hi, n=len(vals)))
    sdf = pd.DataFrame(summary)
    print("\n", sdf.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cond_color = {"baseline": "#1d4e89",
                  "diff_baseline_minus_ablated": "#cb2027"}
    cond_label = {"baseline": "baseline residual",
                  "diff_baseline_minus_ablated": r"baseline $-$ $L_0H_0$-ablated"}
    for cond in ["baseline", "diff_baseline_minus_ablated"]:
        s = sdf[sdf["condition"] == cond].sort_values("layer")
        if s.empty: continue
        ax.errorbar(s["layer"], s["mean"],
                    yerr=[s["mean"] - s["ci_lo"], s["ci_hi"] - s["mean"]],
                    marker="o", ms=8, lw=2.4, color=cond_color[cond],
                    label=cond_label[cond], capsize=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"matched-ramp projection $-$ mean mismatched-ramp" + "\n(spatial-signal energy in residual)")
    ax.set_title("Spatial signal enters at layer 0 and persists through later layers;\n"
                 "ablating $L_0H_0$ removes the signal at every layer",
                 fontsize=11.5, pad=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_xticks(range(n_layers))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"residual_propagation.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] residual_propagation.* -> {OUT_DIR}")


if __name__ == "__main__":
    main()
