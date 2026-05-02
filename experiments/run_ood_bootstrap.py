"""Compute bootstrap CIs on the existing held-out OOD evaluation data
(unseen color x shape pairs + synonym-paraphrased relations).

Tests whether L0H0 ablation effect is robust across:
  (a) prompt resampling within each held-out set
  (b) per-prompt consistency (do all prompts in a set show the effect?)

Outputs:
  data/ood_bootstrap.csv   per-set bootstrap CIs for L0H0 / L1H2 / control
  data/ood_per_prompt.csv  per-prompt L0H0 ablation effect for outlier check
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
SRC_DIR = join(PROJECT_ROOT, "experiments")
OUT_DIR = join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(join(SRC_DIR, "held_out_eval_eval_rows.csv"))
    sets = sorted(df["set"].unique())
    conditions = ["baseline", "ablate_L0H0", "ablate_L1H2", "ablate_L0H2_control"]

    # ============================
    # Per-set bootstrap CIs
    # ============================
    rows = []
    rng = np.random.default_rng(43)
    for s in sets:
        for c in conditions:
            sub = df[(df["set"] == s) & (df["condition_label"] == c)]
            # group per-prompt mean
            per_prompt = sub.groupby("prompt")["spatial_relationship_loose"].mean().values
            n = len(per_prompt)
            iters = []
            for _ in range(2000):
                ix = rng.integers(0, n, size=n)
                iters.append(per_prompt[ix].mean())
            iters = np.array(iters)
            rows.append(dict(
                set=s, condition=c, n_prompts=n,
                mean=float(per_prompt.mean()),
                ci_lo=float(np.percentile(iters, 2.5)),
                ci_hi=float(np.percentile(iters, 97.5)),
            ))
    bs_df = pd.DataFrame(rows)
    bs_df.to_csv(join(OUT_DIR, "ood_bootstrap.csv"), index=False)
    print(bs_df.to_string(index=False))

    # ============================
    # Compute Δ(L0H0 - baseline) bootstrap CI per set
    # ============================
    print()
    print("=== Δ(L0H0) bootstrap CIs per OOD set ===")
    for s in sets:
        base_g = df[(df["set"] == s) & (df["condition_label"] == "baseline")]
        ablate_g = df[(df["set"] == s) & (df["condition_label"] == "ablate_L0H0")]
        # match on prompt
        base_per = base_g.groupby("prompt")["spatial_relationship_loose"].mean()
        ablate_per = ablate_g.groupby("prompt")["spatial_relationship_loose"].mean()
        common = base_per.index.intersection(ablate_per.index)
        deltas = (base_per[common] - ablate_per[common]).values
        n = len(deltas)
        iters = []
        for _ in range(2000):
            ix = rng.integers(0, n, size=n)
            iters.append(deltas[ix].mean())
        iters = np.array(iters)
        print(f"  {s:>15s}: Δ = {deltas.mean():+.3f}  95% CI [{np.percentile(iters,2.5):+.3f}, {np.percentile(iters,97.5):+.3f}]  n={n}")

    # ============================
    # Per-prompt deltas (L0H0)
    # ============================
    pp_rows = []
    for s in sets:
        base_g = df[(df["set"] == s) & (df["condition_label"] == "baseline")]
        ablate_g = df[(df["set"] == s) & (df["condition_label"] == "ablate_L0H0")]
        base_per = base_g.groupby("prompt")["spatial_relationship_loose"].mean()
        ablate_per = ablate_g.groupby("prompt")["spatial_relationship_loose"].mean()
        for p in base_per.index.intersection(ablate_per.index):
            pp_rows.append(dict(
                set=s, prompt=p,
                baseline_loose=float(base_per[p]),
                ablated_loose=float(ablate_per[p]),
                delta=float(base_per[p] - ablate_per[p]),
            ))
    pd.DataFrame(pp_rows).to_csv(join(OUT_DIR, "ood_per_prompt.csv"), index=False)


if __name__ == "__main__":
    main()
