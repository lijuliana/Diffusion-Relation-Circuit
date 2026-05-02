"""Phase B4 — combine 4 candidate-discovery methods into a consensus ranking.

Methods:
  1. OV-QK weight alignment  (verify_top_heads_align.csv -> abs_cosine, abs_projection)
  2. Path patching recovery-loss (path_patching_discovery.csv -> mean_loss)
  3. In-vivo activation correlation with L0H0 (activation_correlation.csv -> abs_mean_corr)
  4. In-vivo per-position write-magnitude correlation w/ spatial ramp
     (spatial_ramp_projection.csv -> ramp_abs_corr)

For each method, rank all 36 heads. Sum 4 ranks -> consensus rank.
L0H0 is excluded from "downstream" picks (it IS the source).
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = join(PROJECT_ROOT, "results")


def load_table(path, layer_col="layer", head_col="head"):
    df = pd.read_csv(path)
    df["label"] = df.apply(lambda r: f"L{int(r[layer_col])}H{int(r[head_col])}", axis=1)
    return df


def main():
    align = load_table(join(PROJECT_ROOT, "experiments", "verify_top_heads_align.csv"))
    align["abs_proj"] = align["projection"].abs()
    pp = pd.read_csv(join(DATA_DIR, "path_patching_discovery.csv"))
    pp_agg = pp.groupby(["candidate_layer", "candidate_head"]).agg(
        recovery_loss=("recovery_loss", "mean"),
    ).reset_index()
    pp_agg["label"] = pp_agg.apply(
        lambda r: f"L{int(r['candidate_layer'])}H{int(r['candidate_head'])}", axis=1
    )
    ac = pd.read_csv(join(DATA_DIR, "activation_correlation.csv"))
    sr = pd.read_csv(join(DATA_DIR, "spatial_ramp_projection.csv"))

    # Path-patching only sweeps non-L0H0 candidates (30 heads), but we need all 36
    # in the unified table; pad missing with NaN.
    all_heads = [f"L{l}H{h}" for l in range(6) for h in range(6)]

    cons = pd.DataFrame({"label": all_heads})
    cons = cons.merge(align[["label", "abs_cosine", "abs_proj", "energy"]], on="label", how="left")
    cons = cons.merge(pp_agg[["label", "recovery_loss"]], on="label", how="left")
    cons = cons.merge(ac[["label", "abs_mean_corr"]], on="label", how="left")
    cons = cons.merge(sr[["label", "ramp_abs_corr"]], on="label", how="left")

    # Higher = more downstream-likely.
    # Method 1: rank by abs_proj (OV write strength)
    # Method 2: rank by recovery_loss
    # Method 3: rank by abs_mean_corr
    # Method 4: rank by ramp_abs_corr
    # Build mean-rank (rank 1 = best). NaN -> imputed with column median so heads
    # that weren't measured get a neutral rank rather than being penalized.
    def rank_col(s, ascending=False):
        vals = s.fillna(float(s.median()))
        return vals.rank(ascending=ascending, method="min")

    cons["rank_align"]    = rank_col(cons["abs_proj"], ascending=False)
    cons["rank_path"]     = rank_col(cons["recovery_loss"], ascending=False)
    cons["rank_actcorr"]  = rank_col(cons["abs_mean_corr"], ascending=False)
    cons["rank_ramp"]     = rank_col(cons["ramp_abs_corr"], ascending=False)
    cons["consensus_rank"] = (cons["rank_align"] + cons["rank_path"] +
                              cons["rank_actcorr"] + cons["rank_ramp"]) / 4

    # Mark L0H0 (source) — exclude from candidate lists
    out = cons.sort_values("consensus_rank").reset_index(drop=True)
    out["is_source"] = out["label"] == "L0H0"

    out_path = join(DATA_DIR, "candidate_consensus.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\n[save] -> {out_path}")

    # Top 5 downstream candidates (excluding L0H0)
    candidates = out[~out["is_source"]].head(5)
    print()
    print("=== TOP 5 CONSENSUS DOWNSTREAM CANDIDATES ===")
    print(candidates[["label", "consensus_rank",
                      "rank_align", "rank_path", "rank_actcorr", "rank_ramp",
                      "abs_proj", "recovery_loss", "abs_mean_corr",
                      "ramp_abs_corr"]].to_string(index=False))


if __name__ == "__main__":
    main()
