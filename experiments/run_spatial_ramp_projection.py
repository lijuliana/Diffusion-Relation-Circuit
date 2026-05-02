"""Phase B3 — per-head spatial-ramp projection IN VIVO.

Loads `head_out_pos_pattern.npz` (saved by run_activation_correlation.py).
For each prompt with relation r, builds the canonical 8x8 ramp R_r and
correlates the per-position write-magnitude pattern of every head with R_r.

Per-head score = mean over prompts of |Pearson(head_pos_pattern_p, R_{rel(p)})|.

A high score means the head's per-position write magnitude tracks the
direction of the spatial relation in vivo — a direct downstream signature.
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_ramp(rel: str, base: int = 8) -> np.ndarray:
    directions = {
        "above": (-1, 0), "below": (1, 0),
        "left": (0, -1), "right": (0, 1),
        "left_of": (0, -1), "right_of": (0, 1),
        "upper_left": (-1, -1), "upper_right": (-1, 1),
        "lower_left": (1, -1), "lower_right": (1, 1),
    }
    dy, dx = directions[rel]
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, base), np.linspace(-1, 1, base), indexing="ij"
    )
    ramp = (dy * yy + dx * xx).astype(np.float32)
    ramp -= ramp.mean()
    return ramp.flatten()


def main():
    npz_path = join(PROJECT_ROOT, "results",
                    "head_out_pos_pattern.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"missing {npz_path}; run B2 first.")
    data = np.load(npz_path, allow_pickle=True)
    pos = data["pos_pattern"]  # (P, L, H, 64)
    rels = data["relations"]
    n_prompts, n_layers, n_heads, n_pos = pos.shape
    print(f"shape: {pos.shape}, relations: {set(rels.tolist())}")

    # Build ramp templates per relation
    ramps = {r: build_ramp(r) for r in set(rels.tolist())}

    # Per-prompt per-(L,H) abs correlation with the ground-truth ramp
    abs_corr = np.zeros((n_prompts, n_layers, n_heads), dtype=np.float32)
    for p in range(n_prompts):
        ramp = ramps[rels[p]]
        ramp_z = (ramp - ramp.mean()) / (ramp.std() + 1e-9)
        for li in range(n_layers):
            for hi in range(n_heads):
                v = pos[p, li, hi]
                if v.std() < 1e-9:
                    abs_corr[p, li, hi] = 0
                    continue
                vz = (v - v.mean()) / (v.std() + 1e-9)
                abs_corr[p, li, hi] = abs(float(np.mean(vz * ramp_z)))

    mean = abs_corr.mean(axis=0)
    std = abs_corr.std(axis=0)
    rows = []
    for li in range(n_layers):
        for hi in range(n_heads):
            rows.append(dict(
                layer=li, head=hi, label=f"L{li}H{hi}",
                ramp_abs_corr=float(mean[li, hi]),
                ramp_std=float(std[li, hi]),
                n_prompts=n_prompts,
            ))
    out_df = pd.DataFrame(rows).sort_values("ramp_abs_corr", ascending=False)
    out_path = join(PROJECT_ROOT, "results",
                    "spatial_ramp_projection.csv")
    out_df.to_csv(out_path, index=False)
    print()
    print("Top 15 heads by mean |ramp correlation|:")
    print(out_df.head(15).to_string(index=False))
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()
