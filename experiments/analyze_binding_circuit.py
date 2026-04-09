"""Analysis of layer-pair and triple-ablation interaction scores on
binding (color, shape) accuracy.

Computes:
  Layer-pair interactions:
    I_X(layer_pair) = Delta_X(layer0 + layerL) - Delta_X(layer0) - Delta_X(layerL alone)
  Triple-ablation interactions:
    I_X(h | src) = Delta_X(src + h) - Delta_X(src) - Delta_X(h alone)
  where X in {spatial_loose, color, shape}.

Baselines come from layer_pair_ablation.csv. Layer-X alone effects come
from layer_ablation_results.csv (same 19-prompt training-template set).
Single-head Delta(h alone) comes from single_head_binding_19prompt.csv.
"""
import os
from os.path import join

import numpy as np
import pandas as pd

PROJECT_ROOT = "/DiffusionInterp"
DATA_DIR    = join(PROJECT_ROOT, "paper_original_completion", "data")
EXP_DIR     = join(PROJECT_ROOT, "experiments")


def main():
    pair_df    = pd.read_csv(join(DATA_DIR, "layer_pair_ablation.csv"))
    triple_df  = pd.read_csv(join(DATA_DIR, "triple_ablation.csv"))
    layer_df   = pd.read_csv(join(EXP_DIR,  "layer_ablation_results.csv"))
    single_df  = pd.read_csv(join(DATA_DIR, "single_head_binding_19prompt.csv"))

    metrics = ["spatial_relationship_loose", "color", "shape"]

    # ============================
    # Layer-pair interactions
    # ============================
    print("=== Layer-pair interactions: I(layer X | layer 0) on binding ===")
    print(f"baseline (this run): spatial={pair_df.iloc[0]['spatial_relationship_loose']:.3f}  "
          f"color={pair_df.iloc[0]['color']:.3f}  shape={pair_df.iloc[0]['shape']:.3f}")
    print()
    base = pair_df[pair_df["label"] == "baseline"].iloc[0]
    layer0 = pair_df[pair_df["label"] == "layer0"].iloc[0]

    rows = []
    for L in range(1, 6):
        # both ablated
        both = pair_df[pair_df["label"] == f"layer0_layer{L}"]
        if both.empty: continue
        both = both.iloc[0]
        # layer L alone (from old layer_ablation_results.csv with same prompt set)
        layerL_alone = layer_df[layer_df["label"] == f"layer_{L}"]
        if layerL_alone.empty: continue
        layerL_alone = layerL_alone.iloc[0]
        # Match the baseline metric column name
        # layer_ablation_results uses 'spatial_relationship_loose', 'color', 'shape'
        row = {"layer": L}
        for m in metrics:
            d_layer0      = float(base[m]) - float(layer0[m])
            d_layerL_alone = float(layer_df[layer_df["label"] == "baseline"].iloc[0][m]) - float(layerL_alone[m])
            d_both        = float(base[m]) - float(both[m])
            inter         = d_both - d_layer0 - d_layerL_alone
            row[f"d_layer0_{m}"]      = d_layer0
            row[f"d_layerL_{m}"]      = d_layerL_alone
            row[f"d_both_{m}"]        = d_both
            row[f"interaction_{m}"]   = inter
        rows.append(row)
    pair_inter = pd.DataFrame(rows)
    print(pair_inter[["layer",
                      "interaction_spatial_relationship_loose",
                      "interaction_color",
                      "interaction_shape"]].to_string(index=False))
    print()

    # ============================
    # Triple-ablation interactions: I(h | src)
    # ============================
    print("=== Triple-ablation interactions: I(h | src={L0H0, L1H2}) on binding ===")
    base_t   = triple_df[triple_df["label"] == "triple_baseline"].iloc[0]
    src_only = triple_df[triple_df["label"] == "triple_src_only"].iloc[0]

    cand_labels = [
        ("L0H5", "triple_src_L0H5"),
        ("L1H3", "triple_src_L1H3"),
        ("L2H1", "triple_src_L2H1"),
        ("L2H2", "triple_src_L2H2"),
        ("L0H4", "triple_src_L0H4"),
        ("L0H1", "triple_src_L0H1"),
        ("L1H5", "triple_src_L1H5"),
        ("L0H2", "triple_src_L0H2"),
    ]

    rows = []
    for cand, key in cand_labels:
        srcplush = triple_df[triple_df["label"] == key]
        if srcplush.empty: continue
        srcplush = srcplush.iloc[0]
        # h alone from single_head_binding_19prompt.csv
        halone = single_df[single_df["label"] == cand]
        if halone.empty:
            print(f"[skip] {cand}: missing in single_head_binding_19prompt"); continue
        halone = halone.iloc[0]
        sb_base = single_df[single_df["label"] == "baseline"].iloc[0]
        row = {"candidate": cand}
        for m in metrics:
            d_src      = float(base_t[m]) - float(src_only[m])
            d_h_alone  = float(sb_base[m]) - float(halone[m])
            d_srcplush = float(base_t[m]) - float(srcplush[m])
            inter      = d_srcplush - d_src - d_h_alone
            row[f"d_src_{m}"]       = d_src
            row[f"d_h_alone_{m}"]   = d_h_alone
            row[f"d_srcplush_{m}"]  = d_srcplush
            row[f"interaction_{m}"] = inter
        rows.append(row)
    tri_inter = pd.DataFrame(rows)

    print(tri_inter[["candidate",
                     "interaction_spatial_relationship_loose",
                     "interaction_color",
                     "interaction_shape"]].to_string(index=False))
    print()

    # ============================
    # Save
    # ============================
    pair_inter.to_csv(join(DATA_DIR, "layer_pair_interactions.csv"), index=False)
    tri_inter.to_csv (join(DATA_DIR, "triple_ablation_interactions.csv"), index=False)

    # ============================
    # Summary
    # ============================
    print("=== Summary ===")
    print("Layer-pair: super-additive layers ON BINDING (color+shape interaction):")
    pair_inter["binding_inter"] = (pair_inter["interaction_color"] + pair_inter["interaction_shape"]) / 2
    for _, r in pair_inter.sort_values("binding_inter", ascending=False).iterrows():
        print(f"  Layer {int(r['layer'])}: I_color={r['interaction_color']:+.3f}  "
              f"I_shape={r['interaction_shape']:+.3f}  "
              f"I_spatial={r['interaction_spatial_relationship_loose']:+.3f}")
    print()
    print("Triple-ablation: super-additive heads ON BINDING (color+shape interaction):")
    tri_inter["binding_inter"] = (tri_inter["interaction_color"] + tri_inter["interaction_shape"]) / 2
    for _, r in tri_inter.sort_values("binding_inter", ascending=False).iterrows():
        print(f"  {r['candidate']}: I_color={r['interaction_color']:+.3f}  "
              f"I_shape={r['interaction_shape']:+.3f}  "
              f"I_spatial={r['interaction_spatial_relationship_loose']:+.3f}")


if __name__ == "__main__":
    main()
