"""Phase C1 — pair-ablation for the cross-method-consensus downstream
candidates (L0H5, L1H3, L2H1, L2H2). Source = L0H0. Includes L1H2 as
the previously-known reference.

For each (source, candidate) pair we run all-step zero-ablation in 4
conditions: baseline, src-only, cand-only, both. Interaction = ΔAcc(both)
- [ΔAcc(src) + ΔAcc(cand)]. Positive = super-additive (shared circuit).

Reuses run_pair_ablation_grid from utils/downstream_head_tracing.py.
"""
import os, sys, gc, time, json
from os.path import join

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.pixart_utils import state_dict_convert
from utils.downstream_head_tracing import run_pair_ablation_grid, DEFAULT_BEHAVIOR_COLS
from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES
from utils.notebook_setup import generate_prompts_and_scene_info


CKPT_NAME = "epoch_4000_step_160000.pth"
CKPTDIR = join(PROJECT_ROOT, "results", "objrel_T5_DiT_mini_pilot", "checkpoints")

SOURCE = (0, 0)
CANDIDATES = [
    (0, 5),  # L0H5: rank 6.25 in consensus
    (1, 3),  # L1H3: rank 8.5
    (2, 1),  # L2H1: rank 9.25
    (2, 2),  # L2H2: rank 9.5
    (1, 2),  # L1H2: known reference (rank 5.5)
]


def main():
    transformer, pipeline, tokenizer, device, weight_dtype = load_model_and_pipeline(
        run_name="objrel_T5_DiT_mini_pilot",
        ckpt_epoch=4000, ckpt_step=160000,
        project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)

    prompts, scenes, df = generate_prompts_and_scene_info()
    # Use a compact eval set: 30 prompts spanning all 8 relations
    prompt_idx = []
    by_rel = {}
    for i, s in enumerate(scenes):
        by_rel.setdefault(s["spatial_relationship"], []).append(i)
    for r, ids in sorted(by_rel.items()):
        prompt_idx.extend(ids[:4])
    prompt_idx = prompt_idx[:30]
    use_prompts = [prompts[i] for i in prompt_idx]
    use_scenes  = [scenes[i] for i in prompt_idx]
    print(f"Eval set: {len(use_prompts)} prompts.")

    metric_cols = list(DEFAULT_BEHAVIOR_COLS) + ["color", "shape"]

    summary_rows = []
    eval_rows = []
    for cand in CANDIDATES:
        t0 = time.time()
        print(f"\n=== source=L{SOURCE[0]}H{SOURCE[1]}  candidate=L{cand[0]}H{cand[1]} ===")
        sum_df, eval_df = run_pair_ablation_grid(
            pipeline=pipeline,
            ckptdir=CKPTDIR,
            ckpt_list=[CKPT_NAME],
            prompts=use_prompts,
            scene_infos=use_scenes,
            embedding_cache=cache,
            source_head=SOURCE,
            candidate_heads=[cand],
            state_dict_convert=state_dict_convert,
            device=device,
            weight_dtype=weight_dtype,
            num_images=10,
            num_inference_steps=14,
            guidance_scale=4.5,
            generator_seed=42,
            metric_cols=metric_cols,
            show_prompt_progress=False,
            progress_mode="tqdm",
        )
        summary_rows.append(sum_df)
        eval_rows.append(eval_df)
        print(f"   {time.time()-t0:.1f}s")

    out = pd.concat(summary_rows, ignore_index=True)
    out_path = join(PROJECT_ROOT, "results",
                    "pair_ablation_consensus.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    pd.concat(eval_rows, ignore_index=True).to_csv(
        join(PROJECT_ROOT, "results",
             "pair_ablation_consensus_eval_rows.csv"), index=False)
    print(f"\n[save] -> {out_path}")
    print()

    # Compute interaction scores
    print("=== INTERACTION SCORES (loose spatial accuracy) ===")
    interaction_rows = []
    for cand in CANDIDATES:
        c_lab = f"L{cand[0]}H{cand[1]}"
        sub = out[out["candidate_head"] == c_lab]
        if len(sub) == 0:
            sub = out[out["candidate_head"].isna()]
        # baseline rows have candidate_head=None, condition_key="baseline"
        b = out[out["condition_key"] == "baseline"]["spatial_relationship_loose"].mean()
        s_rows = out[out["condition_key"].str.startswith("src__")
                      & (out["candidate_head"] == c_lab)]
        s = s_rows["spatial_relationship_loose"].mean() if len(s_rows) else np.nan
        c_rows = out[out["condition_key"].str.startswith("cand__")
                      & (out["candidate_head"] == c_lab)]
        cv = c_rows["spatial_relationship_loose"].mean() if len(c_rows) else np.nan
        p_rows = out[out["condition_key"].str.startswith("pair__")
                      & (out["candidate_head"] == c_lab)]
        p = p_rows["spatial_relationship_loose"].mean() if len(p_rows) else np.nan
        d_s = b - s; d_c = b - cv; d_p = b - p
        I = d_p - (d_s + d_c)
        print(f"{c_lab:5s}: baseline={b:.2f} d_src={d_s:+.2f} d_cand={d_c:+.2f} "
              f"d_pair={d_p:+.2f}   interaction = {I:+.3f}")
        interaction_rows.append(dict(
            candidate=c_lab, baseline=b, d_src=d_s, d_cand=d_c, d_pair=d_p, interaction=I,
        ))
    pd.DataFrame(interaction_rows).to_csv(
        join(PROJECT_ROOT, "results",
             "pair_ablation_consensus_interactions.csv"), index=False)


if __name__ == "__main__":
    main()
