"""Two complementary ablation tests for the original
``relation heads -> object-binding heads`` circuit hypothesis:

  A. Layer-pair ablation. Zero {layer 0}, {layer 0 + layer 2},
     {layer 0 + layer 3}, ..., and {layer 0 + layer 5}. If layer X is the
     "object-reader stage", ablating it on top of layer 0 should cause
     super-additive damage on color/shape (binding) accuracy.

  B. Triple-ablation. For each candidate h with the largest single-head
     binding damage (color + shape) plus the consensus top-5, ablate
     {L0H0, L1H2, h} together. The marginal damage of h on top of the
     source pair tests whether h is a downstream binding head.
"""
import os, sys, time, gc
from os.path import join
from typing import List, Tuple

import torch, pandas as pd, numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.zero_head_ablation_utils import apply_zero_head_ablation_multi, restore_processors
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings
from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES
from utils.ablation_eval_prompts import build_training_template_prompts


def all_layer_heads(layer: int, n_heads: int = 6) -> List[Tuple[int, int]]:
    return [(layer, h) for h in range(n_heads)]


def main():
    t0 = time.time()
    transformer, pipe, _, device, weight_dtype = load_model_and_pipeline(
        "objrel_T5_DiT_mini_pilot", 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)
    prompts, scene_infos = build_training_template_prompts(cache, DEFAULT_SPATIAL_PHRASES)
    print(f"[prompts] {len(prompts)} prompts")

    # =========================================================================
    # Test A: Layer-pair ablation
    # =========================================================================
    layer0 = all_layer_heads(0)
    PART_A = [
        ("baseline",       []),
        ("layer0",         layer0),
        ("layer0_layer1",  layer0 + all_layer_heads(1)),
        ("layer0_layer2",  layer0 + all_layer_heads(2)),
        ("layer0_layer3",  layer0 + all_layer_heads(3)),
        ("layer0_layer4",  layer0 + all_layer_heads(4)),
        ("layer0_layer5",  layer0 + all_layer_heads(5)),
    ]
    # =========================================================================
    # Test B: Triple-ablation
    # =========================================================================
    src = [(0, 0), (1, 2)]
    # h candidates: top-5 consensus + top non-source heads by single-head binding damage
    # (from all_heads_single_ablation_heldout.csv we picked L0H4, L0H1, L1H5, L0H2)
    PART_B_CANDS = [
        ("L0H5", (0, 5)),  # consensus
        ("L1H3", (1, 3)),  # consensus
        ("L2H1", (2, 1)),  # consensus
        ("L2H2", (2, 2)),  # consensus
        ("L0H4", (0, 4)),  # 3rd by projection
        ("L0H1", (0, 1)),  # high binding-damage rank
        ("L1H5", (1, 5)),  # top binding damage among non-sources
        ("L0H2", (0, 2)),  # top color damage among non-sources
    ]
    PART_B = [
        ("triple_baseline",       []),
        ("triple_src_only",       src),
    ] + [
        (f"triple_src_{lab}",     src + [coord]) for lab, coord in PART_B_CANDS
    ]

    rows_A, rows_B = [], []
    for label, head_set in PART_A:
        t1 = time.time()
        orig = apply_zero_head_ablation_multi(transformer, head_set) if head_set else None
        try:
            eval_df, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipe, prompts, scene_infos, cache,
                num_images=10, num_inference_steps=14, guidance_scale=4.5,
                generator_seed=42, device=device, weight_dtype=weight_dtype,
                show_prompt_progress=False,
            )
        finally:
            if orig is not None:
                restore_processors(transformer, orig)
        m = {k: float(eval_df[k].mean()) for k in
             ["spatial_relationship_loose", "color", "shape",
              "unique_binding", "exist_binding"] if k in eval_df.columns}
        rows_A.append({"label": label, "n_ablated": len(head_set), **m,
                       "elapsed_s": float(time.time() - t1)})
        print(f"[A] {label:>15s}  loose={m['spatial_relationship_loose']:.3f}  "
              f"color={m['color']:.3f}  shape={m['shape']:.3f}  "
              f"({time.time()-t1:.1f}s)", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    for label, head_set in PART_B:
        t1 = time.time()
        orig = apply_zero_head_ablation_multi(transformer, head_set) if head_set else None
        try:
            eval_df, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipe, prompts, scene_infos, cache,
                num_images=10, num_inference_steps=14, guidance_scale=4.5,
                generator_seed=42, device=device, weight_dtype=weight_dtype,
                show_prompt_progress=False,
            )
        finally:
            if orig is not None:
                restore_processors(transformer, orig)
        m = {k: float(eval_df[k].mean()) for k in
             ["spatial_relationship_loose", "color", "shape",
              "unique_binding", "exist_binding"] if k in eval_df.columns}
        rows_B.append({"label": label, "n_ablated": len(head_set), **m,
                       "elapsed_s": float(time.time() - t1)})
        print(f"[B] {label:>22s}  loose={m['spatial_relationship_loose']:.3f}  "
              f"color={m['color']:.3f}  shape={m['shape']:.3f}  "
              f"({time.time()-t1:.1f}s)", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    out_dir = join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows_A).to_csv(join(out_dir, "layer_pair_ablation.csv"), index=False)
    pd.DataFrame(rows_B).to_csv(join(out_dir, "triple_ablation.csv"), index=False)

    print(f"\n[save] -> {out_dir}/layer_pair_ablation.csv  +  triple_ablation.csv")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
