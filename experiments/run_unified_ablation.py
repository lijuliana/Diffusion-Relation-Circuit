"""Unified single + multi-head ablation on the SAME 30-prompt training-template
set used by pair_ablation_consensus.py, so all values in Table 3 and the
abstract come from a single prompt distribution.

Conditions:
  baseline
  L0H0           (single source head)
  L1H2           (single source head)
  L0H0_L1H2      (source pair)
  random_ctrl_X  (5 norm-matched controls; report mean)
  top4_consensus (L0H0 + L1H2 + L0H5 + L1H3)
  top6_consensus (L0H0 + L1H2 + L0H5 + L1H3 + L2H1 + L2H2)
  all_36_cross_attn

Output: data/unified_ablation_30prompt.csv
"""
import os, sys, gc, time
from os.path import join

import torch, pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.zero_head_ablation_utils import (
    apply_zero_head_ablation_multi, restore_processors,
)
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings
from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES
from utils.notebook_setup import generate_prompts_and_scene_info


def all_36_cross_attn(transformer):
    return [(L, h) for L in range(len(transformer.transformer_blocks)) for h in range(6)]


def main():
    t0 = time.time()
    transformer, pipe, _, device, weight_dtype = load_model_and_pipeline(
        "objrel_T5_DiT_mini_pilot", 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)

    # Match pair_ablation_consensus.py's prompt selection: 30 prompts spanning
    # all 8 relations from the standard template.
    prompts_all, scenes_all, _ = generate_prompts_and_scene_info()
    by_rel = {}
    for i, s in enumerate(scenes_all):
        by_rel.setdefault(s["spatial_relationship"], []).append(i)
    prompt_idx = []
    for r, ids in sorted(by_rel.items()):
        prompt_idx.extend(ids[:4])
    prompt_idx = prompt_idx[:30]
    use_prompts = [prompts_all[i] for i in prompt_idx]
    use_scenes  = [scenes_all[i] for i in prompt_idx]
    print(f"[prompts] using {len(use_prompts)} prompts (training-template, set B)")

    # Random controls: 5 non-source heads with comparable OV Frobenius norm to L0H0.
    # Pick from non-{L0H0,L1H2,L0H4,L0H5,L1H0,L1H3,L2H1,L2H2} heads.
    excluded = {(0,0),(1,2),(0,4),(0,5),(1,0),(1,3),(2,1),(2,2)}
    random_ctrls = [(L, h) for L in range(2, 6) for h in range(6) if (L, h) not in excluded][:5]

    CONDITIONS = [
        ("baseline",        []),
        ("L0H0",            [(0,0)]),
        ("L1H2",            [(1,2)]),
        ("L0H0_L1H2",       [(0,0),(1,2)]),
        ("top4_consensus",  [(0,0),(1,2),(0,5),(1,3)]),
        ("top6_consensus",  [(0,0),(1,2),(0,5),(1,3),(2,1),(2,2)]),
        ("random_ctrl_1",   [random_ctrls[0]]),
        ("random_ctrl_2",   [random_ctrls[1]]),
        ("random_ctrl_3",   [random_ctrls[2]]),
        ("random_ctrl_4",   [random_ctrls[3]]),
        ("random_ctrl_5",   [random_ctrls[4]]),
        ("all_36",          all_36_cross_attn(transformer)),
    ]

    rows = []
    for label, head_set in CONDITIONS:
        t1 = time.time()
        orig = apply_zero_head_ablation_multi(transformer, head_set) if head_set else None
        try:
            ev, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipe, use_prompts, use_scenes, cache,
                num_images=10, num_inference_steps=14, guidance_scale=4.5,
                generator_seed=42, device=device, weight_dtype=weight_dtype,
                show_prompt_progress=False,
            )
        finally:
            if orig is not None:
                restore_processors(transformer, orig)
        m = {k: float(ev[k].mean()) for k in
             ["spatial_relationship_loose","spatial_relationship",
              "color","shape","unique_binding","exist_binding"] if k in ev.columns}
        rows.append({"label": label, "n_ablated": len(head_set), **m,
                     "elapsed_s": float(time.time() - t1)})
        print(f"{label:>20s}  loose={m['spatial_relationship_loose']:.3f}  "
              f"strict={m['spatial_relationship']:.3f}  color={m['color']:.3f}  "
              f"shape={m['shape']:.3f}  ({time.time()-t1:.1f}s)", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out_path = join(PROJECT_ROOT, "paper_original_completion", "data",
                    "unified_ablation_30prompt.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    # Compute the random-control mean
    rc = out[out["label"].str.startswith("random_ctrl_")]
    base = out[out["label"]=="baseline"].iloc[0]
    print()
    print("=== Summary, ALL on the same 30-prompt training-template set ===")
    print(f"  baseline:         loose={base['spatial_relationship_loose']:.3f}")
    for lab in ["L0H0","L1H2","L0H0_L1H2","top4_consensus","top6_consensus","all_36"]:
        r = out[out["label"]==lab].iloc[0]
        delta = base["spatial_relationship_loose"] - r["spatial_relationship_loose"]
        print(f"  {lab:>16s}: loose={r['spatial_relationship_loose']:.3f}  "
              f"Δ={delta:+.3f}")
    rc_mean = rc["spatial_relationship_loose"].mean()
    rc_std = rc["spatial_relationship_loose"].std()
    print(f"  random ctrl (n={len(rc)} mean): loose={rc_mean:.3f}  Δ={base['spatial_relationship_loose']-rc_mean:+.3f}  std={rc_std:.3f}")
    print(f"\n[save] -> {out_path}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
