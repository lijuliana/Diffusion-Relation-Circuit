"""Single-head ablation on the SAME 19-prompt training-template set used by
the layer-pair / triple-ablation experiments. Needed to compute the
super-additive interaction I_triple(h | src) = Delta(src+h) - Delta(src) - Delta(h)
on a consistent prompt set.

Heads tested: the 8 candidates in the triple-ablation experiment plus the
two source heads, plus all 6 heads of layer 2 (to test the layer-2 binding
hypothesis at single-head resolution).
"""
import os, sys, time, gc
from os.path import join

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


HEADS = [
    ("baseline", []),
    ("L0H0", [(0, 0)]),
    ("L1H2", [(1, 2)]),
    ("L0H1", [(0, 1)]),
    ("L0H2", [(0, 2)]),
    ("L0H4", [(0, 4)]),
    ("L0H5", [(0, 5)]),
    ("L1H3", [(1, 3)]),
    ("L1H5", [(1, 5)]),
    # Layer-2 heads as singletons
    ("L2H0", [(2, 0)]),
    ("L2H1", [(2, 1)]),
    ("L2H2", [(2, 2)]),
    ("L2H3", [(2, 3)]),
    ("L2H4", [(2, 4)]),
    ("L2H5", [(2, 5)]),
]


def main():
    t0 = time.time()
    transformer, pipe, _, device, weight_dtype = load_model_and_pipeline(
        "objrel_T5_DiT_mini_pilot", 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)
    prompts, scene_infos = build_training_template_prompts(cache, DEFAULT_SPATIAL_PHRASES)
    print(f"[prompts] {len(prompts)}")

    rows = []
    for label, heads in HEADS:
        t1 = time.time()
        orig = apply_zero_head_ablation_multi(transformer, heads) if heads else None
        try:
            ev, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipe, prompts, scene_infos, cache,
                num_images=10, num_inference_steps=14, guidance_scale=4.5,
                generator_seed=42, device=device, weight_dtype=weight_dtype,
                show_prompt_progress=False,
            )
        finally:
            if orig is not None:
                restore_processors(transformer, orig)
        m = {k: float(ev[k].mean()) for k in
             ["spatial_relationship_loose", "color", "shape",
              "unique_binding", "exist_binding"] if k in ev.columns}
        rows.append({"label": label, **m, "elapsed_s": float(time.time() - t1)})
        print(f"{label:>5s}  loose={m['spatial_relationship_loose']:.3f}  "
              f"color={m['color']:.3f}  shape={m['shape']:.3f}  ({time.time()-t1:.1f}s)",
              flush=True)
        gc.collect(); torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out_path = join(PROJECT_ROOT, "results",
                    "single_head_binding_19prompt.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
