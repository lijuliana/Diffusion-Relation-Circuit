"""OOD generalization test: distractor tokens, extra adjectives, syntax
variants. For each variant, evaluate baseline and L0H0 zero-ablation
spatial accuracy on a held-out 24-prompt set spanning all 8 relations.

Variant types:
  - vanilla:  the standard template "<c1> <s1> is <rel> <c2> <s2>"
  - distract: prepend or append a filler clause unrelated to the spatial relation
  - adj:      add 'small'/'large' adjectives to objects
  - syntax:   "Above the <c2> <s2>, there is the <c1> <s1>" (relation-fronted)

For each variant, T5 embeddings are computed on the fly (the standard
prompt T5 cache won't have these). Tests whether L0H0's role generalises
beyond the training-template syntax.
"""
import os, sys, gc, time, itertools
from os.path import join

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.zero_head_ablation_utils import apply_zero_head_ablation_multi, restore_processors
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings


COLORS = ["red", "blue"]
SHAPES = ["square", "triangle", "circle"]
RELS_TEXT = {
    "above": "above", "below": "below",
    "left": "to the left of", "right": "to the right of",
    "upper_left": "to the upper left of",
    "upper_right": "to the upper right of",
    "lower_left": "to the lower left of",
    "lower_right": "to the lower right of",
}
RELS_FRONTED = {
    "above": ("above", "below"), "below": ("below", "above"),
    "left": ("to the left of", "to the right of"),
    "right": ("to the right of", "to the left of"),
    "upper_left": ("to the upper left of", "to the lower right of"),
    "upper_right": ("to the upper right of", "to the lower left of"),
    "lower_left": ("to the lower left of", "to the upper right of"),
    "lower_right": ("to the lower right of", "to the upper left of"),
}


def build_variant_prompts():
    """For each (color1, shape1, color2, shape2, relation) combination, produce
    a single prompt for each variant type. Use a fixed sample of 24 base prompts
    spanning all 8 relations (3 per relation)."""
    base_combos = []
    rng = np.random.default_rng(43)
    for rel in RELS_TEXT.keys():
        # pick 3 random (c1,s1,c2,s2) with c1!=c2
        opts = []
        for c1, c2 in itertools.product(COLORS, COLORS):
            if c1 == c2: continue
            for s1, s2 in itertools.product(SHAPES, SHAPES):
                if s1 == s2: continue
                opts.append((c1, s1, c2, s2))
        rng.shuffle(opts)
        for o in opts[:3]:
            base_combos.append((*o, rel))

    out = {"vanilla": [], "distract": [], "adj": [], "syntax": []}
    scenes_template = []
    for c1, s1, c2, s2, rel in base_combos:
        text = RELS_TEXT[rel]
        scene = {"color1": c1, "shape1": s1, "color2": c2, "shape2": s2,
                 "spatial_relationship": rel}
        scenes_template.append(scene)
        # vanilla
        out["vanilla"].append(f"{c1} {s1} is {text} {c2} {s2}")
        # distractor
        out["distract"].append(
            f"In a quiet scene with soft lighting, the {c1} {s1} is {text} the {c2} {s2}, and the sky is calm."
        )
        # extra adjectives
        adj_pair = ("small", "large") if rng.random() < 0.5 else ("large", "small")
        out["adj"].append(
            f"the {adj_pair[0]} {c1} {s1} is {text} the {adj_pair[1]} {c2} {s2}"
        )
        # syntax-fronted: "Above the C2 S2 is the C1 S1"
        out["syntax"].append(
            f"{text.capitalize()} the {c2} {s2} is the {c1} {s1}"
        )
    return out, scenes_template


def build_t5_embeddings(prompts, tokenizer, encoder, device, max_len=120):
    """Encode a list of prompts through T5. Returns dict mapping prompt -> dict
    with 'caption_embeds' and 'emb_mask'."""
    cache = {}
    with torch.no_grad():
        for p in prompts:
            toks = tokenizer(p, max_length=max_len, padding="max_length",
                             truncation=True, return_tensors="pt").to(device)
            out = encoder(toks.input_ids, attention_mask=toks.attention_mask)
            cache[f"base::{p}"] = {
                "caption_embeds": out.last_hidden_state.to(torch.float32).cpu(),
                "emb_mask": toks.attention_mask.cpu(),
            }
    return cache


def main():
    t0 = time.time()
    transformer, pipe, tokenizer, device, weight_dtype = load_model_and_pipeline(
        "objrel_T5_DiT_mini_pilot", 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)

    variants, scenes = build_variant_prompts()
    all_new = list(set(itertools.chain(*variants.values())))
    print(f"[prompts] {len(scenes)} scenes; {len(all_new)} unique prompt strings to embed")

    # Need to load T5 to embed new prompts (not in original cache)
    from transformers import T5EncoderModel
    print("[t5] loading T5 encoder for new prompt embeddings ...")
    encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl",
                                              torch_dtype=torch.bfloat16).to(device).eval()
    cache.update(build_t5_embeddings(all_new, tokenizer, encoder, device))
    del encoder; gc.collect(); torch.cuda.empty_cache()
    print("[t5] cache extended")

    rows = []
    for variant_name, prompts in variants.items():
        for cond_label, head_set in [
            ("baseline", []),
            ("L0H0_ablated", [(0, 0)]),
        ]:
            t1 = time.time()
            orig = apply_zero_head_ablation_multi(transformer, head_set) if head_set else None
            try:
                eval_df, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                    pipe, prompts, scenes, cache,
                    num_images=10, num_inference_steps=14, guidance_scale=4.5,
                    generator_seed=42, device=device, weight_dtype=weight_dtype,
                    show_prompt_progress=False,
                )
            finally:
                if orig is not None:
                    restore_processors(transformer, orig)
            m = {k: float(eval_df[k].mean()) for k in
                 ["spatial_relationship_loose", "color", "shape",
                  "unique_binding"] if k in eval_df.columns}
            rows.append({"variant": variant_name, "condition": cond_label, **m,
                         "elapsed_s": float(time.time() - t1)})
            print(f"[{variant_name}] {cond_label:>13s}  loose={m['spatial_relationship_loose']:.3f}  "
                  f"color={m['color']:.3f}  shape={m['shape']:.3f}  ({time.time()-t1:.1f}s)",
                  flush=True)
            gc.collect(); torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out_path = join(PROJECT_ROOT, "paper_original_completion", "data",
                    "ood_generalization.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
