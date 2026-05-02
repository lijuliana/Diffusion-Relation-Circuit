"""Harder-OOD test: distractor / extra-adjective / syntax-fronted prompts.

T5-XXL doesn't fit in /workspace (44 GB > 18 GB free). Workaround:
download just the .bin to /dev/shm (47 GB tmpfs), load in bfloat16,
encode prompts (~96 prompts), save embeddings to /workspace, delete the
T5 weights from /dev/shm before running diffusion.

Variants:
  vanilla:    standard template
  distract:   filler clause prepended ('In a quiet scene, ...')
  adj:        size adjectives ('the small red square ... the large blue circle')
  syntax:     fronted relation ('Above the blue circle is the red square')

For each variant: baseline + L0H0 ablation, 24 prompts x 10 images.
"""
import os, sys, gc, time, itertools, shutil
from os.path import join

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))

# Use /dev/shm for T5 to avoid /workspace disk-full
os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/dev/shm/hf_cache"

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


def build_variants():
    """3 prompts per relation, 4 variant types."""
    base_combos = []
    rng = np.random.default_rng(43)
    for rel in RELS_TEXT.keys():
        opts = []
        for c1, c2 in itertools.product(COLORS, COLORS):
            if c1 == c2: continue
            for s1, s2 in itertools.product(SHAPES, SHAPES):
                if s1 == s2: continue
                opts.append((c1, s1, c2, s2))
        rng.shuffle(opts)
        for o in opts[:3]:
            base_combos.append((*o, rel))

    variants = {"vanilla": [], "distract": [], "adj": [], "syntax": []}
    scenes = []
    for c1, s1, c2, s2, rel in base_combos:
        text = RELS_TEXT[rel]
        scenes.append(dict(color1=c1, shape1=s1, color2=c2, shape2=s2,
                           spatial_relationship=rel))
        variants["vanilla"].append(f"{c1} {s1} is {text} {c2} {s2}")
        variants["distract"].append(
            f"the {c1} {s1} is {text} the {c2} {s2}, in a quiet scene"
        )
        adj_pair = ("small", "large") if rng.random() < 0.5 else ("large", "small")
        variants["adj"].append(
            f"the {adj_pair[0]} {c1} {s1} is {text} the {adj_pair[1]} {c2} {s2}"
        )
        variants["syntax"].append(
            f"{text.capitalize()} the {c2} {s2} is the {c1} {s1}"
        )
    return variants, scenes


def main():
    t0 = time.time()
    transformer, pipe, tokenizer, device, weight_dtype = load_model_and_pipeline(
        "objrel_T5_DiT_mini_pilot", 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)
    variants, scenes = build_variants()
    all_prompts = sorted(set(itertools.chain(*variants.values())))
    print(f"[setup] {len(scenes)} scenes; {len(all_prompts)} unique prompt strings to embed")

    # ====== Encode new prompts via T5 (use PixArt-alpha's fp16 safetensors copy) ======
    # google/t5-v1_1-xxl only ships pytorch_model.bin (44GB fp32) which transformers
    # refuses to load due to CVE-2025-32434. PixArt-alpha/PixArt-XL-2-512x512 ships
    # the same T5-XXL weights as fp16 safetensors (~22GB, 2 shards).
    print("[t5] downloading T5-XXL fp16 safetensors from PixArt-alpha repo to /dev/shm ...")
    from huggingface_hub import hf_hub_download
    repo = "PixArt-alpha/PixArt-XL-2-512x512"
    local = "/dev/shm/t5_xxl"
    os.makedirs(local, exist_ok=True)
    needed = [
        "text_encoder/config.json",
        "text_encoder/model.fp16-00001-of-00002.safetensors",
        "text_encoder/model.fp16-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.fp16.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
    ]
    for fn in needed:
        hf_hub_download(repo, fn, local_dir=local,
                        local_dir_use_symlinks=False)
    # Reorganise to a single dir t5_xxl/ with model.* files at top level
    import shutil as _sh
    text_dir = join(local, "text_encoder")
    tok_dir = join(local, "tokenizer")
    # Rename the index file so transformers picks it up:
    src_idx = join(text_dir, "model.safetensors.index.fp16.json")
    dst_idx = join(text_dir, "model.safetensors.index.json")
    if os.path.exists(src_idx) and not os.path.exists(dst_idx):
        _sh.copy(src_idx, dst_idx)
    # Rename shards from 'fp16-' prefix to standard naming
    src1 = join(text_dir, "model.fp16-00001-of-00002.safetensors")
    src2 = join(text_dir, "model.fp16-00002-of-00002.safetensors")
    dst1 = join(text_dir, "model-00001-of-00002.safetensors")
    dst2 = join(text_dir, "model-00002-of-00002.safetensors")
    if os.path.exists(src1) and not os.path.exists(dst1):
        os.symlink(src1, dst1)
        os.symlink(src2, dst2)
    # Patch the index json to reference the renamed files
    with open(dst_idx) as f:
        idx = f.read()
    idx = idx.replace("model.fp16-", "model-")
    with open(dst_idx, "w") as f:
        f.write(idx)

    # Tokenizer files in same dir
    for fn in os.listdir(tok_dir):
        if not os.path.exists(join(text_dir, fn)):
            os.symlink(join(tok_dir, fn), join(text_dir, fn))

    print("[t5] loading model in bfloat16 ...")
    from transformers import T5EncoderModel, T5Tokenizer
    encoder = T5EncoderModel.from_pretrained(text_dir, torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(device).eval()
    tok = T5Tokenizer.from_pretrained(text_dir)

    # Use max_length=20 to match the pre-existing cached uncond and other prompts.
    # All variants are short enough to fit; distractor was rewritten to fit.
    MAX_LEN = 20
    print(f"[t5] encoding new prompts at max_length={MAX_LEN} ...")
    with torch.no_grad():
        for p in all_prompts:
            toks = tok(p, max_length=MAX_LEN, padding="max_length", truncation=True,
                       return_tensors="pt").to(device)
            n_tok = int(toks.attention_mask.sum().item())
            if n_tok >= MAX_LEN:
                print(f"  [warn] '{p[:50]}...' fills full window ({n_tok}/{MAX_LEN})")
            out = encoder(toks.input_ids, attention_mask=toks.attention_mask)
            cache[f"base::{p}"] = {
                "caption_embeds": out.last_hidden_state.to(torch.float32).cpu(),
                "emb_mask": toks.attention_mask.cpu(),
            }
    print(f"[t5] cache extended; freeing T5 from RAM and /dev/shm ...")
    del encoder, tok; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(local, ignore_errors=True)
    shutil.rmtree("/dev/shm/hf_cache", ignore_errors=True)
    print("[t5] done")

    # ====== Run baseline + L0H0 ablation per variant ======
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
    out_path = join(PROJECT_ROOT, "results",
                    "ood_harder.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
