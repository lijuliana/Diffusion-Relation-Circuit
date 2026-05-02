"""Path patching for downstream-head discovery.

Procedure:
  1. Capture L0H0's clean activations on the source prompt.
  2. Run the corrupt prompt with:
       - L0H0 patched to clean (gives ~83% recovery)
       - AND simultaneously zero h's output (block h's read)
     for each candidate head h in {layer 1-5} x {head 0-5} (30 heads).
  3. Heads where the additional zero kills the recovery are downstream
     readers carrying L0H0's signal; heads with no extra effect are
     not on the path.

Per-pair patch + 30 candidates + 250 steps + 10 imgs ≈ 30 candidates *
30 s = 15 min per pair. Use 4 relation-flip pairs to average.

Output:
  results/path_patching_discovery.csv
"""
from __future__ import annotations
import argparse, gc, os, sys, time
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings


RUN_NAME = "objrel_T5_DiT_mini_pilot"
SRC_LAYER, SRC_HEAD = 0, 0   # L0H0 — the source we patch from clean


def _attn_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb):
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    bs4 = ch = h = w = None
    if input_ndim == 4:
        bs4, ch, h, w = hidden_states.shape
        hidden_states = hidden_states.view(bs4, ch, h * w).transpose(1, 2)
    bs2, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                      else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs2)
        attention_mask = attention_mask.view(bs2, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and attn.norm_cross:
        enc = attn.norm_encoder_hidden_states(enc)
    key = attn.to_k(enc); value = attn.to_v(enc)
    inner_dim = key.shape[-1]; head_dim = inner_dim // attn.heads
    query = query.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    key   = key.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None: query = attn.norm_q(query)
    if attn.norm_k is not None: key   = attn.norm_k(key)
    sc = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * sc
    if attention_mask is not None: scores = scores + attention_mask
    probs = F.softmax(scores, dim=-1)
    head_out = torch.matmul(probs, value)
    return dict(head_out=head_out, residual=residual, attn=attn,
                bs4=bs4, ch=ch, h=h, w=w, input_ndim=input_ndim,
                inner_dim=inner_dim, head_dim=head_dim)


def _attn_finish(state, head_out_override=None):
    attn = state["attn"]
    head_out = head_out_override if head_out_override is not None else state["head_out"]
    bs2 = head_out.shape[0]; inner_dim = state["inner_dim"]
    hs = head_out.transpose(1, 2).reshape(bs2, -1, inner_dim)
    hs = attn.to_out[0](hs); hs = attn.to_out[1](hs)
    if state["input_ndim"] == 4:
        bs4, ch, h, w = state["bs4"], state["ch"], state["h"], state["w"]
        hs = hs.transpose(-1, -2).reshape(bs4, ch, h, w)
    if getattr(attn, "residual_connection", False): hs = hs + state["residual"]
    hs = hs / getattr(attn, "rescale_output_factor", 1.0)
    return hs


class CapturePerStep:
    def __init__(self, original, layer_idx, target_layer):
        self.original = original; self.layer_idx = layer_idx; self.target_layer = target_layer
        self.captures = []
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx != self.target_layer:
            return self.original(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask, temb=temb, *a, **kw)
        st = _attn_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        self.captures.append(st["head_out"].detach().cpu())
        return _attn_finish(st)


class PatchAndZeroPerStep:
    """At src_layer, patch src_head from cached. At zero_layer, zero zero_head's output."""
    def __init__(self, original, layer_idx, src_layer, src_head, src_captures,
                 zero_layer=None, zero_head=None):
        self.original = original; self.layer_idx = layer_idx
        self.src_layer = src_layer; self.src_head = src_head; self.src_captures = src_captures
        self.zero_layer = zero_layer; self.zero_head = zero_head
        self.step_idx_src = 0
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **kw):
        st = _attn_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        head_out = st["head_out"]
        modified = False
        if self.layer_idx == self.src_layer and self.step_idx_src < len(self.src_captures):
            cached = self.src_captures[self.step_idx_src].to(device=head_out.device, dtype=head_out.dtype)
            head_out = head_out.clone()
            head_out[:, self.src_head] = cached[:, self.src_head]
            self.step_idx_src += 1
            modified = True
        if self.zero_layer is not None and self.layer_idx == self.zero_layer:
            if not modified:
                head_out = head_out.clone()
            head_out[:, self.zero_head] = 0.0
        return _attn_finish(st, head_out_override=head_out)


def install_patch_zero(transformer, src_layer, src_head, src_captures,
                       zero_layer=None, zero_head=None):
    orig = {}
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if not hasattr(blk, "attn2"): continue
        orig[li] = blk.attn2.processor
        # Need our processor on every layer that participates (src_layer always; zero_layer if set)
        if li == src_layer or (zero_layer is not None and li == zero_layer):
            blk.attn2.processor = PatchAndZeroPerStep(
                blk.attn2.processor, li, src_layer, src_head, src_captures,
                zero_layer=zero_layer, zero_head=zero_head,
            )
    return orig


def install_capture(transformer, target_layer):
    orig = {}; new_proc = None
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if not hasattr(blk, "attn2"): continue
        orig[li] = blk.attn2.processor
        if li == target_layer:
            new_proc = CapturePerStep(blk.attn2.processor, li, target_layer)
            blk.attn2.processor = new_proc
    return orig, new_proc


def restore(transformer, orig):
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2") and li in orig:
            blk.attn2.processor = orig[li]


PAIRS = [
    ("blue circle is above red square",                       "blue circle is below red square",                "above", "below"),
    ("blue circle is to the upper right of red square",       "blue circle is to the lower left of red square", "upper_right", "lower_left"),
    ("blue circle is to the lower left of red square",        "blue circle is to the upper right of red square","lower_left",  "upper_right"),
    ("blue circle is below red square",                       "blue circle is above red square",                "below",       "above"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--num_inference_steps", type=int, default=250)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--generator_seed", type=int, default=42)
    p.add_argument("--out_csv", default=join(PROJECT_ROOT, "results", "path_patching_discovery.csv"))
    return p.parse_args()


def gen_eval(pipe, prompt, scene, cache, args, device, weight_dtype):
    return evaluate_pipeline_on_prompts_with_cached_embeddings(
        pipe, [prompt], [scene], cache,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator_seed=args.generator_seed,
        device=device, weight_dtype=weight_dtype,
        show_prompt_progress=False,
    )[0]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    t0 = time.time()
    transformer, pipe, _, device, weight_dtype = load_model_and_pipeline(
        RUN_NAME, 4000, 160000, project_root=PROJECT_ROOT,
    )
    pipe.set_progress_bar_config(disable=True)
    cache = load_embedding_cache(RUN_NAME, project_root=PROJECT_ROOT)

    n_layers = len(transformer.transformer_blocks)
    n_heads  = transformer.config.num_attention_heads

    rows = []
    for clean_prompt, corrupt_prompt, clean_rel, corrupt_rel in PAIRS:
        clean_scene = {"color1":"blue","shape1":"circle","color2":"red","shape2":"square","spatial_relationship":clean_rel}
        # 1. Capture L0H0 clean
        orig, cap = install_capture(transformer, SRC_LAYER)
        try:
            gen_eval(pipe, clean_prompt, clean_scene, cache, args, device, weight_dtype)
            captures_l0h0 = list(cap.captures)
        finally:
            restore(transformer, orig)

        # 2. Reference: L0H0 patched (no zero) — gives ~83% recovery
        orig = install_patch_zero(transformer, SRC_LAYER, SRC_HEAD, captures_l0h0)
        try:
            eval_p = gen_eval(pipe, corrupt_prompt, clean_scene, cache, args, device, weight_dtype)
        finally:
            restore(transformer, orig)
        recov_only_patch = float(eval_p["spatial_relationship"].mean())

        # 3. Corrupt baseline (no patch, no zero) — accuracy on clean scene
        eval_c = gen_eval(pipe, corrupt_prompt, clean_scene, cache, args, device, weight_dtype)
        corrupt_at_clean = float(eval_c["spatial_relationship"].mean())

        # 4. Clean reference accuracy on clean scene
        eval_cl = gen_eval(pipe, clean_prompt, clean_scene, cache, args, device, weight_dtype)
        clean_at_clean = float(eval_cl["spatial_relationship"].mean())

        # 5. For each candidate (layer >= 1), additionally zero h while patching L0H0
        for cand_layer in range(1, n_layers):
            for cand_head in range(n_heads):
                t1 = time.time()
                orig = install_patch_zero(transformer, SRC_LAYER, SRC_HEAD, captures_l0h0,
                                          zero_layer=cand_layer, zero_head=cand_head)
                try:
                    eval_pz = gen_eval(pipe, corrupt_prompt, clean_scene, cache, args, device, weight_dtype)
                finally:
                    restore(transformer, orig)
                acc_pz = float(eval_pz["spatial_relationship"].mean())
                # How much did zeroing this candidate kill the L0H0-patch recovery?
                recovery_loss = recov_only_patch - acc_pz
                rows.append(dict(
                    clean_prompt=clean_prompt, corrupt_prompt=corrupt_prompt,
                    clean_rel=clean_rel, corrupt_rel=corrupt_rel,
                    candidate_layer=cand_layer, candidate_head=cand_head,
                    candidate_label=f"L{cand_layer}H{cand_head}",
                    clean_at_clean=clean_at_clean,
                    corrupt_at_clean=corrupt_at_clean,
                    patched_at_clean=recov_only_patch,
                    patch_plus_zero_at_clean=acc_pz,
                    recovery_loss=recovery_loss,
                ))
                print(f"  pair={clean_rel}/{corrupt_rel}  L{cand_layer}H{cand_head}: "
                      f"patch={recov_only_patch:.2f} +zero={acc_pz:.2f}  loss={recovery_loss:+.2f}  "
                      f"({time.time()-t1:.1f}s)", flush=True)
                gc.collect(); torch.cuda.empty_cache()
        del captures_l0h0
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[save] -> {args.out_csv}", flush=True)
    print("\n=== Top-15 candidates by mean recovery loss across pairs ===")
    summary = df.groupby(["candidate_layer","candidate_head","candidate_label"])["recovery_loss"].mean().reset_index()
    print(summary.nlargest(15, "recovery_loss").to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\n[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
