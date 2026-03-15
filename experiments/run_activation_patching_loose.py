"""Priority 2: Clean->corrupt activation patching at the rank-1 reader L2H3.

Activation-patching analysis

  For each of ~20 prompts, create a relation-flipped corrupt version
  (above<->below, left<->right, etc.). Run clean forward pass, cache
  L2H3 output activations at every denoising step. Run corrupt forward
  pass; patch in clean L2H3 activations. Measure

      recovery = (patched_acc - corrupt_acc) / (clean_acc - corrupt_acc)

  Repeat for 2 non-reader control heads to form the control band.

Scoring convention: every generated image is scored against the *clean*
scene_info (same colors/shapes; relation = clean's expected). This makes
all three conditions comparable on a single accuracy axis: clean ≈ 1,
corrupt ≈ 0 (because the corrupt prompt tells the model to draw the
flipped relation), patched lies between if the patch redirected the
spatial code. The recovery fraction is then well-defined.

The cross-attention head outputs need to be captured on every denoising
step. ``utils/notebook_setup.py``'s ``AttentionCaptureProcessor`` stores
only the last call (storage["head_out"] is overwritten). Per project
rules we don't modify utils/*, so this script defines its own per-step
capture / patch processors.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from os.path import join
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")

from utils.notebook_setup import load_model_and_pipeline, load_embedding_cache
from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings


RUN_NAME = "objrel_T5_DiT_mini_pilot"

# Reader and control heads.  L2H3 is the rank-1 QK-OV-composition reader
# (paper Section 4.6 / 4.7).  Control heads must NOT be in the source set
# {L0H0, L1H2} or in the top-5 reader set.
TARGET_HEAD: Tuple[int, int] = (1, 2)  # L1H2 (rank-1 consensus)
CONTROL_HEADS: List[Tuple[int, int]] = [(0, 5), (1, 3), (2, 1), (2, 2), (4, 4)]  # B4 consensus top 4 + 1 norm-control

# Phrase-level relation flip rules.  Order matters only for substring
# disambiguation; we apply the longest matching rule per prompt.
FLIP_PHRASES: List[Tuple[str, str]] = [
    # Vertical
    ("is above ", "is below "),
    ("is below ", "is above "),
    # Diagonal compounds
    ("above and to the left of", "below and to the right of"),
    ("above and to the right of", "below and to the left of"),
    ("below and to the left of", "above and to the right of"),
    ("below and to the right of", "above and to the left of"),
    # Horizontal
    ("is to the left of", "is to the right of"),
    ("is to the right of", "is to the left of"),
    # Diagonals (named)
    ("is to the upper left of", "is to the lower right of"),
    ("is to the upper right of", "is to the lower left of"),
    ("is to the lower left of", "is to the upper right of"),
    ("is to the lower right of", "is to the upper left of"),
]

# Inverse semantic mapping for scene_info["spatial_relationship"]
RELATION_FLIP: dict = {
    "above": "below", "below": "above",
    "left": "right", "right": "left",
    "upper_left": "lower_right", "lower_right": "upper_left",
    "upper_right": "lower_left", "lower_left": "upper_right",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--num_inference_steps", type=int, default=14,
                   help="14 for budgeted run; 250 to match P1's high-fidelity setting.")
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--generator_seed", type=int, default=42)
    p.add_argument("--n_pairs", type=int, default=20)
    p.add_argument("--out_csv",
                   default=join(PROJECT_ROOT, "experiments", "activation_patching_results.csv"))
    p.add_argument("--out_per_image_csv",
                   default=join(PROJECT_ROOT, "experiments", "activation_patching_eval_rows.csv"))
    p.add_argument("--smoke", action="store_true",
                   help="2 prompts, target head only, 5 images. Sanity check.")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Per-step capture / patch processors
#  (Live in this script per the project rule about not modifying utils/*.)
#  Mirror utils/notebook_setup.py's _forward_attn_capture / _forward_attn_patched
#  but accumulate / replay across the full denoising trajectory.
# ---------------------------------------------------------------------------

def _attn_forward_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb):
    """Replicate cross-attention forward up to the per-head output
    (post softmax @ V, pre o-projection).  Returns a state dict with
    head_out and the bookkeeping the finisher needs."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    bs4 = ch = h = w = None
    if input_ndim == 4:
        bs4, ch, h, w = hidden_states.shape
        hidden_states = hidden_states.view(bs4, ch, h * w).transpose(1, 2)
    bs2, seq_len, _ = (
        hidden_states.shape if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs2)
        attention_mask = attention_mask.view(bs2, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)
    enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and attn.norm_cross:
        enc = attn.norm_encoder_hidden_states(enc)
    key = attn.to_k(enc)
    value = attn.to_v(enc)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(bs2, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    sc = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * sc
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = F.softmax(scores, dim=-1)
    head_out = torch.matmul(probs, value)
    return dict(
        head_out=head_out, residual=residual, attn=attn,
        bs4=bs4, ch=ch, h=h, w=w, input_ndim=input_ndim,
        inner_dim=inner_dim, head_dim=head_dim,
    )


def _attn_finish(state, head_out_override=None):
    attn = state["attn"]
    head_out = head_out_override if head_out_override is not None else state["head_out"]
    bs2 = head_out.shape[0]
    inner_dim = state["inner_dim"]
    hidden_states = head_out.transpose(1, 2).reshape(bs2, -1, inner_dim)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if state["input_ndim"] == 4:
        bs4, ch, h, w = state["bs4"], state["ch"], state["h"], state["w"]
        hidden_states = hidden_states.transpose(-1, -2).reshape(bs4, ch, h, w)
    if getattr(attn, "residual_connection", False):
        hidden_states = hidden_states + state["residual"]
    hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
    return hidden_states


class PerStepCaptureProcessor:
    def __init__(self, original, layer_idx: int, target_layer: int):
        self.original = original
        self.layer_idx = layer_idx
        self.target_layer = target_layer
        self.captures: list[torch.Tensor] = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx != self.target_layer:
            return self.original(attn, hidden_states,
                                 encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask, temb=temb, *a, **kw)
        st = _attn_forward_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        self.captures.append(st["head_out"].detach().cpu())
        return _attn_finish(st)


class PerStepPatchProcessor:
    def __init__(self, original, layer_idx: int, target_layer: int,
                 target_heads: Sequence[int], captures: Sequence[torch.Tensor]):
        self.original = original
        self.layer_idx = layer_idx
        self.target_layer = target_layer
        self.target_heads = [int(h) for h in target_heads]
        self.captures = list(captures)
        self.step_idx = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *a, **kw):
        if self.layer_idx != self.target_layer:
            return self.original(attn, hidden_states,
                                 encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask, temb=temb, *a, **kw)
        st = _attn_forward_to_head_out(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if self.step_idx >= len(self.captures):
            raise RuntimeError(
                f"PerStepPatchProcessor exhausted: step_idx={self.step_idx} >= "
                f"len(captures)={len(self.captures)}.  Reinstall a fresh processor between runs."
            )
        cached = self.captures[self.step_idx].to(
            device=st["head_out"].device, dtype=st["head_out"].dtype
        )
        head_out = st["head_out"].clone()
        for h in self.target_heads:
            if 0 <= h < attn.heads:
                head_out[:, h] = cached[:, h]
        self.step_idx += 1
        return _attn_finish(st, head_out_override=head_out)


def install_processor_at_layer(transformer, target_layer: int, factory):
    orig = {}
    new_proc = None
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if not hasattr(blk, "attn2"):
            continue
        orig[li] = blk.attn2.processor
        if li == target_layer:
            new_proc = factory(blk.attn2.processor, li, target_layer)
            blk.attn2.processor = new_proc
    return orig, new_proc


def restore_processors(transformer, orig: dict):
    for name, blk in transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2") and li in orig:
            blk.attn2.processor = orig[li]


# ---------------------------------------------------------------------------
#  Prompt / scene-info pairing
# ---------------------------------------------------------------------------

def make_clean_corrupt_pairs(embedding_cache: dict, n_pairs: int = 20):
    """Return (clean_prompt, corrupt_prompt, clean_scene, corrupt_scene) tuples."""
    from utils.ablation_eval_prompts import build_training_template_prompts
    from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES

    prompts, scene_infos = build_training_template_prompts(
        embedding_cache, DEFAULT_SPATIAL_PHRASES,
    )

    pairs = []
    for prompt, scene in zip(prompts, scene_infos):
        rel = scene["spatial_relationship"]
        if rel not in RELATION_FLIP:
            continue
        applicable = [(a, b) for a, b in FLIP_PHRASES if a in prompt]
        if not applicable:
            continue
        a, b = sorted(applicable, key=lambda ab: -len(ab[0]))[0]
        corrupt_prompt = prompt.replace(a, b)
        if not any(k.endswith(f"::{corrupt_prompt}") for k in embedding_cache):
            continue
        corrupt_scene = dict(scene)
        corrupt_scene["spatial_relationship"] = RELATION_FLIP[rel]
        pairs.append((prompt, corrupt_prompt, scene, corrupt_scene))

    if n_pairs is not None and n_pairs > 0:
        pairs = pairs[:n_pairs]
    return pairs


# ---------------------------------------------------------------------------
#  Generation helpers (single prompt, eval against caller-supplied scene)
# ---------------------------------------------------------------------------

def _gen_and_eval(pipe, prompt: str, scene_for_eval: dict, embedding_cache: dict,
                  *, num_images: int, num_inference_steps: int,
                  guidance_scale: float, generator_seed: int,
                  device, weight_dtype):
    """Generate `num_images` images for `prompt` and evaluate them against
    `scene_for_eval`. Returns the eval_df."""
    eval_df, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
        pipe,
        [prompt], [scene_for_eval], embedding_cache,
        num_images=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator_seed=generator_seed,
        device=device, weight_dtype=weight_dtype,
        show_prompt_progress=False,
    )
    return eval_df


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()
    print(f"[config] num_images={args.num_images}  steps={args.num_inference_steps}  "
          f"cfg={args.guidance_scale}  seed={args.generator_seed}  n_pairs={args.n_pairs}")

    transformer, pipe, tokenizer, device, weight_dtype = load_model_and_pipeline(
        RUN_NAME, 4000, 160000, project_root=PROJECT_ROOT,
    )
    cache = load_embedding_cache(RUN_NAME, project_root=PROJECT_ROOT)

    n_pairs = args.n_pairs if not args.smoke else 2
    num_images = args.num_images if not args.smoke else 5
    pairs = make_clean_corrupt_pairs(cache, n_pairs=n_pairs)
    print(f"[pairs] {len(pairs)} clean/corrupt prompt pairs (e.g.):")
    for cp, kp, _, _ in pairs[:3]:
        print(f"   CLEAN:   {cp!r}")
        print(f"   CORRUPT: {kp!r}")

    test_heads = [TARGET_HEAD] + ([] if args.smoke else CONTROL_HEADS)

    summary_rows: list[dict] = []
    eval_rows_all: list[pd.DataFrame] = []

    for layer_idx, head_idx in test_heads:
        is_target = (layer_idx, head_idx) == TARGET_HEAD
        print(f"\n[head] L{layer_idx}H{head_idx}  ({'TARGET' if is_target else 'CONTROL'})")
        per_pair_stats: list[dict] = []

        for pi, (clean_prompt, corrupt_prompt, clean_scene, corrupt_scene) in enumerate(pairs):
            t_pair = time.time()

            # --- 1. Clean run with capture ---
            orig_procs, capture_proc = install_processor_at_layer(
                transformer, layer_idx,
                factory=lambda original, li, tl: PerStepCaptureProcessor(original, li, tl),
            )
            try:
                eval_clean = _gen_and_eval(
                    pipe, clean_prompt, clean_scene, cache,
                    num_images=num_images,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator_seed=args.generator_seed,
                    device=device, weight_dtype=weight_dtype,
                )
                clean_captures = list(capture_proc.captures)
            finally:
                restore_processors(transformer, orig_procs)
            n_steps_seen = len(clean_captures)
            if n_steps_seen == 0:
                print(f"  [{pi+1}/{len(pairs)}] !! no captures on clean run; skipping pair")
                continue

            # --- 2. Corrupt run, baseline (no patch).
            #         Score against the CLEAN scene so accuracies live on a
            #         single axis: clean_acc = upper bound, corrupt_acc =
            #         lower bound, patched_acc = where the head's clean
            #         signal pulls the corrupt run.
            eval_corrupt = _gen_and_eval(
                pipe, corrupt_prompt, clean_scene, cache,
                num_images=num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator_seed=args.generator_seed,
                device=device, weight_dtype=weight_dtype,
            )

            # --- 3. Patched corrupt run.
            orig_procs2, _patch_proc = install_processor_at_layer(
                transformer, layer_idx,
                factory=lambda original, li, tl: PerStepPatchProcessor(
                    original, li, tl, target_heads=[head_idx], captures=clean_captures,
                ),
            )
            try:
                eval_patched = _gen_and_eval(
                    pipe, corrupt_prompt, clean_scene, cache,
                    num_images=num_images,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator_seed=args.generator_seed,
                    device=device, weight_dtype=weight_dtype,
                )
            finally:
                restore_processors(transformer, orig_procs2)

            clean_acc   = float(eval_clean["spatial_relationship_loose"].mean())
            corrupt_acc = float(eval_corrupt["spatial_relationship_loose"].mean())
            patched_acc = float(eval_patched["spatial_relationship_loose"].mean())
            denom = clean_acc - corrupt_acc
            recovery = (patched_acc - corrupt_acc) / denom if abs(denom) > 1e-9 else float("nan")

            per_pair_stats.append(dict(
                layer=layer_idx, head=head_idx,
                head_label=f"L{layer_idx}H{head_idx}",
                is_target=is_target,
                pair_idx=pi,
                clean_prompt=clean_prompt, corrupt_prompt=corrupt_prompt,
                clean_relation=clean_scene["spatial_relationship"],
                corrupt_relation=corrupt_scene["spatial_relationship"],
                n_steps_captured=n_steps_seen,
                clean_acc=clean_acc, corrupt_acc=corrupt_acc, patched_acc=patched_acc,
                recovery_fraction=recovery,
                pair_time_s=float(time.time() - t_pair),
            ))

            for cond_label, df in [("clean", eval_clean), ("corrupt", eval_corrupt), ("patched", eval_patched)]:
                d = df.copy()
                d["layer"] = layer_idx
                d["head"] = head_idx
                d["head_label"] = f"L{layer_idx}H{head_idx}"
                d["pair_idx"] = pi
                d["condition"] = cond_label
                d["clean_prompt"] = clean_prompt
                d["corrupt_prompt"] = corrupt_prompt
                eval_rows_all.append(d)

            print(f"  [{pi+1}/{len(pairs)}] clean={clean_acc:.2f} corrupt={corrupt_acc:.2f} "
                  f"patched={patched_acc:.2f} recovery={recovery:+.2f} "
                  f"({time.time()-t_pair:.1f}s)")

            del clean_captures
            gc.collect(); torch.cuda.empty_cache()

        # Aggregate per-head stats with bootstrap CI on recovery.
        per_pair_df = pd.DataFrame(per_pair_stats)
        if not per_pair_df.empty:
            agg = dict(
                layer=layer_idx, head=head_idx,
                head_label=f"L{layer_idx}H{head_idx}",
                is_target=is_target,
                n_pairs=int(len(per_pair_df)),
                clean_acc_mean=float(per_pair_df["clean_acc"].mean()),
                corrupt_acc_mean=float(per_pair_df["corrupt_acc"].mean()),
                patched_acc_mean=float(per_pair_df["patched_acc"].mean()),
                recovery_fraction_mean=float(per_pair_df["recovery_fraction"].mean(skipna=True)),
                recovery_fraction_std=float(per_pair_df["recovery_fraction"].std(ddof=0, skipna=True)),
            )
            rng = np.random.default_rng(args.generator_seed)
            vals = per_pair_df["recovery_fraction"].dropna().values
            if vals.size > 1:
                samples = rng.choice(vals, size=(1000, vals.size), replace=True).mean(axis=1)
                agg["recovery_fraction_ci_lo"] = float(np.percentile(samples, 2.5))
                agg["recovery_fraction_ci_hi"] = float(np.percentile(samples, 97.5))
            else:
                agg["recovery_fraction_ci_lo"] = float("nan")
                agg["recovery_fraction_ci_hi"] = float("nan")
            summary_rows.append(agg)
            print(f"  -> mean recovery {agg['recovery_fraction_mean']:+.3f}  "
                  f"95%CI[{agg['recovery_fraction_ci_lo']:+.3f}, {agg['recovery_fraction_ci_hi']:+.3f}]  "
                  f"clean={agg['clean_acc_mean']:.3f} corrupt={agg['corrupt_acc_mean']:.3f} "
                  f"patched={agg['patched_acc_mean']:.3f}")

    # ===== Save =====
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_csv, index=False)
    print(f"\n[save] summary -> {args.out_csv} ({len(summary_df)} rows)")

    if eval_rows_all:
        eval_rows_df = pd.concat(eval_rows_all, ignore_index=True)
        eval_rows_df.to_csv(args.out_per_image_csv, index=False)
        print(f"[save] per-image -> {args.out_per_image_csv} ({len(eval_rows_df)} rows)")

    print("\n=== Activation patching recovery (sorted desc) ===")
    if not summary_df.empty:
        cols = ["head_label", "is_target", "clean_acc_mean", "corrupt_acc_mean",
                "patched_acc_mean", "recovery_fraction_mean",
                "recovery_fraction_ci_lo", "recovery_fraction_ci_hi"]
        avail = [c for c in cols if c in summary_df.columns]
        with pd.option_context("display.float_format", "{:+.3f}".format,
                               "display.max_columns", 30, "display.width", 200):
            print(summary_df[avail].sort_values("recovery_fraction_mean", ascending=False).to_string(index=False))

    target_row = summary_df[summary_df["is_target"]] if not summary_df.empty else pd.DataFrame()
    target_recovery = float(target_row["recovery_fraction_mean"].iloc[0]) if not target_row.empty else float("nan")
    with open(join(PROJECT_ROOT, "experiments", "results.tsv"), "a") as f:
        f.write("\t".join([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "activation_patching",
            "success",
            f"recovery_fraction(L{TARGET_HEAD[0]}H{TARGET_HEAD[1]})",
            f"{target_recovery:+.3f}",
            f"controls={CONTROL_HEADS} num_images={num_images} steps={args.num_inference_steps}",
        ]) + "\n")

    print(f"\n[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
