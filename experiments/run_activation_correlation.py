"""Phase B2 — Activation correlation for downstream-head discovery.

For 30 randomly-sampled prompts (varied colors / shapes / spatial relations),
capture every cross-attention head_out at every diffusion timestep. Compute
Pearson correlation between L0H0's head_out and each other head's head_out:

  corr_h = Pearson(L0H0.head_out.flatten over (pos, head_dim, timestep),
                   h.head_out.flatten over (pos, head_dim, timestep))

Average across prompts. Heads that read from L0H0's per-position direction
will have high in-vivo activation correlation. Output: candidate ranking.

This is deterministic — no eval-noise floor. Complement to path-patching.
"""
import os, sys, time, random, json
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))


def main():
    from diffusion.utils.misc import read_config
    from utils.pixart_utils import (
        construct_diffuser_transformer_from_config, load_pixart_ema_into_transformer,
    )
    from utils.notebook_setup import (
        load_model_and_pipeline, generate_prompts_and_scene_info, load_embedding_cache,
        _forward_attn_capture,
    )

    transformer, pipeline, tokenizer, device, compute_dtype = load_model_and_pipeline(
        run_name="objrel_T5_DiT_mini_pilot",
        ckpt_epoch=4000, ckpt_step=160000,
        project_root=PROJECT_ROOT,
    )
    pipeline.transformer.eval()

    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)
    prompts_all, scenes_all, df_all = generate_prompts_and_scene_info()

    # Pick prompts spanning all 8 relations and varied attributes
    rng = random.Random(43)
    by_rel = {}
    for p, s in zip(prompts_all, scenes_all):
        by_rel.setdefault(s["spatial_relationship"], []).append(p)
    prompts = []
    for r in sorted(by_rel.keys()):
        ps = by_rel[r]
        rng.shuffle(ps)
        prompts.extend(ps[:4])  # 4 per relation x 8 = 32
    prompts = prompts[:30]
    print(f"Sampled {len(prompts)} prompts across {len(by_rel)} relations.")

    # Custom accumulating processor: appends head_out to a list each call (each diffusion step).
    class AccumulateCaptureProcessor:
        def __init__(self, original, layer_idx):
            self.original = original
            self.layer_idx = layer_idx
            self.head_outs = []  # list of (B, H, P, D) cpu tensors
        def reset(self):
            self.head_outs = []
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, *a, **kw):
            storage = {}
            out = _forward_attn_capture(attn, hidden_states, encoder_hidden_states,
                                        attention_mask, temb, storage)
            self.head_outs.append(storage["head_out"])
            return out

    n_layers = sum(1 for _ in pipeline.transformer.transformer_blocks.named_children())
    capture_storages = {}
    orig_procs = {}
    for name, blk in pipeline.transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2"):
            orig_procs[li] = blk.attn2.processor
            cap = AccumulateCaptureProcessor(blk.attn2.processor, li)
            blk.attn2.processor = cap
            capture_storages[li] = cap

    n_heads = pipeline.transformer.config.num_attention_heads
    head_dim = pipeline.transformer.config.attention_head_dim

    # We accumulate per-prompt (n_layers, n_heads) Pearson correlations vs L0H0
    per_prompt_corrs = []
    per_prompt_l0h0_norms = []
    # For Phase B3: per-prompt time-averaged head_out norm-per-position pattern
    # shape (n_prompts, n_layers, n_heads, n_patches)
    per_prompt_pos_pattern = []
    per_prompt_relations = []
    NUM_STEPS = 14  # fewer steps to keep runtime ~ 1 minute

    for prompt_idx, prompt in enumerate(prompts):
        # find embedding
        match_key = next((k for k in cache if k != "" and k.endswith(f"::{prompt}")), None)
        if match_key is None:
            print(f"[skip] no cache for {prompt[:40]}"); continue

        emb = cache[match_key]["caption_embeds"]
        if emb.ndim == 2: emb = emb.unsqueeze(0)
        mask = cache[match_key]["emb_mask"]
        if mask.ndim == 1: mask = mask.unsqueeze(0)

        # Reset captures
        for li in range(n_layers):
            capture_storages[li].reset()

        gen = torch.Generator(device=device).manual_seed(43 + prompt_idx)
        t0 = time.time()
        with torch.no_grad():
            out = pipeline(
                prompt_embeds=emb.to(device).float().to(pipeline.transformer.dtype),
                prompt_attention_mask=mask.to(device),
                num_inference_steps=NUM_STEPS,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                generator=gen,
                output_type="pt",
            )
        elapsed = time.time() - t0

        try:
            stacked_per_layer = []
            for li in range(n_layers):
                if not capture_storages[li].head_outs:
                    raise RuntimeError(f"layer {li} captured no head_outs")
                stk = torch.stack(capture_storages[li].head_outs, dim=0)  # (T, B, H, P, D)
                # Use only conditional half of CFG (index 1) to remove unconditional contribution
                stk = stk[:, 1:2]
                # (T, 1, H, P, D) -> (H, T*1*P*D)
                stk = stk.permute(2, 0, 1, 3, 4).contiguous().view(stk.shape[2], -1)
                stacked_per_layer.append(stk)
            stk_all = torch.stack(stacked_per_layer, dim=0)  # (L, H, F)
        except Exception as e:
            print(f"[stack error] {e}"); continue

        # Compute Pearson correlation between L0H0's vector and every (l, h) vector
        l0h0 = stk_all[0, 0].float()
        l0h0 = l0h0 - l0h0.mean()
        l0h0n = l0h0 / (l0h0.norm() + 1e-9)

        corr_grid = torch.zeros(n_layers, n_heads)
        for li in range(n_layers):
            for hi in range(n_heads):
                v = stk_all[li, hi].float()
                v = v - v.mean()
                vn = v / (v.norm() + 1e-9)
                corr_grid[li, hi] = (l0h0n * vn).sum()

        per_prompt_corrs.append(corr_grid.numpy())
        per_prompt_l0h0_norms.append(float(stk_all[0, 0].float().norm()))

        # Phase B3 input: per-position write magnitude (averaged over time and head_dim).
        # We use the same captured tensor stacks but reshape to keep position axis.
        pos_pattern = np.zeros((n_layers, n_heads, 64), dtype=np.float32)
        for li in range(n_layers):
            stk = torch.stack(capture_storages[li].head_outs, dim=0)[:, 1:2]  # (T,1,H,P,D)
            stk = stk.float().squeeze(1)  # (T,H,P,D)
            # mean over time, l2-norm over head_dim
            tmean = stk.mean(dim=0)  # (H, P, D)
            pos_pattern[li] = tmean.norm(dim=-1).numpy()  # (H, P)
        per_prompt_pos_pattern.append(pos_pattern)

        # store the relation label for B3
        prompt_rel = next(s["spatial_relationship"] for s, p in zip(scenes_all, prompts_all) if p == prompt)
        per_prompt_relations.append(prompt_rel)
        print(f"[{prompt_idx+1:2d}/{len(prompts)}] "
              f"{prompt[:60]:60s}  {elapsed:.1f}s  l0h0_norm={per_prompt_l0h0_norms[-1]:.2f}",
              flush=True)

    # Restore processors
    for name, blk in pipeline.transformer.transformer_blocks.named_children():
        li = int(name)
        if hasattr(blk, "attn2") and li in orig_procs:
            blk.attn2.processor = orig_procs[li]

    # Aggregate across prompts
    corr_arr = np.stack(per_prompt_corrs, axis=0)  # (n_prompts, L, H)
    mean_corr = corr_arr.mean(axis=0)
    std_corr = corr_arr.std(axis=0)
    abs_mean_corr = np.abs(mean_corr)

    rows = []
    for li in range(n_layers):
        for hi in range(n_heads):
            rows.append(dict(
                layer=li, head=hi, label=f"L{li}H{hi}",
                mean_corr=float(mean_corr[li, hi]),
                abs_mean_corr=float(abs_mean_corr[li, hi]),
                std_corr=float(std_corr[li, hi]),
                n_prompts=corr_arr.shape[0],
            ))
    out_df = pd.DataFrame(rows)
    # Exclude L0H0 itself when ranking downstream candidates
    candidates = out_df[out_df["label"] != "L0H0"].copy()
    candidates = candidates.sort_values("abs_mean_corr", ascending=False)
    print()
    print("Top 15 candidates (by |mean Pearson with L0H0|):")
    print(candidates.head(15).to_string(index=False))

    out_path = join(PROJECT_ROOT, "paper_original_completion", "data",
                    "activation_correlation.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path}")

    # Save the per-position pattern tensor for Phase B3.
    pos_arr = np.stack(per_prompt_pos_pattern, axis=0)  # (P_prompts, L, H, 64)
    np.savez_compressed(
        join(PROJECT_ROOT, "paper_original_completion", "data",
             "head_out_pos_pattern.npz"),
        pos_pattern=pos_arr,
        relations=np.array(per_prompt_relations),
    )
    print(f"[save] -> head_out_pos_pattern.npz  shape={pos_arr.shape}")


if __name__ == "__main__":
    main()
