"""Regenerate the cross-attention alignment heatmap and per-relation inner-product
maps using the same computation as the original notebook (`split_workflows_v2/
01_setup_and_head_discovery.ipynb`).

The previous standalone scripts estimated relation effect vectors as
``mean(projected[shape2-token of prompts of relation r]) - global_mean``,
which is a noisy group-mean confounded with shape/colour identity. The
notebook instead uses a regression-based variance partition that partials out
shape and colour confounds, yielding much cleaner relation directions.

This script:
  1. Loads the PixArt-mini transformer at the final checkpoint.
  2. Builds the full ~960-prompt set, extracts the shape2-token T5 embedding
     for each, and projects through caption_projection.
  3. Runs `variance_partition_with_effects` with factors
     {spatial_relationship, shape1, color2shape2} -> clean `effect_vecs`.
  4. For each (layer, head), computes the QK score
     `pos_embed · W_q^T · W_k · e_r` per relation, and aggregates
     cosine / projection / energy via mean across relations.
  5. Writes `experiments/verify_top_heads_align.csv` (per-head, signed,
     averaged across relations).
  6. Generates the L0H0 / L1H2 8-panel ramp figures.

Run from the repo root:
    python experiments/regen_alignment_clean.py
"""
import os, sys, gc
from os.path import join

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))


def find_shape_index(tokens, shape):
    s = shape.strip().lower()
    for i, t in enumerate(tokens):
        tc = t.strip().lower()
        if tc == s or tc == f"▁{s}":
            return i
    for i, t in enumerate(tokens):
        if s in t.strip().lower():
            return i
    return None


def ramp_alignment_metrics(M, dvec, eps=1e-12):
    """Returns dict(cosine, projection, energy) — signed cosine and projection."""
    H, W = M.shape
    d = np.asarray(dvec, dtype=np.float64)
    d = d / np.linalg.norm(d)
    xs = np.linspace(-1, 1, W); ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(xs, ys)
    T = d[0] * X + d[1] * Y
    T = T - T.mean()
    T_norm = np.linalg.norm(T) + eps
    A = M - M.mean()
    A_norm = np.linalg.norm(A) + eps
    proj = float(np.sum(A * T) / T_norm)
    cos = proj / A_norm
    return dict(cosine=cos, projection=proj, energy=A_norm)


def main():
    from diffusion.utils.misc import read_config
    from utils.pixart_utils import (
        construct_diffuser_transformer_from_config, load_pixart_ema_into_transformer,
    )
    from utils.notebook_setup import (
        get_head_weights, transformer_hidden_size, generate_prompts_and_scene_info,
        load_embedding_cache,
    )
    from utils.pixart_pos_embed import get_2d_sincos_pos_embed
    from utils.variance_partition_with_effects import variance_partition_with_effects
    from transformers import T5Tokenizer

    print("Loading config + transformer...")
    cfg = read_config(join(PROJECT_ROOT, "checkpoints", "config.py"))
    transformer = construct_diffuser_transformer_from_config(cfg)
    ckpt = torch.load(
        join(PROJECT_ROOT, "checkpoints", "epoch_4000_step_160000.pth"),
        map_location="cpu", weights_only=False,
    )
    load_pixart_ema_into_transformer(transformer, ckpt["state_dict_ema"])
    del ckpt; gc.collect()
    transformer = transformer.float().eval()

    cap_proj = transformer.caption_projection
    for p in cap_proj.parameters(): p.requires_grad_(False)

    print("Loading T5 embedding cache...")
    cache_path = join(PROJECT_ROOT, "t5_embedding_cache.pt")
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = raw["embedding_allrel_allobj"] if "embedding_allrel_allobj" in raw else raw
    del raw; gc.collect()
    print(f"  loaded {len(cache)} cached embeddings")

    print("Building prompts + extracting shape2-token vectors...")
    prompts, _, df = generate_prompts_and_scene_info()
    df = df.copy(); df["prompt"] = prompts

    tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/PixArt-XL-2-256x256",
                                            subfolder="tokenizer", legacy=False)

    vec_obj2 = []
    keep_idx = []
    for i, r in df.iterrows():
        key = None
        for k in cache:
            if k.endswith(f"::{r['prompt']}"):
                key = k; break
        if key is None: continue
        e = cache[key]["caption_embeds"]
        if e.ndim == 3: e = e[0]
        tok = tokenizer(r['prompt'], max_length=20, padding='max_length',
                        truncation=True, return_tensors='pt')
        toks = [tokenizer.decode([t]) for t in tok['input_ids'][0]]
        idx2 = find_shape_index(toks, r['shape2'])
        if idx2 is None: continue
        with torch.no_grad():
            proj = cap_proj(e.float())
        vec_obj2.append(proj[idx2].detach().cpu().numpy())
        keep_idx.append(i)
    Y = np.stack(vec_obj2)
    df_keep = df.loc[keep_idx].reset_index(drop=True)
    print(f"  Y shape = {Y.shape} ({len(df_keep)} usable prompts)")

    print("Running variance_partition_with_effects...")
    _, _, effect_vecs, levels_map, R2 = variance_partition_with_effects(
        Y,
        {"spatial_relationship": df_keep["spatial_relationship"],
         "shape1":               df_keep["shape1"],
         "color2shape2":         df_keep["color2shape2"]},
        n_perm=0, verbose=False, random_state=0,
    )
    rel_vecs_arr = effect_vecs["spatial_relationship"]   # (n_rel, hidden)
    rel_levels   = list(levels_map["spatial_relationship"])
    print(f"  R2_total = {R2:.3f},  n_relations = {len(rel_levels)}")

    base_size = 8
    hidden = transformer_hidden_size(transformer)
    pos_2d = torch.tensor(get_2d_sincos_pos_embed(hidden, base_size), dtype=torch.float32)

    direction_vector = {
        'above': (0, -1), 'below': (0, 1), 'left': (-1, 0), 'right': (1, 0),
        'left_of': (-1, 0), 'right_of': (1, 0),
        'upper_left': (-1, -1), 'upper_right': (1, -1),
        'lower_left': (-1, 1),  'lower_right': (1, 1),
    }

    cfg_ = cfg
    n_layers = int(cfg_.depth) if hasattr(cfg_, 'depth') else 6
    n_heads  = int(cfg_.num_heads) if hasattr(cfg_, 'num_heads') else 6
    head_dim = hidden // n_heads
    print(f"  model: L={n_layers}, H={n_heads}, hidden={hidden}, head_dim={head_dim}")

    # Per-(layer, head, relation) inner-product matrices and metrics
    align_rows = []
    per_head_qk = {}   # (layer, head) -> dict[relation] -> 8x8 score
    for layer in range(n_layers):
        for head in range(n_heads):
            W_q, W_k, _, _ = get_head_weights(transformer, layer, head)
            W_q = W_q.float(); W_k = W_k.float()
            qk_per_rel = {}
            for ri, rel in enumerate(rel_levels):
                ev = torch.from_numpy(rel_vecs_arr[ri]).float()
                k_proj = W_k @ ev
                qk = pos_2d @ W_q.T @ k_proj
                qk = qk - qk.mean()
                qk_2d = qk.reshape(base_size, base_size).numpy()
                qk_per_rel[rel] = qk_2d
                if rel in direction_vector:
                    m = ramp_alignment_metrics(qk_2d, direction_vector[rel])
                    align_rows.append(dict(layer=layer, head=head, relation=rel,
                                            label=f"L{layer}H{head}", **m))
            per_head_qk[(layer, head)] = qk_per_rel

    align_df_long = pd.DataFrame(align_rows)
    # Aggregate across relations: mean of signed cosine/projection, max of energy
    syn = (align_df_long.groupby(["layer", "head", "label"], as_index=False)
                        .agg(cosine=("cosine", "mean"),
                             projection=("projection", "mean"),
                             energy=("energy", "mean")))
    syn["abs_cosine"] = syn["cosine"].abs()
    syn = syn[["layer", "head", "cosine", "projection", "energy", "abs_cosine", "label"]]
    out_csv = join(PROJECT_ROOT, "experiments", "verify_top_heads_align.csv")
    syn.to_csv(out_csv, index=False)
    print(f"[csv] wrote {len(syn)} rows -> {out_csv}")
    print(syn.sort_values('projection', key=abs, ascending=False).head(8).to_string(index=False))

    # ------------------------ figures: ramp maps ------------------------
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    import matplotlib.pyplot as plt

    figdir = join(PROJECT_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    rels_order = ["above", "below", "left", "right",
                  "upper_left", "upper_right", "lower_left", "lower_right"]
    # Map spatial_relationship levels to plot labels (e.g. left_of -> left)
    rel_alias = {"left_of": "left", "right_of": "right"}

    for layer, head in [(0, 0), (1, 2)]:
        qk_per_rel = per_head_qk[(layer, head)]
        # Build dict keyed by canonical labels
        qk_canon = {}
        for r, m in qk_per_rel.items():
            qk_canon[rel_alias.get(r, r)] = m
        scores = [qk_canon.get(r) for r in rels_order if qk_canon.get(r) is not None]
        if not scores:
            print(f"  [skip] L{layer}H{head}: no relation scores"); continue
        amax = float(np.max(np.abs(np.stack(scores))))
        fig, axes = plt.subplots(2, 4, figsize=(11, 6))
        for i, rel in enumerate(rels_order):
            ax = axes[i // 4][i % 4]
            sc = qk_canon.get(rel)
            if sc is None:
                ax.axis("off"); ax.set_title(rel + " (n/a)", fontsize=10); continue
            im = ax.imshow(sc, cmap="coolwarm", vmin=-amax, vmax=amax, origin="upper")
            ax.set_title(rel, fontsize=11, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(rf"$L_{layer}H_{head}$: image-grid attention score "
                     rf"$\mathrm{{pos\_embed}}\cdot W_Q^\top W_K\cdot e_r$ per relation $r$",
                     fontsize=13, y=1.0)
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.66])
        fig.colorbar(im, cax=cax, label="QK score (mean-centred)")
        fig.tight_layout(rect=[0, 0, 0.91, 0.97])
        for ext in ("png", "pdf"):
            fig.savefig(join(figdir, f"inner_product_map_L{layer}H{head}.{ext}"),
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[fig] inner_product_map_L{layer}H{head}.* -> {figdir}")


if __name__ == "__main__":
    main()
