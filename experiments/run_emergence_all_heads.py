"""Phase A2 — emergence of spatial-relation alignment for ALL 36 heads
across 11 training checkpoints.

For each checkpoint:
  1. Load EMA state into transformer.
  2. Project all cached T5 embeddings through caption_projection.
  3. Variance-partition projected embeddings to extract effect_vecs (per
     spatial relation). [Note: these depend on the checkpoint because
     caption_projection changes.]
  4. For each (layer, head): compute |cosine|, |projection|, energy of
     OV-write to the 8x8 spatial ramp.

Output: results/emergence_all_heads.csv
        with rows (epoch, layer, head, abs_cosine, projection, energy).
"""
import os, sys, gc, time
from os.path import join

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))

CHECKPOINTS = [
    (100, 4000), (250, 10000), (500, 20000), (600, 24000),
    (700, 28000), (750, 30000), (800, 32000), (900, 36000),
    (1000, 40000), (2000, 80000), (4000, 160000),
]

VP_FEATURES = ["color1", "shape1", "color2", "shape2", "spatial_relationship"]


def main():
    from diffusion.utils.misc import read_config
    from utils.pixart_utils import (
        construct_diffuser_transformer_from_config, load_pixart_ema_into_transformer,
    )
    from utils.notebook_setup import (
        load_embedding_cache, generate_prompts_and_scene_info,
        compute_head_alignment, transformer_hidden_size,
    )
    from utils.variance_partition_with_effects import variance_partition_with_effects

    cfg = read_config(join(PROJECT_ROOT, "results", "objrel_T5_DiT_mini_pilot", "config.py"))
    transformer = construct_diffuser_transformer_from_config(cfg).cpu().eval()
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)
    prompts_all, scenes_all, df_all = generate_prompts_and_scene_info()

    rows_out = []
    for epoch, step in CHECKPOINTS:
        ckpt_path = join(PROJECT_ROOT, "results", "objrel_T5_DiT_mini_pilot",
                         "checkpoints", f"epoch_{epoch}_step_{step}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[skip] missing {ckpt_path}"); continue
        t0 = time.time()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_pixart_ema_into_transformer(transformer, ckpt["state_dict_ema"])
        del ckpt; gc.collect()
        cap_proj = transformer.caption_projection.cpu().float()
        for p in cap_proj.parameters(): p.requires_grad_(False)

        # Project the last-non-pad caption token of every prompt into hidden-dim
        # (we use last-non-pad as the canonical "shape2" representative; this is
        # what the original NB uses for variance partition).
        wordvec = []
        keep_idx = []
        for i, prompt in enumerate(prompts_all):
            match = next((k for k in cache if k != "" and k.endswith(f"::{prompt}")), None)
            if match is None: continue
            e = cache[match]["caption_embeds"]
            if e.ndim == 3: e = e[0]
            mask = cache[match]["emb_mask"]
            if mask.ndim == 2: mask = mask[0]
            last = int(mask.sum().item()) - 1
            with torch.no_grad():
                proj = cap_proj(e.float())
            wordvec.append(proj[max(last - 1, 0)].detach().numpy())
            keep_idx.append(i)
        wordvec_proj = np.stack(wordvec, axis=0).astype(np.float32)
        df_use = df_all.iloc[keep_idx].reset_index(drop=True)

        align_df, effect_vecs, levels_map, r2_total, pos_2d, ramp_templates = \
            compute_head_alignment(
                transformer, wordvec_proj, df_use, VP_FEATURES,
                n_perm=0, base_size=8, verbose=False, random_state=43,
            )
        align_df["epoch"] = epoch
        align_df["abs_cosine"] = align_df["cosine"].abs()
        align_df["abs_projection"] = align_df["projection"].abs()
        for _, r in align_df.iterrows():
            rows_out.append(dict(
                epoch=epoch, step=step,
                layer=int(r["layer"]), head=int(r["head"]),
                abs_cosine=float(r["abs_cosine"]),
                abs_projection=float(r["abs_projection"]),
                projection=float(r["projection"]),
                energy=float(r["energy"]),
                r2_total=float(r2_total),
            ))
        print(f"[{epoch:>4d}] r2_total={r2_total:.3f}   "
              f"L0H0 |cos|={align_df.query('layer==0 and head==0').iloc[0]['abs_cosine']:.3f}   "
              f"L1H2 |cos|={align_df.query('layer==1 and head==2').iloc[0]['abs_cosine']:.3f}   "
              f"({time.time()-t0:.1f}s)", flush=True)

    out_df = pd.DataFrame(rows_out)
    out_path = join(PROJECT_ROOT, "results",
                    "emergence_all_heads.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n[save] -> {out_path}")
    print(f"   rows: {len(out_df)}")


if __name__ == "__main__":
    main()
