"""For L0H0 and L1H2, produce 8-panel inner-product maps:
each panel is the 8x8 image-grid score of (pos_embed @ W_q^T W_k @ effect_vec)
for one of the 8 relation directions. The 'translator' structure is that the
panel for relation r should look like ramp_r.
"""
import os, sys
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, join(PROJECT_ROOT, "PixArt-alpha"))


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

    cfg = read_config(join(PROJECT_ROOT, "results", "objrel_T5_DiT_mini_pilot", "config.py"))
    transformer = construct_diffuser_transformer_from_config(cfg)
    ckpt = torch.load(
        join(PROJECT_ROOT, "results", "objrel_T5_DiT_mini_pilot", "checkpoints",
             "epoch_4000_step_160000.pth"),
        map_location="cpu", weights_only=False,
    )
    load_pixart_ema_into_transformer(transformer, ckpt["state_dict_ema"])
    del ckpt

    cap_proj = transformer.caption_projection.cpu().float()
    for p in cap_proj.parameters(): p.requires_grad_(False)
    cache = load_embedding_cache("objrel_T5_DiT_mini_pilot", project_root=PROJECT_ROOT)

    # Build relation effect vectors (mean projected shape2-token embedding minus global mean)
    prompts_all, scene_infos_all, df_all = generate_prompts_and_scene_info()
    df_all = df_all.copy(); df_all["prompt"] = prompts_all
    rel_vecs = {}
    for rel, sub in df_all.groupby("spatial_relationship"):
        sub = sub.head(30); vs = []
        for _, r in sub.iterrows():
            for k in cache:
                if k != "" and k.endswith(f"::{r['prompt']}"):
                    e = cache[k]["caption_embeds"]
                    if e.ndim == 3: e = e[0]
                    with torch.no_grad():
                        proj = cap_proj(e.float())
                    mask = cache[k]["emb_mask"]
                    if mask.ndim == 2: mask = mask[0]
                    last = int(mask.sum().item()) - 1
                    vs.append(proj[max(last - 1, 0)].detach())
                    break
        if vs: rel_vecs[rel] = torch.stack(vs).mean(dim=0)
    global_mean = torch.stack(list(rel_vecs.values())).mean(dim=0)
    rel_vecs = {r: (v - global_mean).float() for r, v in rel_vecs.items()}

    base_size = 8
    hidden = transformer_hidden_size(transformer)
    pos_2d = torch.tensor(get_2d_sincos_pos_embed(hidden, base_size), dtype=torch.float32)
    rels_order = ["above", "below", "left", "right", "upper_left", "upper_right", "lower_left", "lower_right"]

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    import matplotlib.pyplot as plt

    out_dir = join(PROJECT_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)

    for layer, head in [(0, 0), (1, 2)]:
        W_q, W_k, _, _ = get_head_weights(transformer, layer, head)
        # 2 rows x 4 cols
        fig, axes = plt.subplots(2, 4, figsize=(11, 6))
        # Compute global vmin/vmax across relations for shared scale
        all_scores = []
        per_rel_scores = {}
        for rel in rels_order:
            ev = rel_vecs[rel]
            k_proj = W_k @ ev
            qk = pos_2d @ W_q.T @ k_proj  # (64,)
            qk = qk - qk.mean()
            score_2d = qk.reshape(base_size, base_size).numpy()
            per_rel_scores[rel] = score_2d
            all_scores.append(score_2d)
        amax = float(np.max(np.abs(np.stack(all_scores))))
        for i, rel in enumerate(rels_order):
            ax = axes[i // 4][i % 4]
            im = ax.imshow(per_rel_scores[rel], cmap="coolwarm", vmin=-amax, vmax=amax, origin="upper")
            ax.set_title(rel, fontsize=11, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(rf"$L_{layer}H_{head}$: image-grid attention score $\mathrm{{pos\_embed}}\cdot W_Q^\top W_K\cdot e_r$ per relation $r$",
                     fontsize=13, y=1.0)
        # Shared colorbar
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.66])
        fig.colorbar(im, cax=cax, label="QK score (mean-centred)")
        fig.tight_layout(rect=[0, 0, 0.91, 0.97])
        for ext in ("png", "pdf"):
            fig.savefig(join(out_dir, f"inner_product_map_L{layer}H{head}.{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[fig] inner_product_map_L{layer}H{head}.* -> {out_dir}")


if __name__ == "__main__":
    main()
