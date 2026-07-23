"""Multi-seed emergence analysis: E* CIs and routing-head consistency.

For each replication seed's checkpoint directory (from
train_replication_seed.py) plus the original run (seed 43, PixArt-alpha key
format), this script:
  1. re-runs the QK spatial-ramp alignment scan at every checkpoint,
  2. finds E* = first epoch with max-head |p| > 1 (Phase-II onset),
  3. records the top-2 heads by final-checkpoint |p| and whether both lie in
     layers 0-1,
and writes
  results/multiseed_emergence.csv           per seed x epoch x head |p|
  results/multiseed_summary.csv             per seed: E*, top heads
  results/multiseed_macros.tex              LaTeX macros (E*, seed counts)

Usage:
  python analyze_multiseed.py \
    --seed-dir 44=/path/ckpts_seed44 --seed-dir 45=/path/ckpts_seed45 \
    --seed-dir 46=/path/ckpts_seed46 \
    --original 43=/path/objrel_T5_DiT_mini_pilot/checkpoints \
    --t5-cache /path/t5_embedding_cache.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))  # DRC keeps utility modules in src/
import importlib, types
if "utils" not in sys.modules:  # let `from utils.X import ...` resolve to src/X
    _u = types.ModuleType("utils"); _u.__path__ = [str(REPO_ROOT / "src")]; sys.modules["utils"] = _u


class _Cfg:
    model = "PixArt_mini_2"
    image_size = 128
    patch_size = 2
    pred_sigma = True
    caption_channels = 4096
    train_sampling_steps = 1000


def build_transformer():
    from utils.pixart_utils import construct_diffuser_transformer_from_config
    return construct_diffuser_transformer_from_config(_Cfg())


def load_ckpt_into(transformer, path: Path) -> None:
    from utils.pixart_utils import load_pixart_ema_into_transformer
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
    if ckpt.get("format") == "diffusers" or any(k.startswith("transformer_blocks.") for k in sd):
        transformer.load_state_dict(
            {k: v for k, v in sd.items() if k in transformer.state_dict()}, strict=False)
        if "pos_embed.pos_embed" in sd:
            transformer.pos_embed.pos_embed.data.copy_(sd["pos_embed.pos_embed"])
    else:  # original PixArt-alpha key format
        load_pixart_ema_into_transformer(transformer, sd, strict=False)


def scan_epochs(ckpt_dir: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(ckpt_dir.glob("epoch_*.pth")):
        m = re.match(r"epoch_(\d+)", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


def alignment_for_ckpt(transformer, wordvec_proj_fn, scene_df, vp_features) -> pd.DataFrame:
    """|p| per head. wordvec_proj_fn re-projects the fixed raw T5 word vectors
    through THIS checkpoint's caption projection (it is trained per run)."""
    from utils.notebook_setup import compute_head_alignment
    wordvec_proj = wordvec_proj_fn(transformer)
    align_df, *_ = compute_head_alignment(
        transformer, wordvec_proj, scene_df, vp_features,
        n_perm=0, verbose=False,
    )
    align_df["abs_projection"] = align_df["projection"].abs()
    return align_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", action="append", default=[],
                    help="SEED=/path/to/ckpts (replication, diffusers format)")
    ap.add_argument("--original", default=None,
                    help="SEED=/path/to/ckpts (original run, pixart-alpha format)")
    ap.add_argument("--t5-cache", type=Path, required=True)
    ap.add_argument("--out-results", type=Path,
                    default=REPO_ROOT / "results")
    ap.add_argument("--out-macros", type=Path,
                    default=REPO_ROOT / "results" / "multiseed_macros.tex")
    ap.add_argument("--pstar-threshold", type=float, default=1.0)
    args = ap.parse_args()

    from utils.notebook_setup import generate_prompts_and_scene_info

    prompts, scene_infos, scene_df = generate_prompts_and_scene_info()
    vp_features = ["spatial_relationship", "shape1", "color2shape2"]

    # Fixed raw 4096-d word vectors (shape2 token) from the shared T5 cache;
    # per-checkpoint we re-project through that run's caption projection.
    raw = torch.load(args.t5_cache, map_location="cpu", weights_only=False)
    cache = raw.get("embedding_allrel_allobj", raw)
    from transformers import T5Tokenizer
    from utils.notebook_setup import extract_projected_word_vectors
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

    def wordvec_proj_fn(transformer):
        _, proj = extract_projected_word_vectors(
            cache, transformer, tokenizer, prompts, scene_infos,
            target_object="shape2")
        return proj

    runs = list(args.seed_dir)
    if args.original:
        runs.append(args.original)

    rows, summaries = [], []
    for spec in runs:
        seed_s, _, path_s = spec.partition("=")
        seed, ckpt_dir = int(seed_s), Path(path_s)
        epochs = scan_epochs(ckpt_dir)
        if not epochs:
            print(f"WARN: no checkpoints in {ckpt_dir}")
            continue
        transformer = build_transformer()
        per_epoch_max = {}
        final_align = None
        for ep in sorted(epochs):
            load_ckpt_into(transformer, epochs[ep])
            adf = alignment_for_ckpt(transformer, wordvec_proj_fn, scene_df, vp_features)
            adf["seed"], adf["epoch"] = seed, ep
            rows.append(adf)
            per_epoch_max[ep] = float(adf["abs_projection"].max())
            final_align = adf
            print(f"seed {seed} epoch {ep}: max|p|={per_epoch_max[ep]:.2f}")

        estar = next((ep for ep in sorted(per_epoch_max)
                      if per_epoch_max[ep] > args.pstar_threshold), None)
        top2 = final_align.nlargest(2, "abs_projection")[["layer", "head"]].values.tolist()
        summaries.append(dict(
            seed=seed,
            estar=estar,
            top1=f"L{top2[0][0]}H{top2[0][1]}",
            top2=f"L{top2[1][0]}H{top2[1][1]}",
            top2_early_layers=all(l <= 1 for l, _ in top2),
        ))

    args.out_results.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(args.out_results / "multiseed_emergence.csv", index=False)
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(args.out_results / "multiseed_summary.csv", index=False)
    print(sdf)

    estars = sdf["estar"].dropna().astype(int)
    if estars.empty:
        print("WARN: no seed crossed the |p| threshold — leaving paper macros "
              "in not-ready mode (check runs/threshold).")
        return
    macros = f"""% AUTO-GENERATED by analyze_multiseed.py — do not edit by hand.
\\newif\\ifmultiseedready
\\multiseedreadytrue
\\newcommand{{\\nseeds}}{{{len(sdf)}}}
\\newcommand{{\\estarlist}}{{{', '.join(map(str, sorted(estars)))}}}
\\newcommand{{\\estarmean}}{{{estars.mean():.0f}}}
\\newcommand{{\\estarrange}}{{{estars.min()}--{estars.max()}}}
\\newcommand{{\\topheadconsistency}}{{{sdf['top2_early_layers'].sum()}/{len(sdf)}}}
"""
    args.out_macros.write_text(macros)
    print(f"wrote {args.out_macros}")


if __name__ == "__main__":
    main()
