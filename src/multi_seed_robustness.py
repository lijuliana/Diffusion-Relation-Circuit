"""
Multi-seed / bootstrap helpers for notebook robustness checks.

Design goals
------------
- Keep default compute small: use a short seed tuple (3–5) and low ``n_perm``
  for bootstrap alignment passes.
- **Bootstrap resampling** of prompt rows changes ``effect_vecs`` and thus
  head-alignment cosines (unlike permutation ``random_state`` alone).
- **Diffusion ``generator_seed``** varies stochastic decoding for cached-embed
  evals without changing the T5 cache.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

# Odd, spaced seeds — easy to extend / shrink in notebooks.
DEFAULT_ROBUSTNESS_SEEDS: tuple[int, ...] = (41, 43, 45, 47, 49)


def bootstrap_row_indices(n: int, seed: int) -> np.ndarray:
    """Length-``n`` with-replacement row indices (NumPy Generator)."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(low=0, high=int(n), size=int(n), endpoint=False)


def align_df_top_head_pairs(
    align_df: pd.DataFrame,
    k: int = 6,
    score_col: str = "abs_cosine",
) -> list[tuple[int, int]]:
    """Return top-``k`` (layer, head) pairs from a head-alignment dataframe."""
    if score_col not in align_df.columns:
        raise KeyError(f"{score_col!r} not in align_df columns: {list(align_df.columns)}")
    if "layer" in align_df.columns and "head" in align_df.columns:
        sub = align_df.nlargest(int(k), score_col)
        return [(int(r.layer), int(r.head)) for r in sub.itertuples(index=False)]
    if "layer_idx" in align_df.columns and "head_idx" in align_df.columns:
        sub = align_df.nlargest(int(k), score_col)
        return [(int(r.layer_idx), int(r.head_idx)) for r in sub.itertuples(index=False)]
    raise ValueError(
        "align_df must have columns (layer, head) or (layer_idx, head_idx); "
        f"got {list(align_df.columns)}"
    )


def topk_stability_table(topk_per_seed: list[list[tuple[int, int]]]) -> pd.DataFrame:
    """
    Fraction of seeds in which each head appears in the per-seed top-k list.

    ``topk_per_seed`` has one list per seed (each inner list length k).
    """
    if not topk_per_seed:
        return pd.DataFrame(columns=["layer", "head", "in_topk_frac", "n_seeds"])
    n_seeds = len(topk_per_seed)
    c: Counter[tuple[int, int]] = Counter()
    for lst in topk_per_seed:
        for h in lst:
            c[h] += 1
    rows = [
        {"layer": h[0], "head": h[1], "in_topk_frac": c[h] / n_seeds, "n_seeds": n_seeds}
        for h in c
    ]
    return pd.DataFrame(rows).sort_values(["in_topk_frac", "layer", "head"], ascending=[False, True, True])


def multi_bootstrap_head_alignment(
    transformer,
    wordvec_proj: np.ndarray,
    scene_info_df: pd.DataFrame,
    vp_features: list[str],
    seeds: Sequence[int],
    *,
    n_perm: int = 24,
    top_k: int = 6,
    base_size: int = 8,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[list[tuple[int, int]]], list[pd.DataFrame]]:
    """
    Repeated bootstrap head alignment (same contract as ``compute_head_alignment``).

    Returns
    -------
    stability_df : fraction of seeds each head appears in top-k
    tops : per-seed top-k head lists
    align_dfs : raw alignment tables per seed
    """
    from utils.notebook_setup import compute_head_alignment

    n = int(len(wordvec_proj))
    tops: list[list[tuple[int, int]]] = []
    align_dfs: list[pd.DataFrame] = []
    for s in seeds:
        idx = bootstrap_row_indices(n, int(s))
        adf, *_rest = compute_head_alignment(
            transformer,
            wordvec_proj[idx],
            scene_info_df.iloc[idx].reset_index(drop=True),
            vp_features,
            n_perm=int(n_perm),
            base_size=int(base_size),
            verbose=verbose,
            random_state=int(s),
        )
        align_dfs.append(adf)
        tops.append(align_df_top_head_pairs(adf, k=int(top_k), score_col="abs_cosine"))
    stab = topk_stability_table(tops)
    return stab, tops, align_dfs


def eval_cached_ablation_effect_multiseed(
    pipeline,
    prompts: list,
    scene_infos: list,
    embedding_cache: dict,
    layer_idx: int,
    head_idx: int,
    seeds: Sequence[int],
    *,
    device,
    weight_dtype,
    num_inference_steps: int = 14,
    guidance_scale: float = 4.5,
    num_images: int = 1,
    show_prompt_progress: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Baseline vs single-head zero ablation for each diffusion ``generator_seed``.

    Returns per-seed rows and a Series of mean/std for acc_baseline, acc_ablated, acc_drop.
    """
    from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings
    from utils.zero_head_ablation_utils import apply_zero_head_ablation, restore_processors

    rows: list[dict] = []
    for s in seeds:
        eval_b, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
            pipeline,
            prompts,
            scene_infos,
            embedding_cache,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator_seed=int(s),
            device=device,
            weight_dtype=weight_dtype,
            show_prompt_progress=show_prompt_progress,
        )
        orig = apply_zero_head_ablation(pipeline.transformer, int(layer_idx), [int(head_idx)])
        try:
            eval_a, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipeline,
                prompts,
                scene_infos,
                embedding_cache,
                num_images=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator_seed=int(s),
                device=device,
                weight_dtype=weight_dtype,
                show_prompt_progress=show_prompt_progress,
            )
        finally:
            restore_processors(pipeline.transformer, orig)

        acc_b = float(eval_b["spatial_relationship"].mean())
        acc_a = float(eval_a["spatial_relationship"].mean())
        rows.append(
            dict(
                seed=int(s),
                acc_baseline=acc_b,
                acc_ablated=acc_a,
                acc_drop=acc_b - acc_a,
            )
        )
    df = pd.DataFrame(rows)
    summary = pd.concat(
        [
            df[["acc_baseline", "acc_ablated", "acc_drop"]].mean().rename(lambda x: f"{x}_mean"),
            df[["acc_baseline", "acc_ablated", "acc_drop"]].std(ddof=0).rename(lambda x: f"{x}_std"),
        ]
    )
    return df, summary


def emergence_top_head_per_epoch(
    evolution_df: pd.DataFrame,
    *,
    score_col: str = "abs_cosine",
    layer_col: str = "layer_idx",
    head_col: str = "head_idx",
) -> pd.DataFrame:
    """
    For each epoch, which head has the highest ``score_col`` (deterministic summary).

    Used as a baseline before comparing to bootstrap runs.
    """
    if evolution_df.empty:
        return pd.DataFrame(columns=["epoch", "layer", "head", score_col])
    out_rows = []
    for ep, grp in evolution_df.groupby("epoch"):
        j = int(grp[score_col].values.argmax())
        row = grp.iloc[j]
        out_rows.append(
            {
                "epoch": ep,
                "layer": int(row[layer_col]),
                "head": int(row[head_col]),
                score_col: float(row[score_col]),
            }
        )
    return pd.DataFrame(out_rows).sort_values("epoch")


def eval_pair_ablation_multiseed(
    pipeline,
    prompts: list,
    scene_infos: list,
    embedding_cache: dict,
    head_pairs: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    *,
    device,
    weight_dtype,
    num_inference_steps: int = 14,
    guidance_scale: float = 4.5,
    num_images: int = 1,
    show_prompt_progress: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Baseline vs **multi-head** zero ablation (``head_pairs``) for each diffusion seed.
    """
    from utils.eval_cached_embeddings import evaluate_pipeline_on_prompts_with_cached_embeddings
    from utils.zero_head_ablation_utils import apply_zero_head_ablation_multi, restore_processors

    pairs = [(int(a), int(b)) for a, b in head_pairs]
    rows: list[dict] = []
    for s in seeds:
        eval_b, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
            pipeline,
            prompts,
            scene_infos,
            embedding_cache,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator_seed=int(s),
            device=device,
            weight_dtype=weight_dtype,
            show_prompt_progress=show_prompt_progress,
        )
        orig = apply_zero_head_ablation_multi(pipeline.transformer, pairs)
        try:
            eval_a, _ = evaluate_pipeline_on_prompts_with_cached_embeddings(
                pipeline,
                prompts,
                scene_infos,
                embedding_cache,
                num_images=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator_seed=int(s),
                device=device,
                weight_dtype=weight_dtype,
                show_prompt_progress=show_prompt_progress,
            )
        finally:
            restore_processors(pipeline.transformer, orig)

        acc_b = float(eval_b["spatial_relationship"].mean())
        acc_a = float(eval_a["spatial_relationship"].mean())
        rows.append(
            dict(
                seed=int(s),
                acc_baseline=acc_b,
                acc_ablated=acc_a,
                acc_drop=acc_b - acc_a,
                heads="+".join(f"L{li}H{hi}" for li, hi in pairs),
            )
        )
    df = pd.DataFrame(rows)
    summary = pd.concat(
        [
            df[["acc_baseline", "acc_ablated", "acc_drop"]].mean().rename(lambda x: f"{x}_mean"),
            df[["acc_baseline", "acc_ablated", "acc_drop"]].std(ddof=0).rename(lambda x: f"{x}_std"),
        ]
    )
    return df, summary


def downstream_rank_stability_from_tables(
    rank_tables: list[pd.DataFrame],
    *,
    head_col_layer: str = "layer_idx",
    head_col_idx: str = "head_idx",
    rank_col: str | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Stability of top-k ranked candidates across repeated ranking tables (e.g. per-seed).

    Each table must be sortable by ``rank_col`` (defaults to first numeric column
    named like ``chain_score`` / ``write_score`` / ``score``).
    """
    if not rank_tables:
        return pd.DataFrame()
    tops: list[list[tuple[int, int]]] = []
    for df in rank_tables:
        if rank_col is None:
            candidates = [
                c
                for c in df.columns
                if c in ("chain_score", "write_score", "read_score", "score", "total_score")
            ]
            rc = candidates[0] if candidates else df.select_dtypes(include=[np.number]).columns[0]
        else:
            rc = rank_col
        sub = df.nlargest(int(top_k), rc)
        tops.append(
            [(int(r[head_col_layer]), int(r[head_col_idx])) for _, r in sub.iterrows()]
        )
    return topk_stability_table(tops)


__all__ = [
    "DEFAULT_ROBUSTNESS_SEEDS",
    "bootstrap_row_indices",
    "align_df_top_head_pairs",
    "topk_stability_table",
    "multi_bootstrap_head_alignment",
    "eval_cached_ablation_effect_multiseed",
    "eval_pair_ablation_multiseed",
    "emergence_top_head_per_epoch",
    "downstream_rank_stability_from_tables",
]
