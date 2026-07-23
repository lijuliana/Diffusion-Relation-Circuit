# DiT Mechanistic Interpretability — Spatial-Relation Binding

Code, notebooks, and figures for the paper

> **A Two-Stage Cross-Attention Circuit for Spatial-Relation Binding in Small Diffusion Transformers**

We mechanistically resolve how a small PixArt-mini DiT (6 layers, 36 cross-attention heads, T5-XXL conditioning) converts a textual spatial relation into image layout. Two early-layer heads `{L0H0, L1H2}` implement spatial routing; Layer 2 implements distributed object binding. The full circuit emerges via a sharp phase transition during epochs 750–1000.

---

## Repository layout

```
dit-mech-interp/
├── README.md
├── LICENSE
├── requirements.txt
├── PixArt-alpha/    # vendored DiT model code (architecture + diffusion utils)
├── src/             # core utility modules (model loading, eval, hooks, etc.)
├── scripts/         # standalone reproduction scripts
│   └── multiseed/   # multi-seed retrain + emergence-analysis pipeline
├── notebooks/       # interactive walkthroughs (head discovery, ablation, variance partition)
├── results/         # CSVs of per-experiment results
└── figures/         # final paper figures (PDF + PNG)
```

### Heavy assets (not tracked in the Git repo)

The following files exceed GitHub's per-file 100 MB limit and are **not** in this repository. To run the notebooks, you'll need to obtain them separately and place them at the paths below:

| Path | Size | What it is |
|---|---|---|
| `checkpoints/epoch_{100,250,500,600,700,750,800,900,1000,2000,4000}_step_*.pth` | ~274 MB each, ~2.8 GB total | Trained PixArt-mini weights at 11 training epochs |
| `checkpoints/config.py` | small | Model/training config |
| `t5_embedding_cache.pt` | ~166 MB | Cached T5-XXL embeddings for all training prompts |
| `data/shapes_dataset_pilot.pth` | ~469 MB | Synthetic 2-color × 3-shape × 8-relation training data |

Contact the authors for access. The committed CSVs in [`results/`](results/) and PDFs/PNGs in [`figures/`](figures/) **do not require any of the heavy assets**.

---

## Setup

Tested on macOS (MPS) and Linux (CUDA).

```bash
git clone <this-repo>
cd dit-mech-interp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Quick sanity check:
```bash
python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"
```

---

## Paper figures and results

Final paper figures (PDF + PNG) live in [`figures/`](figures/) and the per-experiment CSVs that produced them live in [`results/`](results/).

---

## Notebooks

For interactive exploration of the discovery pipeline:

1. [`01_setup_and_head_discovery.ipynb`](notebooks/01_setup_and_head_discovery.ipynb) — model loading, embedding cache, alignment scan over all 36 heads.
2. [`02_ablation_and_causality.ipynb`](notebooks/02_ablation_and_causality.ipynb) — zero-head ablation, pair-ablation, super-additive interaction.
3. [`03_identifiability_and_variance_partition.ipynb`](notebooks/03_identifiability_and_variance_partition.ipynb) — variance-partitioned R² of the caption-projection geometry.

The notebooks reference modules in [`src/`](src/) and assume the project root is on `sys.path`.

---

## Multi-seed replication

[`scripts/multiseed/`](scripts/multiseed/) reproduces the phase-transition and
early-layer-routing robustness check across independent training seeds:

1. `render_shapes_dataset.py` — regenerate the synthetic object-relation dataset (spec-matched to the pilot).
2. `make_t5_cache.py` — build the frozen T5-XXL embedding table for the 144 prompts.
3. `train_replication_seed.py` — retrain PixArt-mini from scratch for one seed.
4. `analyze_multiseed.py` — re-run the alignment scan per checkpoint, locate the Phase-II onset `E*`, and write [`results/multiseed_summary.csv`](results/) / `multiseed_emergence.csv`.

Each script's `--help` documents its arguments.

Across three replication seeds (plus the original), the sharp phase transition
and early-layer routing localization reproduce, while the specific head indices
permute. These scripts import the utility modules under `src/` and, like the
notebooks, assume the project root is on `sys.path`.

---

## Compute

All experiments fit on a single NVIDIA A100 (80 GB) — PixArt-mini inference uses ~3 GB VRAM. Approximate budgets:

- Training (4000 epochs, batch 256, 128 px): ~15 GPU-hours
- Full ablation/patching/checkpoint sweeps reported in the paper: ~80 GPU-hours
- Total project (incl. preliminary scans): ~120 GPU-hours

---

## Model & data

- **Architecture:** PixArt-mini DiT, L=6 transformer blocks, H=6 cross-attention heads/block (36 total), d=384, d_head=64, with frozen T5-XXL conditioning and a learned 4096→384 caption projection.
- **Training:** 4000 epochs, batch 256, image 128 px, seed 43, CFG dropout 0.1, EMA.
- **Dataset:** synthetic prompts of the form `"[color1] [shape1] is [relation] [color2] [shape2]"` — 2 colours × 3 shapes (with same-colour and same-shape pairs excluded → 12 ordered (colour, shape)-pair templates) × 8 included spatial relations expanding into 12 unique natural-language phrase strings (1–3 paraphrases per relation; `in_front`/`behind` excluded) → **144 unique prompts**, sampled into **~10,000 prompt-image training pairs per epoch** (40 steps × 256 batch, derived from `epoch_E_step_S` checkpoint accounting).
- **Checkpoints:** {100, 250, 500, 600, 700, 750, 800, 900, 1000, 2000, 4000} epochs.

---

## Licenses

- Code: MIT (see [`LICENSE`](LICENSE))
- Trained model weights & synthetic prompt data: CC-BY-4.0
- Vendored PixArt-alpha code: AGPL-3.0 (see `PixArt-alpha/`)
- T5-XXL: Apache-2.0 (Google)
- OpenCV (post-hoc evaluator): Apache-2.0

---

## Citation

```
@misc{li2026twostage,
  title  = {A Two-Stage Cross-Attention Circuit for Spatial-Relation Binding in Small Diffusion Transformers},
  author = {Li, Juliana and Wang, Binxu},
  year   = {2026}
}
```
