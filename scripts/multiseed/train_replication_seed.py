"""Train one PixArt-mini replication seed on the object-relation shapes dataset.

Matches the original run's recipe (CLAUDE.md / paper Sec. 3): PixArt-mini
(6L x 6H, d=384), frozen T5-XXL conditioning via a cached embedding table,
image 128 px, batch 256, 40 steps/epoch (~10k pairs/epoch), CFG dropout 0.1,
AdamW, EMA. Only the seed differs across runs. Truncated to 2000 epochs by
default: the phase transition under study is at 750-1000 and behavioral
accuracy has largely saturated by 2000 (Table temporal in the paper).

Checkpoints are saved in DIFFUSERS key format (plus EMA) as
``epoch_{E}_seed{S}.pth``; use analyze_multiseed.py (not the notebook
loaders, which expect PixArt-alpha key naming) to read them.

Single-GPU friendly: the model is ~30M params; a batch-256 step at 128 px fits
in <10 GB, so request ONE GPU per seed and run seeds as separate pods/jobs.

Usage:
  python train_replication_seed.py --seed 44 \
      --data /path/shapes_out --t5-cache /path/t5_embedding_cache.pt \
      --out /path/ckpts_seed44
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))  # DRC keeps utility modules in src/
import importlib, types
if "utils" not in sys.modules:  # let `from utils.X import ...` resolve to src/X
    _u = types.ModuleType("utils"); _u.__path__ = [str(REPO_ROOT / "src")]; sys.modules["utils"] = _u
sys.path.insert(0, str(REPO_ROOT / "PixArt-alpha"))

CKPT_EPOCHS = (100, 250, 500, 600, 700, 750, 800, 900, 1000, 1500, 2000)
MODEL_MAX_LENGTH = 20


class _Cfg:
    model = "PixArt_mini_2"
    image_size = 128
    patch_size = 2
    pred_sigma = True
    caption_channels = 4096
    train_sampling_steps = 1000


def load_t5_table(cache_path: Path, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(P, L, 4096) embedding table for the 144 prompts + the null ('') embedding."""
    raw = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = raw["embedding_allrel_allobj"] if "embedding_allrel_allobj" in raw else raw

    def look(key_prompt: str):
        for k in (f"base::{key_prompt}", key_prompt):
            if k in cache:
                return cache[k]
        for k in cache:
            if str(k).endswith(f"::{key_prompt}"):
                return cache[k]
        raise KeyError(f"prompt not in T5 cache: {key_prompt!r}")

    embs, masks = [], []
    for p in prompts:
        d = look(p)
        e = d["caption_embeds"]
        e = e[0] if e.ndim == 3 else e
        m = d.get("attention_mask")
        if m is not None:
            m = m[0] if m.ndim == 2 else m
        else:
            m = torch.ones(e.shape[0])
        embs.append(e[:MODEL_MAX_LENGTH].float())
        masks.append(m[:MODEL_MAX_LENGTH].float())
    table = torch.stack(embs).to(device)
    mask = torch.stack(masks).to(device)
    try:
        d0 = look("")
        e0 = d0["caption_embeds"]
        null = (e0[0] if e0.ndim == 3 else e0)[:MODEL_MAX_LENGTH].float().to(device)
    except KeyError:
        null = torch.zeros_like(table[0])
    return table, mask, null


@torch.no_grad()
def encode_latents(images: np.ndarray, device, batch: int = 128) -> torch.Tensor:
    """VAE-encode uint8 (N,H,W,3) -> fp32 latents (N,4,16,16), scaled."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float32).to(device)
    sf = vae.config.scaling_factor
    outs = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        outs.append(vae.encode(x).latent_dist.sample() * sf)
    del vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return torch.cat(outs).float().cpu()


def ema_update(ema: dict, model: torch.nn.Module, decay: float = 0.9999) -> None:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].mul_(decay).add_(v, alpha=1 - decay)
            else:
                ema[k].copy_(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir from render_shapes_dataset.py")
    ap.add_argument("--t5-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cfg-dropout", type=float, default=0.1)
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (default: auto; note MPS lacks float64 for the "
                         "IDDPM schedule — use cpu for local smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from utils.pixart_utils import construct_diffuser_transformer_from_config
    from diffusion import IDDPM  # vendored PixArt-alpha

    bank = json.loads((args.data / "prompts.json").read_text())
    prompts = bank["prompts"]
    ds = np.load(args.data / "shapes_dataset.npz")
    images, prompt_idx = ds["images"], ds["prompt_idx"]

    lat_path = args.data / "latents.pt"
    if lat_path.exists():
        latents = torch.load(lat_path, map_location="cpu", weights_only=True)
    else:
        print("encoding latents (one-time)...")
        latents = encode_latents(images, device)
        torch.save(latents, lat_path)
    t5_table, t5_mask, t5_null = load_t5_table(args.t5_cache, prompts, device)

    model = construct_diffuser_transformer_from_config(_Cfg()).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"seed={args.seed} device={device} params={n_params/1e6:.1f}M")
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    diffusion = IDDPM(timestep_respacing="")  # 1000-step linear schedule, learned sigma
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    n = len(latents)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            idx = rng.integers(0, n, size=args.batch)
            x0 = latents[idx].to(device)
            pids = prompt_idx[idx]
            y = t5_table[pids].clone()
            mask = t5_mask[pids].clone()
            drop = torch.from_numpy(rng.random(args.batch) < args.cfg_dropout).to(device)
            if drop.any():
                y[drop] = t5_null
                mask[drop] = 1.0
            t = torch.from_numpy(rng.integers(0, diffusion.num_timesteps, size=args.batch)).to(device)
            terms = diffusion.training_losses_diffusers(
                model, x0, t,
                model_kwargs=dict(encoder_hidden_states=y, encoder_attention_mask=mask),
            )
            loss = terms["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model)
        if epoch % 50 == 0 or epoch in CKPT_EPOCHS:
            print(f"epoch {epoch}  loss {loss.item():.4f}  {(time.time()-t0)/60:.1f} min")
        if epoch in CKPT_EPOCHS and epoch <= args.epochs:
            torch.save(
                {"state_dict": model.state_dict(), "state_dict_ema": ema,
                 "format": "diffusers", "seed": args.seed, "epoch": epoch},
                args.out / f"epoch_{epoch}_seed{args.seed}.pth",
            )
    print(f"done: seed {args.seed} in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
