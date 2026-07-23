"""Build the T5-XXL embedding cache for the 144 replication prompts (+ null).

The original t5_embedding_cache.pt is not distributed here; this rebuilds
an equivalent cache anywhere with ~25 GB of GPU (or CPU, slowly). T5 encoding
is deterministic given the weights, so the rebuilt cache matches the original
for shared prompts. Same key convention: ``base::<prompt>`` plus ``""``.

Usage:
  python make_t5_cache.py --prompts /path/shapes_out/prompts.json \
      --out /path/t5_embedding_cache.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL_MAX_LENGTH = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True, help="prompts.json from render_shapes_dataset.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="google/t5-v1_1-xxl")
    args = ap.parse_args()

    from transformers import T5EncoderModel, T5Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    encoder = T5EncoderModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    prompts = json.loads(args.prompts.read_text())["prompts"]
    cache = {}
    with torch.no_grad():
        for text, key in [("", "")] + [(p, f"base::{p}") for p in prompts]:
            tok = tokenizer(text, max_length=MODEL_MAX_LENGTH, padding="max_length",
                            truncation=True, return_tensors="pt").to(device)
            emb = encoder(tok.input_ids, attention_mask=tok.attention_mask).last_hidden_state
            cache[key] = {
                "caption_embeds": emb.float().cpu(),
                "attention_mask": tok.attention_mask.float().cpu(),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embedding_allrel_allobj": cache}, args.out)
    print(f"wrote {len(cache)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
