"""Render the synthetic object-relation shapes dataset for replication training.

Reimplements the published data specification (128 px canvas, red/blue x
square/triangle/circle, 8 spatial relations, radius 16 px, conventions matched
to src/cv2_eval_utils.py). The original 469 MB `shapes_dataset_pilot.pth`
is not distributed here; this renderer regenerates an equivalent
distribution so replication seeds can train anywhere.

NOTE for the paper: replication runs use this reimplemented generator matched
to the published spec, not the byte-identical original dataset. Data is fixed
across training seeds (``--data-seed``) so runs differ only in model init,
batch order, noise draws, and CFG-dropout draws.

Output: ``<out>/shapes_dataset.npz`` with
  images  uint8 (N, 128, 128, 3)
  prompt_idx int64 (N,)           index into prompts.json
plus ``<out>/prompts.json`` — the 144 unique prompts and scene infos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))  # DRC keeps utility modules in src/
import importlib, types
if "utils" not in sys.modules:  # let `from utils.X import ...` resolve to src/X
    _u = types.ModuleType("utils"); _u.__path__ = [str(REPO_ROOT / "src")]; sys.modules["utils"] = _u

from utils.relation_shape_dataset_lib import DEFAULT_SPATIAL_PHRASES  # noqa: E402

CANVAS = 128
RADIUS = 16  # matches cv2_eval_utils.find_classify_objects(radius=16.0)
COLORS = {"red": (255, 0, 0), "blue": (0, 0, 255)}
# (dy, dx) sign convention: "obj1 above obj2" => y1 < y2 (image row axis).
REL_DIR = {
    "above": (-1, 0), "below": (1, 0), "left": (0, -1), "right": (0, 1),
    "upper_left": (-1, -1), "upper_right": (-1, 1),
    "lower_left": (1, -1), "lower_right": (1, 1),
}
PRIMARY_MIN = 20   # offset (px) along each active axis — clearly satisfies the loose >=8 px rule
ORTHO_MAX = 40     # max off-axis offset for cardinal relations (relaxed — original data is random placement)
MIN_SEP = int(2 * RADIUS * 1.2)  # keep shapes disjoint (strict evaluator requires no overlap)


def generate_prompt_bank():
    """144 unique prompts: 12 ordered (color, shape)-pair templates x 12 unique
    phrase strings. Some phrases appear under multiple relation keys in
    DEFAULT_SPATIAL_PHRASES (e.g. "above and to the right of" under both
    `above` and `upper_right`); dedupe by prompt string, labelling each phrase
    with its most specific relation (diagonal beats cardinal) so a rendered
    placement satisfies every reading of the phrase."""
    from itertools import product

    def specificity(rel):
        dy, dx = REL_DIR[rel]
        return abs(dy) + abs(dx)

    phrase_rel: dict[str, str] = {}
    for rel, texts in DEFAULT_SPATIAL_PHRASES.items():
        if rel in ("in_front", "behind"):
            continue
        for text in texts:
            if text not in phrase_rel or specificity(rel) > specificity(phrase_rel[text]):
                phrase_rel[text] = rel

    prompts, infos = [], []
    for c1, c2 in product(COLORS, COLORS):
        if c1 == c2:
            continue
        for s1, s2 in product(["square", "triangle", "circle"], repeat=2):
            if s1 == s2:
                continue
            for text, rel in phrase_rel.items():
                prompts.append(f"{c1} {s1} is {text} {c2} {s2}")
                infos.append(dict(color1=c1, shape1=s1, color2=c2, shape2=s2,
                                  spatial_relationship=rel))
    return prompts, infos


def draw_shape(img: np.ndarray, shape: str, color: tuple, cx: int, cy: int) -> None:
    import cv2
    if shape == "circle":
        cv2.circle(img, (cx, cy), RADIUS, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (cx - RADIUS, cy - RADIUS), (cx + RADIUS, cy + RADIUS), color, -1)
    elif shape == "triangle":
        h = int(RADIUS * np.sqrt(3))
        pts = np.array([[cx, cy - h * 2 // 3],
                        [cx - RADIUS, cy + h // 3],
                        [cx + RADIUS, cy + h // 3]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(shape)


def sample_positions(rel: str, rng: np.random.Generator, max_tries: int = 200):
    dy, dx = REL_DIR[rel]
    m = RADIUS + 2
    for _ in range(max_tries):
        x1, y1 = rng.integers(m, CANVAS - m, 2)
        x2, y2 = rng.integers(m, CANVAS - m, 2)
        ddx, ddy = x1 - x2, y1 - y2  # obj1 minus obj2
        ok = True
        for d, sign in ((ddy, dy), (ddx, dx)):
            if sign != 0:
                ok &= (d * sign >= PRIMARY_MIN)
            else:
                ok &= (abs(d) <= ORTHO_MAX)
        ok &= (abs(ddx) >= MIN_SEP or abs(ddy) >= MIN_SEP)
        if ok:
            return int(x1), int(y1), int(x2), int(y2)
    raise RuntimeError(f"could not place objects for relation {rel!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-images", type=int, default=10_000)
    ap.add_argument("--data-seed", type=int, default=1000)
    args = ap.parse_args()

    prompts, infos = generate_prompt_bank()
    assert len(prompts) == 144, f"expected 144 unique prompts, got {len(prompts)}"

    rng = np.random.default_rng(args.data_seed)
    images = np.zeros((args.num_images, CANVAS, CANVAS, 3), dtype=np.uint8)
    prompt_idx = rng.integers(0, len(prompts), size=args.num_images)
    for i, pi in enumerate(prompt_idx):
        info = infos[pi]
        x1, y1, x2, y2 = sample_positions(info["spatial_relationship"], rng)
        img = images[i]
        draw_shape(img, info["shape1"], COLORS[info["color1"]], x1, y1)
        draw_shape(img, info["shape2"], COLORS[info["color2"]], x2, y2)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "shapes_dataset.npz",
                        images=images, prompt_idx=prompt_idx.astype(np.int64))
    (args.out / "prompts.json").write_text(json.dumps(
        {"prompts": prompts, "scene_infos": infos}, indent=1))
    print(f"wrote {args.num_images} images -> {args.out}")


if __name__ == "__main__":
    main()
