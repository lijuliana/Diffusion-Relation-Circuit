# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
#
# Lazy exports: importing diffusion.data.* or diffusion.model.builder must not
# load training samplers or PixArt nets (xformers). Use explicit imports when needed.

from __future__ import annotations

from typing import Any

__all__ = ("IDDPM", "DPMS", "SASolverSampler")


def __getattr__(name: str) -> Any:
    if name == "IDDPM":
        from .iddpm import IDDPM

        return IDDPM
    if name == "DPMS":
        from .dpm_solver import DPMS

        return DPMS
    if name == "SASolverSampler":
        from .sa_sampler import SASolverSampler

        return SASolverSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
