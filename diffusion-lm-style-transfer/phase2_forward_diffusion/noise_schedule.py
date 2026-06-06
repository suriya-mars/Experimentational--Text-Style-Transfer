from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    betas: torch.Tensor       # (T,)
    alphas: torch.Tensor      # (T,)   1 - beta_t
    alphas_cumprod: torch.Tensor   # (T,)   ā_t
    alphas_cumprod_prev: torch.Tensor  # (T,)  ā_{t-1}
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @property
    def T(self) -> int:
        return len(self.betas)


def linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    return _build_schedule(betas)


def cosine_schedule(T: int, s: float = 0.008) -> NoiseSchedule:
    """Improved cosine schedule from Nichol & Dhariwal 2021."""
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = betas.clamp(max=0.999)
    return _build_schedule(betas)


def _build_schedule(betas: torch.Tensor) -> NoiseSchedule:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]])

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return NoiseSchedule(
        betas=betas.float(),
        alphas=alphas.float(),
        alphas_cumprod=alphas_cumprod.float(),
        alphas_cumprod_prev=alphas_cumprod_prev.float(),
        sqrt_alphas_cumprod=alphas_cumprod.sqrt().float(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt().float(),
        posterior_variance=posterior_variance.float(),
    )
