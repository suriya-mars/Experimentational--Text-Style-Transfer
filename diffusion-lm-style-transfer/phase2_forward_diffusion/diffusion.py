from __future__ import annotations

import torch

from .noise_schedule import NoiseSchedule


class ForwardDiffusion:
    """
    Implements the closed-form forward process:
        q(x_t | x_0) = N(x_t; sqrt(ā_t) * x_0, (1 - ā_t) * I)
    """

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t given x_0 and timestep t.
        x0: (B, L, d)
        t:  (B,) integer timestep indices
        Returns x_t: (B, L, d)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self._gather(self.schedule.sqrt_alphas_cumprod, t, x0)
        sqrt_one_minus = self._gather(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def q_posterior_mean_variance(
        self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute q(x_{t-1} | x_t, x_0) mean and variance."""
        schedule = self.schedule
        posterior_mean = (
            self._gather(schedule.betas * schedule.alphas_cumprod_prev.sqrt() / (1.0 - schedule.alphas_cumprod), t, x0) * x0
            + self._gather(schedule.alphas.sqrt() * (1.0 - schedule.alphas_cumprod_prev) / (1.0 - schedule.alphas_cumprod), t, xt) * xt
        )
        posterior_variance = self._gather(schedule.posterior_variance, t, x0)
        return posterior_mean, posterior_variance

    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Gather schedule values at timestep t and broadcast to ref shape."""
        out = values.to(ref.device)[t]
        # broadcast over (L, d)
        return out.view(ref.shape[0], *([1] * (ref.ndim - 1)))
