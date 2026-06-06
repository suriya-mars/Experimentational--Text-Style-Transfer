from __future__ import annotations

import torch

from phase2_forward_diffusion import ForwardDiffusion


class DDIMSampler:
    """
    Deterministic DDIM sampler for fast inference.
    Reduces steps from T to num_steps via a strided subsequence.
    """

    def __init__(self, forward_diff: ForwardDiffusion, num_steps: int = 50):
        self.fwd = forward_diff
        T = forward_diff.schedule.T
        # evenly spaced timestep subsequence
        self.timesteps = list(reversed(range(0, T, T // num_steps)))

    @torch.no_grad()
    def sample(
        self,
        predict_x0_fn,
        shape: tuple[int, ...],
        device: torch.device,
        style_cond: torch.Tensor | None = None,
        init_xt: torch.Tensor | None = None,
        start_step_idx: int = 0,
    ) -> torch.Tensor:
        """
        predict_x0_fn: callable(xt, t, style_cond) -> x0_hat
        shape: (B, L, d)
        start_step_idx: use >0 for SDEdit (partial noising)
        Returns final x0_hat (B, L, d).
        """
        schedule = self.fwd.schedule
        xt = init_xt if init_xt is not None else torch.randn(shape, device=device)

        steps = self.timesteps[start_step_idx:]
        for i, t_val in enumerate(steps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x0_hat = predict_x0_fn(xt, t, style_cond)

            if i < len(steps) - 1:
                t_prev = steps[i + 1]
                alpha_t = schedule.alphas_cumprod[t_val]
                alpha_prev = schedule.alphas_cumprod[t_prev]
                # DDIM deterministic update
                pred_eps = (xt - alpha_t.sqrt() * x0_hat) / (1 - alpha_t).sqrt()
                xt = alpha_prev.sqrt() * x0_hat + (1 - alpha_prev).sqrt() * pred_eps

        return x0_hat
