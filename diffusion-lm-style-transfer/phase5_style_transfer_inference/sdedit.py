from __future__ import annotations

import torch

from phase2_forward_diffusion import ForwardDiffusion
from .ddim import DDIMSampler


class SDEditTransfer:
    """
    SDEdit-style partial noising for content-preserving style transfer.

    1. Embed source text → x_0_src
    2. Noise to t* ~ q(x_t* | x_0_src)
    3. Denoise with target style condition
    Higher t_frac → more style change, less content preservation.
    """

    def __init__(self, forward_diff: ForwardDiffusion, ddim_sampler: DDIMSampler):
        self.fwd = forward_diff
        self.ddim = ddim_sampler

    @torch.no_grad()
    def transfer(
        self,
        x0_src: torch.Tensor,
        predict_x0_fn,
        style_cond: torch.Tensor,
        t_frac: float = 0.5,
    ) -> torch.Tensor:
        """
        x0_src:        (B, L, d) source embeddings
        predict_x0_fn: callable(xt, t, style_cond) -> x0_hat
        style_cond:    (B, d_style) target style vector
        t_frac:        float in (0, 1] — fraction of T to noise to
        Returns x0_hat (B, L, d).
        """
        T = self.fwd.schedule.T
        t_star = max(1, int(t_frac * T))
        t = torch.full((x0_src.shape[0],), t_star, device=x0_src.device, dtype=torch.long)
        xt_star = self.fwd.q_sample(x0_src, t)

        # find the index in DDIM timestep list closest to t_star
        start_idx = next(
            (i for i, ts in enumerate(self.ddim.timesteps) if ts <= t_star),
            0,
        )

        return self.ddim.sample(
            predict_x0_fn=predict_x0_fn,
            shape=x0_src.shape,
            device=x0_src.device,
            style_cond=style_cond,
            init_xt=xt_star,
            start_step_idx=start_idx,
        )
