from __future__ import annotations

import torch
import torch.nn as nn


class CFGWrapper(nn.Module):
    """
    Classifier-Free Guidance wrapper around any denoiser.

    At training time, randomly drops the style condition with probability p_uncond.
    At inference time, interpolates conditional and unconditional predictions:
        x0_hat = (1 + γ) * f(xt, t, c) - γ * f(xt, t, ∅)
    """

    def __init__(self, denoiser: nn.Module, p_uncond: float = 0.1):
        super().__init__()
        self.denoiser = denoiser
        self.p_uncond = p_uncond
        # learnable null embedding used as unconditional condition
        self.null_style = nn.Parameter(torch.zeros(1, 256))

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        style_cond: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict:
        if self.training:
            # randomly replace condition with null embedding
            mask = (torch.rand(x0.shape[0], device=x0.device) < self.p_uncond)
            null = self.null_style.expand(x0.shape[0], -1)
            cond = torch.where(mask.unsqueeze(-1), null, style_cond)
            return self.denoiser(x0, t, style_cond=cond, padding_mask=padding_mask)
        raise RuntimeError("Use guided_predict for inference.")

    @torch.no_grad()
    def guided_predict(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        style_cond: torch.Tensor,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        null = self.null_style.expand(xt.shape[0], -1).to(xt.device)
        x0_cond = self.denoiser.predict_x0(xt, t, style_cond=style_cond)
        x0_uncond = self.denoiser.predict_x0(xt, t, style_cond=null)
        return (1 + guidance_scale) * x0_cond - guidance_scale * x0_uncond
