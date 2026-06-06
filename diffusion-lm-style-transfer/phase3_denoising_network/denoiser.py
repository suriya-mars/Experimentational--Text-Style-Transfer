from __future__ import annotations

import torch
import torch.nn as nn

from phase2_forward_diffusion import ForwardDiffusion
from .transformer import DenoisingTransformer


class Denoiser(nn.Module):
    """
    Wraps DenoisingTransformer with the diffusion training loss (L_simple).
    Predicts x_0 and computes MSE against ground truth x_0.
    """

    def __init__(self, transformer: DenoisingTransformer, forward_diff: ForwardDiffusion):
        super().__init__()
        self.transformer = transformer
        self.forward_diff = forward_diff

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        style_cond: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.
        Returns dict with 'loss' (scalar), 'x0_pred' (B, L, d), 'xt' (B, L, d).
        """
        noise = torch.randn_like(x0)
        xt = self.forward_diff.q_sample(x0, t, noise=noise)
        x0_pred = self.transformer(xt, t, style_cond, padding_mask)
        loss = nn.functional.mse_loss(x0_pred, x0)
        return {"loss": loss, "x0_pred": x0_pred, "xt": xt}

    @torch.no_grad()
    def predict_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        style_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transformer(xt, t, style_cond)
