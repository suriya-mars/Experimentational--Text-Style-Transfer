from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Combined training loss:
        L = L_simple + λ_embed * L_embed + λ_style * L_style

    L_simple:  MSE between predicted and true x_0
    L_embed:   anchors embedding weights to diffusion manifold
    L_style:   cross-entropy on style classifier predictions (optional)
    """

    def __init__(
        self,
        lambda_embed: float = 0.01,
        lambda_style: float = 0.1,
    ):
        super().__init__()
        self.lambda_embed = lambda_embed
        self.lambda_style = lambda_style

    def forward(
        self,
        x0_pred: torch.Tensor,
        x0_true: torch.Tensor,
        embed_loss: torch.Tensor | None = None,
        style_logits: torch.Tensor | None = None,
        style_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_simple = F.mse_loss(x0_pred, x0_true)

        total = l_simple
        losses = {"l_simple": l_simple}

        if embed_loss is not None:
            total = total + self.lambda_embed * embed_loss
            losses["l_embed"] = embed_loss

        if style_logits is not None and style_labels is not None:
            l_style = F.cross_entropy(style_logits, style_labels)
            total = total + self.lambda_style * l_style
            losses["l_style"] = l_style

        losses["total"] = total
        return losses
