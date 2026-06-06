from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleClassifier(nn.Module):
    """
    Classifies style from noised x_0_hat predictions.
    Trained on embeddings at random timesteps so it can guide denoising at any t.
    """

    def __init__(self, embed_dim: int, num_styles: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_styles),
        )

    def forward(self, x0_hat: torch.Tensor) -> torch.Tensor:
        """x0_hat: (B, L, d) → mean pool → logits: (B, num_styles)"""
        pooled = x0_hat.mean(dim=1)   # (B, d)
        return self.net(pooled)

    def log_prob(self, x0_hat: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x0_hat)
        return -F.cross_entropy(logits, style_ids, reduction="none")


class ClassifierGuidance:
    """
    Applies classifier guidance to shift x_0_hat toward the target style.

    guided_x0 = x0_hat + γ * ∇_{x0_hat} log p_φ(style | x0_hat)
    """

    def __init__(self, classifier: StyleClassifier, guidance_scale: float = 3.0):
        self.classifier = classifier
        self.guidance_scale = guidance_scale

    def guide(self, x0_hat: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
        """
        x0_hat:       (B, L, d)  — requires_grad must be True
        target_style: (B,) int   — target style class ids
        Returns guided x0_hat (same shape).
        """
        x = x0_hat.detach().requires_grad_(True)
        log_p = self.classifier.log_prob(x, target_style).sum()
        grad = torch.autograd.grad(log_p, x)[0]
        return x0_hat + self.guidance_scale * grad.detach()
