from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundingLayer(nn.Module):
    """
    Maps continuous diffusion output x_0_hat back to discrete token ids.

    Hard rounding: argmax over cosine similarity to embedding matrix.
    Soft rounding: differentiable softmax version for training-time anchor loss.
    """

    def __init__(self, embedding_weight: torch.Tensor, temperature: float = 1.0):
        super().__init__()
        # embedding_weight: (V, d) — shared with EmbeddingLayer, not a copy
        self.register_buffer("embedding_weight", embedding_weight)
        self.temperature = temperature

    def soft_round(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) → logits: (B, L, V)  — differentiable, used during denoising steps."""
        # cosine similarity to all vocab embeddings
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.embedding_weight, dim=-1)
        logits = torch.einsum("bld,vd->blv", x_norm, w_norm) / self.temperature
        return logits

    def hard_round(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) → token_ids: (B, L)  — non-differentiable, used at final step."""
        logits = self.soft_round(x)
        return logits.argmax(dim=-1)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if hard:
            return self.hard_round(x)
        return self.soft_round(x)
