from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Learnable word embedding that maps discrete token ids to a continuous space
    used as x_0 for the diffusion process.

    L_embed = ||EMB(w) - sg(x_0)||^2 anchors the embedding matrix to the
    diffusion manifold during joint training.
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, L) → x_0: (B, L, d)"""
        return self.embedding(token_ids)

    def embedding_loss(self, x0_pred: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Anchor loss: pulls embedding weights toward the diffusion manifold."""
        x0_embed = self.embedding(token_ids)
        return torch.mean((x0_embed - x0_pred.detach()) ** 2)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight
