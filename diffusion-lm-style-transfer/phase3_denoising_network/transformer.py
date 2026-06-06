from __future__ import annotations

import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding


class TransformerLayer(nn.Module):
    """Single bidirectional transformer layer with time-step conditioning."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # inject timestep via additive bias
        x = x + time_emb.unsqueeze(1)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DenoisingTransformer(nn.Module):
    """
    Bidirectional transformer denoiser.
    Predicts x_0 directly (not noise ε) — enables vocabulary anchoring at each step.
    Style condition is injected via cross-attention in each layer.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
        style_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_emb = SinusoidalTimeEmbedding(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )

        # optional cross-attention for style conditioning
        self.style_dim = style_dim
        if style_dim is not None:
            self.style_proj = nn.Linear(style_dim, embed_dim)
            self.cross_attn_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
                    for _ in range(n_layers)
                ]
            )
            self.cross_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        style_cond: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        xt:         (B, L, d)  — noised embeddings
        t:          (B,)       — integer timesteps
        style_cond: (B, S, d_style) or (B, d_style) — style context
        Returns x_0_hat: (B, L, d)
        """
        time_emb = self.time_emb(t)  # (B, d)
        x = xt

        if style_cond is not None and style_cond.ndim == 2:
            style_cond = style_cond.unsqueeze(1)  # (B, 1, d_style)

        for i, layer in enumerate(self.layers):
            x = layer(x, time_emb, padding_mask)

            if self.style_dim is not None and style_cond is not None:
                style_kv = self.style_proj(style_cond)
                attn_out, _ = self.cross_attn_layers[i](x, style_kv, style_kv)
                x = self.cross_norms[i](x + attn_out)

        return self.output_proj(x)
