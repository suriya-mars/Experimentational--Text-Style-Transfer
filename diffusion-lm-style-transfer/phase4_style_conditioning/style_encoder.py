from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class StyleEncoder(nn.Module):
    """
    Encodes source text into a style-content vector using a frozen BERT backbone
    followed by a trainable projection head.
    """

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        out_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns CLS-pooled style vector: (B, out_dim)"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # CLS token
        return self.proj(cls)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        return self.forward(enc["input_ids"], enc["attention_mask"])
