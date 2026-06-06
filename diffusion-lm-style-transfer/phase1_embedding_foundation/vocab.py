from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


class Vocabulary:
    """Wraps a HuggingFace tokenizer and exposes vocab size and special token ids."""

    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def mask_id(self) -> Optional[int]:
        return self.tokenizer.mask_token_id

    def encode(self, text: str, max_length: int = 64, padding: bool = True) -> dict:
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=True,
            return_tensors="pt",
        )

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "Vocabulary":
        obj = cls.__new__(cls)
        obj.tokenizer = AutoTokenizer.from_pretrained(str(path))
        return obj
