from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


class StyleAccuracy:
    """Measures what % of outputs match the target style via an external classifier."""

    def __init__(self, model_name: str = "textattack/bert-base-uncased-yelp-polarity"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], target_label: int) -> float:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        logits = self.model(**enc).logits
        preds = logits.argmax(-1)
        return (preds == target_label).float().mean().item()


class ContentPreservation:
    """BERTScore F1 between source and output texts (content similarity)."""

    def __call__(self, sources: list[str], outputs: list[str], lang: str = "en") -> float:
        _, _, f1 = bert_score(outputs, sources, lang=lang, verbose=False)
        return f1.mean().item()


class Fluency:
    """GPT-2 perplexity — lower is more fluent."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def perplexity(self, texts: list[str]) -> float:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        loss = self.model(input_ids=input_ids, labels=labels).loss
        return torch.exp(loss).item()


class SelfBLEU:
    """Diversity metric — lower self-BLEU means more diverse outputs."""

    def __call__(self, texts: list[str]) -> float:
        tokenized = [t.split() for t in texts]
        smoothie = SmoothingFunction().method1
        scores = []
        for i, hyp in enumerate(tokenized):
            refs = tokenized[:i] + tokenized[i + 1:]
            if refs:
                scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothie))
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvalResults:
    style_accuracy: float
    content_preservation: float
    fluency_perplexity: float
    self_bleu: float

    def __str__(self) -> str:
        return (
            f"Style Acc:    {self.style_accuracy:.3f}\n"
            f"Content (F1): {self.content_preservation:.3f}\n"
            f"Fluency (PPL):{self.fluency_perplexity:.1f}\n"
            f"Self-BLEU:    {self.self_bleu:.3f}"
        )
