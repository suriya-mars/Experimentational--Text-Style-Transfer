from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from phase1_embedding_foundation import EmbeddingLayer, RoundingLayer
from phase4_style_conditioning import StyleEncoder
from .ddim import DDIMSampler
from .sdedit import SDEditTransfer


@dataclass
class InferenceConfig:
    mode: Literal["partial", "full"] = "partial"
    t_frac: float = 0.5           # used in partial mode
    ddim_steps: int = 50
    guidance_scale: float = 3.0
    guidance_type: Literal["classifier", "cfg"] = "cfg"


class StyleTransferPipeline:
    """
    End-to-end inference pipeline: text → text.
    Wraps embedding, style encoding, sampling, and rounding.
    """

    def __init__(
        self,
        embedding: EmbeddingLayer,
        rounding: RoundingLayer,
        style_encoder: StyleEncoder,
        ddim_sampler: DDIMSampler,
        sdedit: SDEditTransfer,
        vocab,
        cfg=None,
        guidance=None,
    ):
        self.embedding = embedding
        self.rounding = rounding
        self.style_encoder = style_encoder
        self.ddim = ddim_sampler
        self.sdedit = sdedit
        self.vocab = vocab
        self.cfg = cfg          # CFGWrapper
        self.guidance = guidance  # ClassifierGuidance

    @torch.no_grad()
    def transfer(
        self,
        source_texts: list[str],
        target_style_texts: list[str],
        config: InferenceConfig | None = None,
        device: str = "cuda",
    ) -> list[str]:
        cfg = config or InferenceConfig()
        device_ = torch.device(device)

        # 1. Encode source to x_0
        enc = self.vocab.encode(source_texts[0], padding=True)
        src_ids = enc["input_ids"].to(device_)
        x0_src = self.embedding(src_ids)

        # 2. Encode target style
        style_cond = self.style_encoder.encode_text(target_style_texts, device=device)

        # 3. Build predict_x0_fn with guidance
        def predict_x0_fn(xt, t, style_cond):
            if self.cfg is not None:
                return self.cfg.guided_predict(xt, t, style_cond, cfg.guidance_scale)
            return self.sdedit.ddim.fwd  # fallback — override in subclass

        # 4. Sample
        if cfg.mode == "partial":
            x0_hat = self.sdedit.transfer(x0_src, predict_x0_fn, style_cond, cfg.t_frac)
        else:
            x0_hat = self.ddim.sample(predict_x0_fn, x0_src.shape, device_, style_cond)

        # 5. Round to tokens and decode
        token_ids = self.rounding.hard_round(x0_hat)
        return [self.vocab.decode(ids) for ids in token_ids]
