"""Stage 3: Fine-tune denoiser with style conditioning and consistency loss."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1_embedding_foundation import EmbeddingLayer
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule
from phase3_denoising_network import Denoiser
from phase4_style_conditioning import StyleClassifier, CFGWrapper
from .losses import DiffusionLoss


def finetune(
    data_loader: DataLoader,
    embedding: EmbeddingLayer,
    denoiser: Denoiser,
    classifier: StyleClassifier,
    *,
    T: int = 2000,
    lr: float = 5e-5,
    lambda_style: float = 0.1,
    p_uncond: float = 0.1,
    max_steps: int = 50_000,
    save_dir: str = "./checkpoints",
    device: str = "cuda",
    log_every: int = 100,
):
    dev = torch.device(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    schedule = cosine_schedule(T)
    fwd_diff = ForwardDiffusion(schedule)
    cfg_wrapper = CFGWrapper(denoiser, p_uncond=p_uncond).to(dev)
    embedding = embedding.to(dev)
    classifier = classifier.to(dev).requires_grad_(False)

    loss_fn = DiffusionLoss(lambda_style=lambda_style)
    optimizer = AdamW(
        list(cfg_wrapper.parameters()) + list(embedding.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    step = 0
    for batch in tqdm(data_loader, total=max_steps, desc="Finetune"):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(dev)
        style_labels = batch["labels"].to(dev)
        style_cond = batch["style_emb"].to(dev)

        x0 = embedding(input_ids)
        t = torch.randint(0, T, (x0.shape[0],), device=dev)

        out = cfg_wrapper(x0, t, style_cond)
        embed_loss = embedding.embedding_loss(out["x0_pred"], input_ids)
        style_logits = classifier(out["x0_pred"])

        losses = loss_fn(
            out["x0_pred"],
            x0.detach(),
            embed_loss=embed_loss,
            style_logits=style_logits,
            style_labels=style_labels,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(cfg_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % log_every == 0:
            tqdm.write(
                f"step={step}  total={losses['total'].item():.4f}  "
                f"l_simple={losses['l_simple'].item():.4f}  "
                f"l_style={losses.get('l_style', torch.tensor(0.)).item():.4f}"
            )

    torch.save(
        {"cfg_wrapper": cfg_wrapper.state_dict(), "embedding": embedding.state_dict()},
        save_path / "finetune_final.pt",
    )
    return cfg_wrapper, embedding
