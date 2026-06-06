"""Stage 1: Unconditional diffusion LM pretraining on a large text corpus."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1_embedding_foundation import EmbeddingLayer, RoundingLayer, Vocabulary
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule
from phase3_denoising_network import Denoiser, DenoisingTransformer
from .losses import DiffusionLoss


def pretrain(
    data_loader: DataLoader,
    *,
    vocab_size: int = 30522,
    embed_dim: int = 128,
    n_layers: int = 12,
    n_heads: int = 8,
    T: int = 2000,
    lr: float = 1e-4,
    max_steps: int = 100_000,
    save_dir: str = "./checkpoints",
    device: str = "cuda",
    log_every: int = 100,
):
    dev = torch.device(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    embedding = EmbeddingLayer(vocab_size, embed_dim).to(dev)
    schedule = cosine_schedule(T)
    fwd_diff = ForwardDiffusion(schedule)
    transformer = DenoisingTransformer(embed_dim, n_layers, n_heads)
    denoiser = Denoiser(transformer, fwd_diff).to(dev)
    loss_fn = DiffusionLoss()

    optimizer = AdamW(
        list(embedding.parameters()) + list(denoiser.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    step = 0
    for batch in tqdm(data_loader, total=max_steps, desc="Pretrain"):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(dev)
        x0 = embedding(input_ids)
        t = torch.randint(0, T, (x0.shape[0],), device=dev)

        out = denoiser(x0, t)
        embed_loss = embedding.embedding_loss(out["x0_pred"], input_ids)
        losses = loss_fn(out["x0_pred"], x0.detach(), embed_loss=embed_loss)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % log_every == 0:
            tqdm.write(
                f"step={step}  loss={losses['total'].item():.4f}  "
                f"l_simple={losses['l_simple'].item():.4f}"
            )

    torch.save(
        {"embedding": embedding.state_dict(), "denoiser": denoiser.state_dict()},
        save_path / "pretrain_final.pt",
    )
    return embedding, denoiser


if __name__ == "__main__":
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def collate(batch):
        texts = [b["text"] for b in batch if b["text"].strip()]
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

    loader = DataLoader(ds, batch_size=32, collate_fn=collate, num_workers=4)
    pretrain(loader, max_steps=100_000)
