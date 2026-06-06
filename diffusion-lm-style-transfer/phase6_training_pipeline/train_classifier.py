"""Stage 2: Train style classifier on noised embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1_embedding_foundation import EmbeddingLayer
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule
from phase4_style_conditioning import StyleClassifier


def train_classifier(
    data_loader: DataLoader,
    embedding: EmbeddingLayer,
    *,
    embed_dim: int = 128,
    num_styles: int = 2,
    T: int = 2000,
    lr: float = 1e-4,
    max_steps: int = 20_000,
    save_dir: str = "./checkpoints",
    device: str = "cuda",
    log_every: int = 100,
):
    dev = torch.device(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    schedule = cosine_schedule(T)
    fwd_diff = ForwardDiffusion(schedule)
    classifier = StyleClassifier(embed_dim, num_styles).to(dev)
    embedding = embedding.to(dev).requires_grad_(False)

    optimizer = AdamW(classifier.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    step = 0
    for batch in tqdm(data_loader, total=max_steps, desc="Train Classifier"):
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(dev)
        style_labels = batch["labels"].to(dev)

        with torch.no_grad():
            x0 = embedding(input_ids)
            t = torch.randint(0, T, (x0.shape[0],), device=dev)
            xt = fwd_diff.q_sample(x0, t)

        logits = classifier(xt)
        loss = loss_fn(logits, style_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % log_every == 0:
            acc = (logits.argmax(-1) == style_labels).float().mean()
            tqdm.write(f"step={step}  loss={loss.item():.4f}  acc={acc.item():.3f}")

    torch.save(classifier.state_dict(), save_path / "classifier_final.pt")
    return classifier
