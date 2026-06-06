import pytest
import torch
from phase6_training_pipeline import DiffusionLoss


def test_loss_simple_only():
    loss_fn = DiffusionLoss()
    x0_pred = torch.randn(4, 16, 64)
    x0_true = torch.randn(4, 16, 64)
    losses = loss_fn(x0_pred, x0_true)
    assert "total" in losses
    assert "l_simple" in losses
    assert losses["total"].shape == ()


def test_loss_with_all_terms():
    loss_fn = DiffusionLoss(lambda_embed=0.1, lambda_style=0.5)
    x0_pred = torch.randn(4, 16, 64)
    x0_true = torch.randn(4, 16, 64)
    embed_loss = torch.tensor(0.5)
    style_logits = torch.randn(4, 2)
    style_labels = torch.randint(0, 2, (4,))
    losses = loss_fn(x0_pred, x0_true, embed_loss, style_logits, style_labels)
    assert "l_embed" in losses
    assert "l_style" in losses
    assert losses["total"].item() > 0


def test_loss_total_increases_with_lambda():
    loss_fn_low = DiffusionLoss(lambda_style=0.01)
    loss_fn_high = DiffusionLoss(lambda_style=10.0)
    x0_pred = torch.randn(4, 16, 64)
    x0_true = torch.randn(4, 16, 64)
    style_logits = torch.randn(4, 2)
    style_labels = torch.randint(0, 2, (4,))
    l_low = loss_fn_low(x0_pred, x0_true, style_logits=style_logits, style_labels=style_labels)
    l_high = loss_fn_high(x0_pred, x0_true, style_logits=style_logits, style_labels=style_labels)
    assert l_high["total"].item() > l_low["total"].item()
