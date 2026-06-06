import pytest
import torch
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule
from phase3_denoising_network import Denoiser, DenoisingTransformer


T, BATCH, SEQ, DIM = 200, 2, 16, 64
STYLE_DIM = 128


@pytest.fixture
def denoiser():
    schedule = cosine_schedule(T)
    fwd = ForwardDiffusion(schedule)
    transformer = DenoisingTransformer(embed_dim=DIM, n_layers=2, n_heads=4, style_dim=STYLE_DIM)
    return Denoiser(transformer, fwd)


def test_denoiser_loss_is_positive(denoiser):
    x0 = torch.randn(BATCH, SEQ, DIM)
    t = torch.randint(0, T, (BATCH,))
    out = denoiser(x0, t)
    assert out["loss"].item() > 0


def test_denoiser_x0_pred_shape(denoiser):
    x0 = torch.randn(BATCH, SEQ, DIM)
    t = torch.randint(0, T, (BATCH,))
    out = denoiser(x0, t)
    assert out["x0_pred"].shape == x0.shape


def test_denoiser_with_style_cond(denoiser):
    x0 = torch.randn(BATCH, SEQ, DIM)
    t = torch.randint(0, T, (BATCH,))
    style = torch.randn(BATCH, STYLE_DIM)
    out = denoiser(x0, t, style_cond=style)
    assert out["loss"].item() > 0
    assert out["x0_pred"].shape == x0.shape


def test_predict_x0_no_grad(denoiser):
    xt = torch.randn(BATCH, SEQ, DIM)
    t = torch.randint(0, T, (BATCH,))
    x0_pred = denoiser.predict_x0(xt, t)
    assert x0_pred.shape == xt.shape
