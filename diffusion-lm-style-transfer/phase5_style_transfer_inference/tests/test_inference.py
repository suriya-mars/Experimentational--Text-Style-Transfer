import pytest
import torch
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule
from phase5_style_transfer_inference import DDIMSampler, SDEditTransfer


T, BATCH, SEQ, DIM = 100, 2, 8, 32


@pytest.fixture
def fwd_diff():
    return ForwardDiffusion(cosine_schedule(T))


@pytest.fixture
def ddim(fwd_diff):
    return DDIMSampler(fwd_diff, num_steps=10)


@pytest.fixture
def sdedit(fwd_diff, ddim):
    return SDEditTransfer(fwd_diff, ddim)


def _dummy_predict(xt, t, style_cond):
    return torch.zeros_like(xt)


def test_ddim_sample_shape(ddim):
    shape = (BATCH, SEQ, DIM)
    result = ddim.sample(_dummy_predict, shape, device=torch.device("cpu"))
    assert result.shape == shape


def test_sdedit_output_shape(sdedit):
    x0_src = torch.randn(BATCH, SEQ, DIM)
    style_cond = torch.randn(BATCH, 64)
    result = sdedit.transfer(x0_src, _dummy_predict, style_cond, t_frac=0.4)
    assert result.shape == x0_src.shape


def test_partial_vs_full_differ(fwd_diff, ddim, sdedit):
    x0_src = torch.randn(BATCH, SEQ, DIM)
    style_cond = torch.randn(BATCH, 64)
    low = sdedit.transfer(x0_src, _dummy_predict, style_cond, t_frac=0.2)
    high = sdedit.transfer(x0_src, _dummy_predict, style_cond, t_frac=0.8)
    assert not torch.allclose(low, high)
