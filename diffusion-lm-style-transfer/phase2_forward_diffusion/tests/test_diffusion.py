import pytest
import torch
from phase2_forward_diffusion import ForwardDiffusion, cosine_schedule, linear_schedule


T, BATCH, SEQ, DIM = 200, 4, 16, 64


@pytest.fixture(params=["cosine", "linear"])
def forward_diff(request):
    schedule = cosine_schedule(T) if request.param == "cosine" else linear_schedule(T)
    return ForwardDiffusion(schedule)


def test_q_sample_shape(forward_diff):
    x0 = torch.randn(BATCH, SEQ, DIM)
    t = torch.randint(0, T, (BATCH,))
    xt = forward_diff.q_sample(x0, t)
    assert xt.shape == x0.shape


def test_noise_increases_with_t(forward_diff):
    x0 = torch.zeros(1, SEQ, DIM)
    t_low = torch.tensor([10])
    t_high = torch.tensor([T - 10])
    xt_low = forward_diff.q_sample(x0, t_low)
    xt_high = forward_diff.q_sample(x0, t_high)
    assert xt_high.var() > xt_low.var()


def test_cosine_betas_monotone():
    s = cosine_schedule(T)
    assert (s.betas[1:] >= s.betas[:-1]).all(), "cosine betas should be non-decreasing"


def test_schedule_values_in_range():
    s = cosine_schedule(T)
    assert (s.betas >= 0).all() and (s.betas <= 1).all()
    assert (s.alphas_cumprod >= 0).all() and (s.alphas_cumprod <= 1).all()
