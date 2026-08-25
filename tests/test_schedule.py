import pytest
import torch

from diana.diffusion.schedule import DiffusionSchedule


@pytest.fixture(params=["linear", "cosine"])
def schedule(request) -> DiffusionSchedule:
    return DiffusionSchedule(
        schedule_type=request.param,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_param=0.008,
    )


def test_alpha_bar_monotonically_decreasing(schedule):
    abar = schedule.alphas_cumprod
    assert torch.all(abar[1:] <= abar[:-1]), "alpha_bar must never increase"


def test_betas_within_clamped_bounds(schedule):
    betas = schedule.betas
    assert torch.all(betas >= 0.0), "negative beta poisons the cumprod"
    assert torch.all(betas <= 0.999), "beta >= 1 makes alpha <= 0 -> sqrt(negative)"


def test_linear_and_cosine_differ():
    linear = DiffusionSchedule("linear", 1000, 1e-4, 0.02, 0.008)
    cosine = DiffusionSchedule("cosine", 1000, 1e-4, 0.02, 0.008)
    assert not torch.allclose(linear.alphas_cumprod, cosine.alphas_cumprod)


def test_cosine_terminal_alpha_bar_small_but_positive():
    s = DiffusionSchedule("cosine", 1000, 1e-4, 0.02, 0.008)
    terminal = s.alphas_cumprod[-1].item()
    assert 0.0 < terminal < 1e-3, f"expected tiny positive alpha_bar_T, got {terminal}"


def test_tables_are_float32_buffers():
    s = DiffusionSchedule("linear", 100, 1e-4, 0.02, 0.008)
    for name in ("betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
        buf = getattr(s, name)
        assert buf.dtype == torch.float32
        assert name in dict(s.named_buffers())
        assert name not in dict(s.named_parameters()), "schedule tables must never train"


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        DiffusionSchedule("quadratic", 100, 1e-4, 0.02, 0.008)
