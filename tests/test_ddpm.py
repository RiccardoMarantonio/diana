import pytest
import torch

from diana.diffusion.ddpm import DDPM
from diana.diffusion.schedule import DiffusionSchedule
from diana.models.unet import UNet


@pytest.fixture(scope="module")
def ddpm() -> DDPM:
    torch.manual_seed(0)
    # T=100 with paper-range betas would NOT reach pure noise (sum(beta)
    # must be >> 1 for abar_T -> 0; that's WHY DDPM uses T=1000). Scale the
    # range up so the terminal step genuinely saturates -- keeps tests fast.
    schedule = DiffusionSchedule("linear", 100, 1e-3, 0.1, 0.008)
    unet = UNet(
        img_size=32,
        in_channels=3,
        base_channels=16,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attention_resolutions=[8],
        dropout=0.0,
    )
    return DDPM(unet, schedule)


def test_q_sample_shape_and_endpoints(ddpm):
    x0 = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(x0)

    t_clean = torch.zeros(4, dtype=torch.long)  # abar ~ 1: mostly signal
    t_pure = torch.full((4,), 99, dtype=torch.long)  # abar ~ 0: mostly noise

    xt_clean = ddpm.q_sample(x0, t_clean, noise)
    xt_pure = ddpm.q_sample(x0, t_pure, noise)

    assert xt_clean.shape == x0.shape
    # At t=0 signal dominates; at t=T-1 noise dominates.
    corr_clean = torch.corrcoef(torch.stack([x0.flatten(), xt_clean.flatten()]))[0, 1]
    corr_pure = torch.corrcoef(torch.stack([x0.flatten(), xt_pure.flatten()]))[0, 1]
    assert corr_clean > 0.95
    assert abs(corr_pure) < 0.3


def test_loss_is_scalar_finite(ddpm):
    x0 = torch.randn(2, 3, 32, 32)
    loss = ddpm.loss(x0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in ddpm.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_stride_times_descending_within_bounds(ddpm):
    times = ddpm._stride_times(25)
    assert len(times) == 25
    assert torch.all(times[1:] < times[:-1]), "reverse chain must descend"
    assert times[0] == 99 and times[-1] == 0


def test_sample_shapes(ddpm):
    out = ddpm.sample(n_samples=2, img_size=32, num_steps=10)
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_reconstruct_partial_chain_preserves_structure(ddpm):
    torch.manual_seed(42)
    x0 = torch.randn(1, 3, 32, 32)
    rec = ddpm.reconstruct(x0, num_steps=20, t_start=30)
    assert rec.shape == x0.shape

    # Reconstruction from a LOW start should stay far closer to the input
    # than pure generation is.
    rec_low = ddpm.reconstruct(x0, num_steps=20, t_start=5)
    dist_low = (rec_low - x0).pow(2).mean().item()
    dist_gen = (ddpm.sample(1, 32, num_steps=10) - x0).pow(2).mean().item()
    assert dist_low < dist_gen


def test_objective_guard():
    with pytest.raises(NotImplementedError):
        DDPM(None, None, objective="pred_x0")
