import itertools

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


def test_respaced_alpha_is_product_of_skipped_alphas(ddpm):
    """A strided hop must remove the noise of EVERY skipped step."""
    alphas = ddpm.schedule.alphas
    alpha, beta = ddpm._respaced_step(40, 30)
    expected = alphas[31:41].prod()  # steps 40 .. 31 are crossed going 40 -> 30
    assert torch.allclose(alpha, expected, rtol=1e-4)
    assert torch.allclose(beta, 1.0 - expected, atol=1e-5)


def test_respaced_unit_hop_matches_raw_schedule(ddpm):
    for t in (1, 17, 99):
        alpha, beta = ddpm._respaced_step(t, t - 1)
        assert torch.allclose(alpha, ddpm.schedule.alphas[t], rtol=1e-4)
        assert torch.allclose(beta, ddpm.schedule.betas[t], atol=1e-5)


def test_respaced_final_hop_reaches_clean_image(ddpm):
    alpha, _ = ddpm._respaced_step(0, None)
    assert torch.allclose(alpha, ddpm.schedule.alphas[0], rtol=1e-4)


def test_full_chain_matches_per_index_sampler(ddpm):
    """Stride 1 must reproduce the original (unit-step) ancestral sampler."""

    def legacy(x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        s = ddpm.schedule
        for i, t in enumerate(times.tolist()):
            eps = ddpm.unet(x, torch.full((x.shape[0],), t, dtype=torch.long))
            mean = (x - s.betas[t] / torch.sqrt(1.0 - s.alphas_cumprod[t]) * eps) / torch.sqrt(s.alphas[t])
            x = mean + torch.sqrt(s.betas[t]) * torch.randn_like(x) if i < len(times) - 1 else mean
        return x

    times = ddpm._stride_times(None)  # every index: stride 1
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.manual_seed(7)
        new = ddpm._reverse(x.clone(), times)
        torch.manual_seed(7)
        old = legacy(x.clone(), times)
    assert torch.allclose(new, old, rtol=1e-3, atol=1e-3)


def test_reconstruct_first_label_is_t_start(ddpm):
    """t_start is off the stride grid (99, 94, ...); the first call must still
    be labeled t_start, since x was noised to exactly that level."""
    labels: list[int] = []
    handle = ddpm.unet.register_forward_hook(lambda _m, args, _out: labels.append(int(args[1][0])))
    try:
        ddpm.reconstruct(torch.randn(1, 3, 32, 32), num_steps=20, t_start=32)
    finally:
        handle.remove()
    assert labels[0] == 32
    assert all(a > b for a, b in itertools.pairwise(labels)), "labels must descend"
    assert labels[-1] == 0


def test_objective_guard():
    with pytest.raises(NotImplementedError):
        DDPM(None, None, objective="pred_x0")
