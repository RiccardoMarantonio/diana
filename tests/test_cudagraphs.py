import pytest
import torch

from diana.train import build_ddpm
from diana.utils.amp import GradScaler
from diana.utils.cudagraphs import CudaGraphStep
from tests.test_train import _cfg


class TestLossFrom:
    def test_matches_manual_math(self):
        cfg = _cfg()
        model = build_ddpm(cfg)
        model.eval()  # disable dropout so the two forwards share masks
        x0 = torch.randn(4, 3, 8, 8)
        t = torch.full((4,), 5, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = model.q_sample(x0, t, noise)
        expected = torch.nn.functional.mse_loss(model.unet(x_t, t), noise)
        assert torch.allclose(model.loss_from(x0, t, noise), expected)


class TestPassthrough:
    def _make(self):
        cfg = _cfg()
        model = build_ddpm(cfg)
        model.train()
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        return cfg, model, scaler

    def test_disabled_flag_is_passthrough(self):
        _, model, scaler = self._make()
        graph = CudaGraphStep(model, torch.device("cpu"), scaler, enabled=False)
        assert not graph.enabled
        assert not graph.active
        x = torch.randn(2, 3, 8, 8)
        loss = graph.step(x)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert all(p.grad is not None for p in model.parameters())

    def test_enabled_flag_never_arms_off_cuda(self):
        _, model, scaler = self._make()
        graph = CudaGraphStep(model, torch.device("cpu"), scaler, enabled=True)
        assert not graph.enabled  # cuda guard wins over the flag

class TestGraphedScaler:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs real CUDA (first runs on the T4)")
    def test_replay_reads_the_live_loss_scale(self):
        """The graph bakes in the *address* of the scaler's scale tensor, not
        its value. torch updates it in place, so a replay after a scale change
        must produce proportionally scaled gradients."""
        device = torch.device("cuda")
        cfg = _cfg(dropout=0.0, grad_accum_steps=1, use_cuda_graphs=True)
        model = build_ddpm(cfg).to(device)
        model.train()
        scaler = GradScaler(device, enabled=True)
        graph = CudaGraphStep(model, device, scaler, enabled=True)
        x = torch.randn(4, 3, cfg.img_size, cfg.img_size, device=device)

        def grad_norm_at(scale: float) -> float:
            scaler._scaler.update(new_scale=scale)
            model.zero_grad(set_to_none=False)
            torch.manual_seed(0)  # identical timestep/noise draws every call
            graph.step(x)
            torch.cuda.synchronize()
            return float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm())

        torch.manual_seed(0)
        graph.step(x)  # primes + captures + replays once, creating the scale tensor
        assert graph.active
        high, low = grad_norm_at(1024.0), grad_norm_at(512.0)
        assert torch.isfinite(torch.tensor([high, low])).all()
        assert low / high == pytest.approx(0.5, rel=0.05)

    def test_config_rejects_graphs_with_accumulation(self):
        try:
            _cfg(use_cuda_graphs=True, grad_accum_steps=2)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for use_cuda_graphs with grad_accum_steps > 1")

def test_failed_capture_runs_clean_eager_backward(monkeypatch):
    cfg = _cfg(dropout=0.)
    model = build_ddpm(cfg)
    scaler = GradScaler(torch.device("cpu"), False)
    graph = CudaGraphStep(model, torch.device("cpu"), scaler, False)
    graph._enabled = True

    def fail(*args):
        for p in model.parameters():
            p.grad = torch.full_like(p, float("nan"))
        graph._enabled = False

    monkeypatch.setattr(graph, "_capture", fail)
    loss = graph.step(torch.randn(2, 3, 16, 16))
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not graph.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_graph_updates_weights_across_epochs(amp):
    from diana.models.ema import EMA
    from diana.train import build_optimizer, train_epoch
    cfg = _cfg(dropout=0., grad_accum_steps=1, use_cuda_graphs=True,
               use_amp=amp, lr_warmup_steps=0)
    device = torch.device("cuda")
    model = build_ddpm(cfg).to(device)
    scaler = GradScaler(device, amp)
    opt = build_optimizer(cfg, model)
    ema = EMA(model, .9)
    graph = CudaGraphStep(model, device, scaler, True)
    if amp:
        scaler._scaler.scale(torch.ones((), device=device))
        scaler._scaler.update(new_scale=1.)
    batches = [torch.randn(4, 3, 16, 16) for _ in range(2)]
    step = 0
    for epoch in range(2):
        before = next(model.parameters()).detach().clone()
        _, step = train_epoch(model, batches, opt, scaler, ema, cfg, epoch, step, graph)
        assert graph.active
        assert not torch.equal(before, next(model.parameters()))
