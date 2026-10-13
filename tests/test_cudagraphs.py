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