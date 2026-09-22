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

    def test_config_rejects_graphs_with_accumulation(self):
        try:
            _cfg(use_cuda_graphs=True, grad_accum_steps=2)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for use_cuda_graphs with grad_accum_steps > 1")