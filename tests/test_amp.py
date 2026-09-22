import pytest
import torch

from diana.utils.amp import GradScaler, autocast


def _make_optimizer(model: torch.nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=0.1)


class TestAutocast:
    def test_cpu_is_fp32_noop(self):
        with autocast(torch.device("cpu"), enabled=True):
            out = torch.randn(8, 8) @ torch.randn(8, 8)
        assert out.dtype == torch.float32

    def test_disabled_is_fp32(self):
        with autocast(torch.device("cpu"), enabled=False):
            out = torch.randn(2, 2)
        assert out.dtype == torch.float32


class TestGradScalerDisabled:
    def _model(self):
        return torch.nn.Linear(4, 1)

    def test_not_enabled_on_cpu(self):
        scaler = GradScaler(torch.device("cpu"), enabled=True)
        assert scaler.enabled is False

    def test_backward_clip_step_roundtrip(self):
        model = self._model()
        opt = _make_optimizer(model)
        scaler = GradScaler(torch.device("cpu"), enabled=True)
        loss = (model(torch.randn(8, 4)) ** 2).mean()
        scaler.backward(loss)
        norm = scaler.clip_grad_norm_(opt, max_norm=1.0)
        assert isinstance(norm, float)
        assert scaler.step(opt) is True
        scaler.update()

    def test_step_before_clip_raises(self):
        model = self._model()
        opt = _make_optimizer(model)
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        scaler.backward((model(torch.randn(8, 4)) ** 2).mean())
        with pytest.raises(RuntimeError, match="step"):
            scaler.step(opt)

    def test_clip_limits_grad_norm(self):
        model = self._model()
        opt = _make_optimizer(model)
        model.weight.grad = torch.full_like(model.weight, 100.0)
        model.bias.grad = torch.zeros_like(model.bias)
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        # clip_grad_norm_ returns the PRE-clip total norm...
        assert scaler.clip_grad_norm_(opt, max_norm=0.5) == pytest.approx(200.0)
        # ...and rescales grads so the post-clip total norm hits max_norm.
        clipped = torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], float("inf")
        )
        assert clipped == pytest.approx(0.5)

    def test_state_dict_empty_when_disabled(self):
        scaler = GradScaler(torch.device("cpu"), enabled=True)
        assert scaler.state_dict() == {}


class TestGradScalerEnabled:
    def test_enabled_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("cuda not available")
        assert GradScaler(torch.device("cuda"), enabled=True).enabled is True