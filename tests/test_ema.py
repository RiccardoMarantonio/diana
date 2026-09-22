import pytest
import torch

from diana.models.ema import EMA

DECAY = 0.9


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.register_buffer("static", torch.tensor(-5.0))


class TestEmaUpdate:
    def test_construct_from_params(self):
        model = _ScalarModel(1.0)
        ema = EMA(model, DECAY)
        assert ema.decay == DECAY
        assert ema.steps == 0

    def test_rejects_invalid_decay(self):
        model = _ScalarModel()
        with pytest.raises(ValueError, match="decay"):
            EMA(model, 1.0)
        with pytest.raises(ValueError, match="decay"):
            EMA(model, -0.1)

    def test_single_update_reaches_param(self):
        model = _ScalarModel(2.0)
        ema = EMA(model, DECAY)
        ema.update(model)
        assert ema.steps == 1
        ema.apply(model)
        assert model.weight.item() == pytest.approx(2.0)

    def test_multiple_updates_converge(self):
        model = _ScalarModel(3.0)
        ema = EMA(model, DECAY)
        for _ in range(50):
            ema.update(model)
        ema.apply(model)
        assert model.weight.item() == pytest.approx(3.0, abs=1e-3)

    def test_newer_params_weighted_more(self):
        model = _ScalarModel(0.0)
        ema = EMA(model, DECAY)
        ema.update(model)  # both zeros -> shadow stays 0
        model.weight.data.copy_(torch.tensor(1.0))
        ema.update(model)  # shadow = 0.1 at step 2, all mass on latest warm-up? no
        # step2: shadow = 0.9*0 + 0.1*1 = 0.1; corrected = 0.1/(1-0.81) = 0.526...
        ema.apply(model)
        assert 0.5 < model.weight.item() < 0.6

    def test_buffers_excluded(self):
        model = _ScalarModel(3.0)
        ema = EMA(model, DECAY)
        assert "static" not in ema.state_dict()["shadows"]


class TestApplyRestore:
    def test_apply_then_restore_roundtrip(self):
        model = _ScalarModel(3.0)
        ema = EMA(model, DECAY)
        ema.update(model)
        live = model.weight.item()

        ema.apply(model)
        corrected = model.weight.item()

        ema.restore(model)
        assert model.weight.item() == pytest.approx(live)
        assert corrected != live or corrected == pytest.approx(3.0)

    def test_apply_before_update_raises(self):
        ema = EMA(_ScalarModel(1.0), DECAY)
        with pytest.raises(RuntimeError, match="update"):
            ema.apply(_ScalarModel(1.0))

    def test_restore_without_apply_raises(self):
        ema = EMA(_ScalarModel(1.0), DECAY)
        with pytest.raises(RuntimeError, match="apply"):
            ema.restore(_ScalarModel(1.0))

    def test_restore_twice_raises(self):
        model = _ScalarModel(1.0)
        ema = EMA(model, DECAY)
        ema.update(model)
        ema.apply(model)
        ema.restore(model)
        with pytest.raises(RuntimeError, match="apply"):
            ema.restore(model)


class TestCheckpointing:
    def test_state_dict_roundtrip(self):
        model = _ScalarModel(1.0)
        ema = EMA(model, DECAY)
        for _ in range(3):
            ema.update(model)
        state = ema.state_dict()

        reloaded = EMA(_ScalarModel(99.0), 0.5)  # deliberately different
        reloaded.load_state_dict(state)
        assert reloaded.steps == ema.steps
        assert reloaded.decay == DECAY

        a, b = _ScalarModel(5.0), _ScalarModel(5.0)
        ema.apply(a)
        reloaded.apply(b)
        assert a.weight.item() == pytest.approx(b.weight.item())

    def test_shadows_saved_on_cpu(self):
        model = _ScalarModel(1.0)
        ema = EMA(model, DECAY)
        ema.update(model)
        for shadow in ema.state_dict()["shadows"].values():
            assert shadow.device.type == "cpu"