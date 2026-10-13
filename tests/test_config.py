import pytest

from diana.config import Config, load_toml_config, parse_args


def _write_toml(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.toml"
    path.write_text(content)
    return str(path)


class TestDefaults:
    def test_requires_data_path(self):
        with pytest.raises(ValueError, match="data_path"):
            parse_args([])

    def test_cli_data_path_only(self):
        cfg = parse_args(["--data_path", "/tmp/foo"])
        assert cfg.data_path == "/tmp/foo"
        assert cfg.img_size == 64
        assert cfg.base_channels == 64
        assert cfg.channel_mults == [1, 2, 4, 8]
        assert cfg.device == "auto"
        assert cfg.use_amp is False

    def test_cli_overrides_defaults(self):
        cfg = parse_args(
            ["--data_path", "/tmp/foo", "--img_size", "128", "--base_channels", "128"]
        )
        assert cfg.img_size == 128
        assert cfg.base_channels == 128


class TestObjective:
    def test_only_the_implemented_objective_is_accepted(self):
        assert Config(data_path="x", objective="pred_noise").objective == "pred_noise"
        for unimplemented in ("pred_x0", "pred_v"):
            with pytest.raises(ValueError, match="objective"):
                Config(data_path="x", objective=unimplemented)

    def test_cli_rejects_unimplemented_objective(self):
        with pytest.raises(SystemExit):
            parse_args(["--data_path", "x", "--objective", "pred_v"])


class TestToml:
    def test_toml_supplies_values(self, tmp_path):
        path = _write_toml(
            tmp_path,
            'data_path = "/data/hazelnut"\nimg_size = 32\nchannel_mults = [1, 2, 4]\n',
        )
        cfg = parse_args(["--config", path])
        assert cfg.data_path == "/data/hazelnut"
        assert cfg.img_size == 32
        assert cfg.channel_mults == [1, 2, 4]
        assert cfg.device == "auto"

    def test_cli_overrides_toml(self, tmp_path):
        path = _write_toml(
            tmp_path, 'data_path = "/d"\nimg_size = 256\nbeta_end = 0.5\n'
        )
        cfg = parse_args(["--config", path, "--img_size", "128"])
        assert cfg.img_size == 128
        assert cfg.beta_end == 0.5
        assert cfg.data_path == "/d"

    def test_toml_bool_flags(self, tmp_path):
        path = _write_toml(tmp_path, "use_amp = true\npin_memory = true\n")
        cfg = parse_args(["--config", path, "--data_path", "/x"])
        assert cfg.use_amp is True
        assert cfg.pin_memory is True

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            parse_args(["--config", str(tmp_path / "nope.toml")])

    def test_unknown_key(self, tmp_path):
        path = _write_toml(tmp_path, "bogus_key = 1\n")
        with pytest.raises(ValueError, match="bogus_key"):
            parse_args(["--config", path])

    def test_invalid_value_still_validated(self, tmp_path):
        path = _write_toml(tmp_path, 'data_path = "/d"\nbeta_end = 1.5\n')
        with pytest.raises(ValueError, match="beta_end"):
            parse_args(["--config", path])


class TestLoadTomlConfig:
    def test_loads_and_rejects_unknown(self, tmp_path):
        path = _write_toml(tmp_path, "img_size = 16\n")
        assert load_toml_config(path) == {"img_size": 16}

def test_resume_supplies_config_without_repeating_flags(tmp_path):
    import dataclasses

    import torch
    cfg = Config(data_path="synthetic", category="saved", img_size=32)
    torch.save({"config": dataclasses.asdict(cfg)}, tmp_path / "last.pt")
    restored = parse_args(["--resume_from", str(tmp_path), "--epochs", "200"])
    assert restored.category == "saved"
    assert restored.img_size == 32
    assert restored.data_path == "synthetic"
    assert restored.epochs == 200
