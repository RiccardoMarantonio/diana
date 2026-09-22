import random

import pytest
import torch

from diana.utils.device import (
    configure_backends,
    resolve_device,
    set_seed,
    worker_init_fn,
)


class TestResolveDevice:
    def test_auto_resolves_to_available(self):
        device = resolve_device("auto")
        assert device.type in {"cuda", "mps", "cpu"}
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_explicit_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="device"):
            resolve_device("tpu")

    def test_missing_cuda_raises(self):
        if torch.cuda.is_available():
            pytest.skip("cuda available; cannot test the failure path")
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")

    def test_missing_mps_raises(self):
        if torch.backends.mps.is_available():
            pytest.skip("mps available; cannot test the failure path")
        with pytest.raises(RuntimeError, match="mps"):
            resolve_device("mps")


class TestConfigureBackends:
    def test_non_cuda_is_noop(self):
        configure_backends(torch.device("cpu"), benchmark=True)
        # nothing to assert beyond "it must not raise on a CPU-only box"

    def test_cuda_sets_benchmark(self):
        if not torch.cuda.is_available():
            pytest.skip("cuda not available")
        configure_backends(torch.device("cuda"), benchmark=True)
        assert torch.backends.cudnn.benchmark is True


class TestSetSeed:
    def test_reproducible_stream(self):
        set_seed(7)
        first = torch.rand(4)
        set_seed(7)
        second = torch.rand(4)
        assert torch.equal(first, second)

    def test_different_seeds_differ(self):
        set_seed(1)
        first = torch.rand(8)
        set_seed(2)
        second = torch.rand(8)
        assert not torch.equal(first, second)

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="seed"):
            set_seed(-1)


class TestWorkerInit:
    def test_runs_without_error(self):
        assert worker_init_fn(0) is None

    def test_different_workers_differ(self):
        random.seed(1234)  # pin a known starting state
        worker_init_fn(0)
        seed_a = random.random()
        worker_init_fn(1)
        seed_b = random.random()
        assert seed_a != seed_b