"""Device resolution, backend tuning, and reproducible seeding.

Every global torch/random side effect lives here so the rest of the codebase
stays pure. Convention: call :func:`resolve_device` + :func:`set_seed` exactly
once at program start, *before* building the model or dataloaders.
"""

import random
from collections.abc import Callable

import torch

from diana.config import VALID_DEVICES

WorkerInitFn = Callable[[int], None]


def resolve_device(device: str) -> torch.device:
    """Resolve the ``auto`` shorthand to the best available accelerator.

    Preference order: CUDA > MPS (Apple Silicon) > CPU. An *explicit* request
    for an unavailable device raises, so a Colab run can never silently fall
    back to CPU and waste a session.
    """
    if device not in VALID_DEVICES:
        raise ValueError(
            f"device must be one of {sorted(VALID_DEVICES)}, got {device!r}"
        )
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested device 'cuda' but it is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested device 'mps' but it is not available")
    return resolved


def configure_backends(device: torch.device, benchmark: bool = False) -> None:
    """Tune cuDNN for the resolved device. No-op on non-CUDA devices."""
    if device.type != "cuda":
        return
    # Spec-mandated HPC switch: auto-selects kernels per input shape.
    torch.backends.cudnn.benchmark = benchmark


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, torch (CPU + all GPUs) RNG with a shared seed.

    With ``deterministic=True`` also pins cuDNN to deterministic algorithms,
    which disables benchmark mode -- used for exact-repro runs only.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed a dataloader ``worker_init_fn`` from the torch base seed.

    ``torch.initial_seed()`` inside a worker already derives a unique seed per
    worker from the global base seed, so back-to-back runs are reproducible
    while workers still draw non-identical streams.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)