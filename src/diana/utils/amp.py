"""Automatic Mixed Precision (AMP) helpers.

Two pieces with a uniform interface across devices:

* :func:`autocast` -- fp16 autocast context (active on CUDA, no-op elsewhere).
* :class:`GradScaler` -- loss scaling wrapper that *enforces* the mandated
  update ordering ``backward -> clip_grad_norm_ -> step -> update`` regardless
  of device, so wrong-order calls fail loudly even on the MPS dev box where
  scaling is a no-op.
"""

import contextlib
from collections.abc import Iterator

import torch
from torch.amp import GradScaler as _TorchGradScaler


@contextlib.contextmanager
def autocast(device: torch.device, enabled: bool = True) -> Iterator[None]:
    """Run a block under fp16 autocast on CUDA; no-op otherwise.

    ``enabled`` is the user's ``use_amp`` flag. If it is requested on a
    non-CUDA device we stay silent and run fp32 -- the caller should surface
    the effective state (``scaler.enabled``) in its startup log so a
    misconfiguration is visible rather than silently degrading.
    """
    active = enabled and device.type == "cuda"
    if active:
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield


class GradScaler:
    """Loss scaler with an ordering-enforcing step.

    On CUDA this wraps :class:`torch.amp.GradScaler` for fp16 training; on any
    other device it is a pure pass-through so ``train.py`` needs no device
    branches. Either way, :meth:`step` refuses to run before
    :meth:`clip_grad_norm_`, guarding against the classic AMP footgun of
    clipping already-scaled (huge) gradients.
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self._device = device
        self._enabled = enabled and device.type == "cuda"
        # Build the native scaler unconditionally (its methods become no-ops
        # when disabled) so attribute types are well-defined for checkers.
        self._scaler = _TorchGradScaler("cuda", enabled=self._enabled)
        self._unscaled = False

    @property
    def enabled(self) -> bool:
        """True when loss scaling is actually active (CUDA + use_amp)."""
        return self._enabled

    def backward(self, loss: torch.Tensor) -> None:
        """Apply AMP loss scaling, then backprop."""
        self._scaler.scale(loss).backward()

    def clip_grad_norm_(self, optimizer: torch.optim.Optimizer, max_norm: float) -> float:
        """Unscale (CUDA) then clip by total norm. Must precede :meth:`step`."""
        self._scaler.unscale_(optimizer)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self._unscaled = True
        return float(norm)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Run the optimizer step, refusing to fire before clipping.

        Returns True when the optimizer actually stepped (False when AMP
        skipped the step due to inf/nan grads).
        """
        if not self._unscaled:
            raise RuntimeError(
                "GradScaler.step() called before clip_grad_norm_(); expected "
                "backward -> clip_grad_norm_ -> step ordering"
            )
        self._unscaled = False
        if self._enabled:
            return bool(self._scaler.step(optimizer))
        self._scaler.step(optimizer)
        return True

    def update(self) -> None:
        """Adjust the scale factor after each optimizer step (CUDA only)."""
        self._scaler.update()

    def state_dict(self) -> dict:
        return self._scaler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._scaler.load_state_dict(state)