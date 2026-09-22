"""Exponential Moving Average (EMA) of model weights.

The EMA copy averages weights across training steps:
``shadow <- decay * shadow + (1 - decay) * param``.
Because the shadow starts at zero we bias-correct at extraction time
(``shadow / (1 - decay**steps)``), which removes the cold-start underestimation
visible in the first ~1/decay steps and makes early checkpoints honest.

Rules decided during design:
* ``update()`` is called once per *optimizer step* (not per diffusion ``t``),
  so the EMA sees the same cadence as the weights it averages.
* Buffers are excluded -- they are static (e.g. the schedule tables).

Convention: construct the EMA *after* moving the model to its final device.
"""

import torch
from torch import nn


class EMA:
    """Bias-corrected shadows for every parameter in ``model``."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        params = dict(model.named_parameters())
        if not params:
            raise ValueError("model has no parameters to shadow")
        self._decay = float(decay)
        self._device = next(iter(params.values())).device
        self._shadows: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param) for name, param in params.items()
        }
        self._steps = 0
        self._saved: dict[str, torch.Tensor] | None = None

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def steps(self) -> int:
        return self._steps

    def update(self, model: nn.Module) -> None:
        """Advance the averages toward ``model``'s current weights."""
        for name, param in model.named_parameters():
            self._shadows[name].mul_(self._decay).add_(
                param.detach(), alpha=1.0 - self._decay
            )
        self._steps += 1

    def _correction(self) -> float:
        """``1 - decay**steps``: normalizes the cold-start weighted sum."""
        return 1.0 - self._decay**self._steps

    def apply(self, model: nn.Module) -> None:
        """Write the bias-corrected shadows into ``model``'s parameters.

        Saves the live weights so :meth:`restore` can put them back. Use this
        around sampling/inference only -- never during training.
        """
        if self._steps == 0:
            raise RuntimeError("EMA.apply() called before any update()")
        self._saved = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        factor = self._correction()
        for name, param in model.named_parameters():
            param.data.copy_(self._shadows[name] / factor)

    def restore(self, model: nn.Module) -> None:
        """Put the pre-:meth:`apply` weights back into ``model``."""
        if self._saved is None:
            raise RuntimeError("EMA.restore() called without a matching apply()")
        for name, param in model.named_parameters():
            param.data.copy_(self._saved[name])
        self._saved = None

    def state_dict(self) -> dict:
        return {
            "decay": self._decay,
            "steps": self._steps,
            "shadows": {k: v.detach().cpu() for k, v in self._shadows.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._decay = float(state["decay"])
        self._steps = int(state["steps"])
        self._shadows = {
            k: v.to(self._device) for k, v in state["shadows"].items()
        }