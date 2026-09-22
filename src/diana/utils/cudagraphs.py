"""CUDA-graph acceleration for the training step (spec HPC requirement).

Transparent single-path wrapper around ``forward + backward``:

* **Off-CUDA / disabled** -- exact passthrough: ``model.loss(x)`` and
  ``scaler.backward``, so every other device runs the code that already has
  60+ green tests behind it.
* **CUDA + enabled** -- the first call for a batch shape primes gradient
  buffers, captures ``forward + scaled backward`` as one ``CUDAGraph``, and
  replays it on every later step. The stochastic parts (timestep draw, noise
  draw) happen *outside* capture into static buffers, so per-step randomness
  stays fresh; a captured ``randn`` would silently freeze training.

Everything else the optimizer needs stays outside the graph on purpose:
``clip_grad_norm_ -> step -> update`` mutate weights in place after replay,
and those in-place writes are visible to the *next* replay, so weights
continue to evolve normally.

Known trade-offs (made explicit rather than hidden):

* Dropout masks are frozen at capture time. For parity between graphed and
  ungraphed runs, use ``dropout = 0`` in the graphed config.
* Gradient accumulation is incompatible (a replay can't accumulate inten-
  tionally); ``Config`` enforces ``grad_accum_steps == 1``.
* A failed *capture* disables the wrapper and falls back to the eager path so
  a driver hiccup cannot kill a run. Replay failures are unrecoverable by
  design and surface as runtime errors.
"""

import torch

from diana.diffusion.ddpm import DDPM
from diana.utils.amp import GradScaler, autocast


class CudaGraphStep:
    """Graphed ``loss_from + scaler.backward`` for a fixed batch shape."""

    def __init__(
        self,
        model: DDPM,
        device: torch.device,
        scaler: GradScaler,
        enabled: bool,
    ):
        self._model = model
        self._device = device
        self._scaler = scaler
        self._enabled = enabled and device.type == "cuda"
        self._graph: torch.cuda.CUDAGraph | None = None
        self._buffers: dict[str, torch.Tensor] | None = None
        self._shape: tuple[int, ...] | None = None
        self._loss = torch.empty((), device=device)

    @property
    def enabled(self) -> bool:
        """True when CUDA capture was actually armed (cuda + flag)."""
        return self._enabled

    @property
    def active(self) -> bool:
        """True after a successful capture; the graph is being replayed."""
        return self._enabled and self._graph is not None

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Forward + backward the DDPM loss on a batch; returns the loss.

        On the graphed path the timestep and noise draws are made here on the
        current (autograd) stream, copied into static buffers, and consumed by
        the replayed graph -- randomness stays fresh, compute stays graphed.
        """
        if not self._enabled:
            with autocast(self._device, enabled=self._scaler.enabled):
                loss = self._model.loss(x)
            self._scaler.backward(loss)
            return loss.detach()

        batch = x.shape[0]
        num_steps = self._model.schedule.num_timesteps
        t = torch.randint(0, num_steps, (batch,), device=self._device, dtype=torch.long)
        noise = torch.randn_like(x)

        shape = tuple(x.shape)
        if self._graph is None or shape != self._shape:
            self._capture(x, t, noise, shape)

        assert self._buffers is not None and self._graph is not None
        self._buffers["x"].copy_(x, non_blocking=True)
        self._buffers["t"].copy_(t, non_blocking=True)
        self._buffers["noise"].copy_(noise, non_blocking=True)
        self._graph.replay()
        return self._loss.detach()

    # ------------------------------------------------------------------

    def _capture(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, shape: tuple[int, ...]) -> None:
        """Priming backward, static-buffer allocation, then one capture."""
        model = self._model
        try:
            model.zero_grad(set_to_none=False)
            with autocast(self._device, enabled=self._scaler.enabled):
                primed = model.loss_from(x, t, noise)
            self._scaler.backward(primed)
            model.zero_grad(set_to_none=False)  # keep grad buffers, zero values

            buffers = {
                "x": torch.empty_like(x),
                "t": torch.empty_like(t),
                "noise": torch.empty_like(noise),
            }
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with autocast(self._device, enabled=self._scaler.enabled), torch.cuda.graph(graph):
                loss = model.loss_from(buffers["x"], buffers["t"], buffers["noise"])
                self._loss.copy_(loss.detach())
                self._scaler.backward(loss)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - fall back, never kill a run
            self._enabled = False
            self._graph = None
            self._buffers = None
            print(f"[cudagraphs] capture failed ({exc}); falling back to eager steps")
            return

        self._buffers = buffers
        self._graph = graph
        self._shape = shape
        print(f"[cudagraphs] captured forward+backward for batch shape {shape}")