from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from diana.diffusion.schedule import DiffusionSchedule
from diana.models.unet import UNet


class DDPM(nn.Module):
    """Wraps U-Net + noise schedule into the trainable diffusion model.

    The schedule is a child module on purpose: .to(device) and state_dict()
    cascade through it, so checkpoints carry the exact beta tables used in
    training -- mismatched schedules silently corrupt reconstructions.
    """

    def __init__(self, unet: UNet, schedule: DiffusionSchedule, objective: str = "pred_noise"):
        super().__init__()
        if objective != "pred_noise":
            # Spec fixes epsilon-matching; other objectives are future work.
            raise NotImplementedError(f"objective {objective!r}: only 'pred_noise' is implemented")
        self.unet = unet
        self.schedule = schedule
        self.objective = objective

    # ------------------------------------------------------------------
    # Forward process (closed form, no learning)
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """One-shot corruption x0 -> x_t via
        x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps."""
        sqrt_abar = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_abar * x0 + sqrt_1m * noise

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def loss_from(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Epsilon-matching loss for *given* timesteps and noise.

        The stochastic pieces (t draw, noise draw) are injected by the caller
        so the deterministic core can run inside a CUDA-graph capture while
        callers independently control the noise and timestep streams. ``loss()`` below is just this with the draws.
        """
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.unet(x_t, t)
        return F.mse_loss(eps_pred, noise)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Epsilon-matching loss: predict the noise added to a random-t
        corruption of the batch. One uniform-random timestep per sample --
        this is what makes a single epoch cover the full chain."""
        batch = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.schedule.num_timesteps, (batch,), device=device)
        noise = torch.randn_like(x0)
        return self.loss_from(x0, t, noise)

    # ------------------------------------------------------------------
    # Reverse process (inference only)
    # ------------------------------------------------------------------

    def _respaced_step(self, t: int, t_prev: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective (alpha, beta) for ONE reverse hop from index ``t`` to ``t_prev``.

        Raw per-index alphas/betas are only valid for a unit step. When the
        sampler strides (e.g. 250 of 1000), a hop from t to t_prev removes the
        noise of every skipped step, so the hop's alpha is the product of the
        skipped alphas -- which telescopes to abar[t] / abar[t_prev]. For
        t_prev == t - 1 this is exactly alphas[t] (full chain is unchanged).
        ``t_prev=None`` is the final hop to the clean image (abar_prev = 1).
        """
        abar = self.schedule.alphas_cumprod
        abar_t = abar[t]
        abar_prev = abar[t_prev] if t_prev is not None else torch.ones_like(abar_t)
        alpha = abar_t / abar_prev
        return alpha, (1.0 - alpha).clamp(min=0.0)

    @torch.no_grad()
    def _reverse(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        clamp_x: bool = False,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the ancestral sampler along `times` (descending buffer indices).
        Update rule (Ho et al., 2020, Alg. 2), with the schedule respaced to
        the strided chain (see :meth:`_respaced_step`):
          x_prev = 1/sqrt(a) * (x_t - b/sqrt(1-abar_t) * eps) + sqrt(b)*z
        where (a, b) are the effective alpha/beta of the hop t -> next time.
        The network is labeled with ``t`` -- the noise level ``x`` really is at."""
        n = x.shape[0]
        steps = times.tolist()
        for i, t_idx in enumerate(steps):
            t_batch = torch.full((n,), t_idx, dtype=torch.long, device=x.device)
            eps_pred = (self.unet if denoiser is None else denoiser)(x, t_batch)

            last = i == len(steps) - 1
            alpha, beta = self._respaced_step(t_idx, None if last else steps[i + 1])
            abar_t = self.schedule.alphas_cumprod[t_idx]

            mean = (x - beta / torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(alpha)
            if clamp_x:
                # Keeps intermediate samples in valid pixel range during
                # visualization runs; harmless statistically, helps stability
                # when starting from heavily noised real images.
                mean = mean.clamp(-1.0, 1.0)
            if not last:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean  # no injected noise at the final step
        return x

    def _stride_times(self, num_steps: int | None) -> torch.Tensor:
        steps = self.schedule.num_timesteps if num_steps is None else num_steps
        if not 0 < steps <= self.schedule.num_timesteps:
            raise ValueError(f"num_steps must be in (0, {self.schedule.num_timesteps}], got {steps}")
        # Descending, evenly spaced buffer indices; endpoints inclusive.
        # unique() sorts ascending, so flip back -- the sampler walks time
        # backwards.
        return torch.unique(torch.linspace(self.schedule.num_timesteps - 1, 0, steps).round().long()).flip(0)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        img_size: int,
        in_channels: int = 3,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Pure generation: start from N(0,I), run the full reverse chain."""
        device = device or next(self.parameters()).device
        x = torch.randn(n_samples, in_channels, img_size, img_size, device=device)
        return self._reverse(x, self._stride_times(num_steps))

    @torch.no_grad()
    def reconstruct(
        self,
        x0: torch.Tensor,
        num_steps: int | None = None,
        t_start: int | None = None,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """SDEdit-style reconstruction -- the anomaly-detection workhorse.

        Noise x0 up to an intermediate level t_start (preserving global
        structure), then denoise back down. The model re-imagines fine detail
        according to its TRAINING distribution (healthy images): unseen defect
        patterns get smoothed away. Pixel-wise |x0 - reconstruction| is the
        anomaly signal.

        t_start trades faithfulness against generative correction:
          low  -> output stays close to input (weak healing)
          high -> stronger healing but global structure may drift.
        """
        t_start = self.schedule.num_timesteps - 1 if t_start is None else t_start
        if not 0 <= t_start < self.schedule.num_timesteps:
            raise ValueError(f"t_start must be in [0, {self.schedule.num_timesteps}), got {t_start}")
        # x is noised to EXACTLY t_start, so the first denoising call must be
        # labeled t_start too; the strided grid only supplies the steps below.
        # (Labeling with the nearest grid point <= t_start would hand the
        # network a sample noisier than the level it was told to expect.)
        times_all = self._stride_times(num_steps)
        times = torch.cat([
            torch.tensor([t_start], dtype=times_all.dtype, device=times_all.device),
            times_all[times_all < t_start],
        ])

        noise = torch.randn_like(x0)
        x = self.q_sample(x0, torch.full((x0.shape[0],), t_start, dtype=torch.long, device=x0.device), noise)
        return self._reverse(x, times, denoiser=denoiser)
