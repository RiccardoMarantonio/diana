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
        the random numbers stay fresh per step (a captured ``randn`` would
        freeze training). ``loss()`` below is just this with the draws.
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

    @torch.no_grad()
    def _reverse(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        clamp_x: bool = False,
    ) -> torch.Tensor:
        """Run the ancestral sampler along `times` (descending buffer indices).
        Update rule (Ho et al., 2020, Alg. 2):
          x_{t-1} = 1/sqrt(a_t) * (x_t - beta_t/sqrt(1-abar_t) * eps) + sqrt(beta_t)*z
        Striding `times` (e.g. 250 of 1000 steps) treats each stride as one
        reverse step: linear speedup, mild quality loss."""
        n = x.shape[0]
        for i, t_idx in enumerate(times.tolist()):
            t_batch = torch.full((n,), t_idx, dtype=torch.long, device=x.device)
            eps_pred = self.unet(x, t_batch)

            abar_t = self.schedule.alphas_cumprod[t_idx]
            beta_t = self.schedule.betas[t_idx]
            alpha_t = self.schedule.alphas[t_idx]

            mean = (x - beta_t / torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(alpha_t)
            if clamp_x:
                # Keeps intermediate samples in valid pixel range during
                # visualization runs; harmless statistically, helps stability
                # when starting from heavily noised real images.
                mean = mean.clamp(-1.0, 1.0)
            if i < len(times) - 1:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean  # no injected noise at the final step
        return x

    def _stride_times(self, num_steps: int | None) -> torch.Tensor:
        steps = num_steps or self.schedule.num_timesteps
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
        times_all = self._stride_times(num_steps)
        times = times_all[times_all <= t_start]
        if len(times) == 0:
            raise ValueError(f"no sampling steps at or below t_start={t_start}")

        noise = torch.randn_like(x0)
        x = self.q_sample(x0, torch.full((x0.shape[0],), t_start, dtype=torch.long, device=x0.device), noise)
        return self._reverse(x, times)
