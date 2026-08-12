import torch
import torch.nn as nn
import math


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        schedule_type: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        schedule_param: float = 0.008,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float64
            )

        elif schedule_type == "cosine":
            t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
            f_t = (
                torch.cos(
                    ((t / num_timesteps) + schedule_param)
                    / (1.0 + schedule_param)
                    * math.pi
                    / 2.0
                )
                ** 2
            )
            alphas_bar = f_t / f_t[0]
            betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
            betas.clamp(0, 0.999)

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, alphas.dim())
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer("betas", betas.to(torch.float16))
        self.register_buffer("alphas", alphas.to(torch.float16))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float16))
        self.register_buffer(
            "sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(torch.float16)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod.to(torch.float16),
        )
