import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Maps integer timesteps t in [0, T) to dense vectors using sin/cos at
    geometrically spaced frequencies (Vaswani et al., 2017). Parameter-free:
    nothing here ever receives a gradient."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"embedding dim must be a positive even int, got {dim}")
        self.dim = dim
        # Geometric frequency ladder spanning [1, 1/10000]: high-frequency
        # channels discriminate adjacent timesteps, low-frequency channels
        # encode coarse position along the diffusion chain.
        half = dim // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        # persistent=False: recomputable from `dim`, so keep it out of state_dict
        # and out of every checkpoint we ship to/from Colab.
        self.register_buffer("freqs", torch.exp(exponent), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int64. Outer product -> args: (B, dim//2).
        # Kept in fp32 deliberately: sin/cos arguments lose too much precision
        # in fp16, and AMP autocast would otherwise downcast them silently.
        args = t.float()[:, None] * self.freqs[None, :]
        # (B, dim): first half sin, second half cos -- interleaving variant
        # also exists; concatenation is what DDPM uses and keeps indexing sane.
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepMLP(nn.Module):
    """Learnable projection of the fixed sinusoidal embedding into the width
    consumed by every ResBlock. This is where 'which timestep am I denoising?'
    becomes a vector the network can actually condition on."""

    def __init__(self, embedding_dim: int, time_emb_dim: int):
        super().__init__()
        if embedding_dim <= 0 or time_emb_dim <= 0:
            raise ValueError("dims must be positive")
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # (B,) -> (B, time_emb_dim). One embedding per sample, broadcast later
        # across all spatial positions inside each ResBlock.
        return self.mlp(t)


class TimestepEmbedding(nn.Module):
    """Composed entry point for the whole timestep pathway: raw integer
    timesteps (B,) in, conditioned vectors (B, time_emb_dim) out.
    Wrapping both stages removes any chance of calling the MLP on raw ints."""

    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(time_emb_dim)
        self.mlp = TimestepMLP(time_emb_dim, time_emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 32 groups; group count must divide channel count,
    so fall back via gcd for odd widths (keeps the block width-agnostic)."""
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ResBlock(nn.Module):
    """The DDPM residual atom. Two conv branches with GroupNorm/SiLU, timestep
    injected as an additive per-channel bias between them, and a residual
    shortcut (1x1 conv only when the channel count changes)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            _group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        # One Linear total: projects (B, time_emb_dim) -> (B, out_channels),
        # then broadcast over H,W -- O(1) in spatial size, unlike a 3x3 conv.
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.block2 = nn.Sequential(
            _group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        # Identity shortcut when shapes already match: zero extra params/memory.
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # temb: (B, time_emb_dim) -> (B, C, 1, 1); broadcasting adds one bias
        # vector to every spatial location without materializing copies.
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)
