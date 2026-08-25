import math

import torch
from torch import nn


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 32 groups; group count must divide channel count,
    so fall back via gcd for odd widths (keeps the block width-agnostic)."""
    return nn.GroupNorm(math.gcd(32, channels), channels)


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
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepMLP(nn.Module):
    """Learnable projection of the fixed sinusoidal embedding into the width
    consumed by every ResBlock."""

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
    timesteps (B,) in, conditioned vectors (B, time_emb_dim) out."""

    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(time_emb_dim)
        self.mlp = TimestepMLP(time_emb_dim, time_emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


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


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions, DDPM-style:
    applied only at low resolutions where the quadratic cost stays cheap."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _group_norm(channels)
        # 1x1 convs as linear projections: same math as nn.Linear on (B,C,H,W)
        # but without flatten/unflatten copies of the activation tensor.
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)  # each (B, C, H, W)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        q, k, v = split(q), split(k), split(v)
        # Fused kernel: softmax(QK^T/sqrt(d))V without materializing the full
        # (B*heads, N, N) matrix -- O(N^2) compute, ~O(N) activation memory.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Strided 3x3 conv (learnable anti-aliasing) rather than max-pool:
        # halving H,W halves activation memory going into the next level.
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Nearest-neighbor 2x then conv: avoids the checkerboard artifacts
        # that transposed convolutions introduce into reconstructions --
        # artifacts the anomaly map would happily mistake for defects.
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class UNet(nn.Module):
    """DDPM denoiser: eps(x_t, t) -> predicted noise, same shape as x_t.

    Encoder/decoder ladders built from channel_mults; skip connections carry
    encoder features to the decoder at matching resolutions; the timestep
    conditions every ResBlock via additive bias.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        attention_resolutions: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        num_levels = len(channel_mults)
        self.time_embedding = TimestepEmbedding(base_channels * 4)
        time_dim = base_channels * 4

        def has_attention(resolution: int) -> bool:
            return resolution in attention_resolutions

        # ---- stem ----
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- build the ladder, mirroring runtime order so channel counts
        # stay correct without any post-hoc shape algebra ----
        # Flat ledger: one int per runtime skip, in exact storage order. The
        # forward() stack is flat (append after every block AND every
        # downsample), so this ledger must be flat too.
        skips_channels: list[int] = [base_channels]  # stem contributes one
        down_levels: list[nn.ModuleList] = []
        downs: list[nn.Module] = []
        c = base_channels
        for level, mult in enumerate(channel_mults):
            out_c = base_channels * mult
            resolution = img_size // (2**level)
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(c, out_c, time_dim, dropout))
                c = out_c
                skips_channels.append(out_c)
            if has_attention(resolution):
                blocks.append(SelfAttention(out_c))
            down_levels.append(nn.ModuleList(blocks))
            if level < num_levels - 1:
                downs.append(Downsample(c))
                skips_channels.append(c)
        self.down_levels = nn.ModuleList(down_levels)
        self.down_samples = nn.ModuleList(downs)

        # ---- bottleneck: two ResBlocks sandwiching attention ----
        self.mid_block1 = ResBlock(c, c, time_dim, dropout)
        mid_resolution = img_size // (2 ** (num_levels - 1))
        self.mid_attn = (
            SelfAttention(c) if has_attention(mid_resolution) else None
        )
        self.mid_block2 = ResBlock(c, c, time_dim, dropout)

        # ---- decoder: num_res_blocks + 1 blocks per level; each fuses ONE
        # popped skip via concat before projecting to the level width. Pops
        # mirror the runtime stack exactly (LIFO over flat entries). ----
        up_levels: list[nn.ModuleList] = []
        ups: list[nn.Module] = []
        for level in reversed(range(num_levels)):
            out_c = base_channels * channel_mults[level]
            resolution = img_size // (2**level)
            blocks = []
            for _ in range(num_res_blocks + 1):
                fused_in = c + skips_channels.pop()
                blocks.append(ResBlock(fused_in, out_c, time_dim, dropout))
                c = out_c
            if has_attention(resolution):
                blocks.append(SelfAttention(out_c))
            up_levels.append(nn.ModuleList(blocks))
            ups.append(Upsample(c) if level > 0 else nn.Identity())
        assert not skips_channels, "skip bookkeeping desynced"
        self.up_levels = nn.ModuleList(up_levels)
        self.up_samples = nn.ModuleList(ups)

        # ---- head: predict noise eps ~ N(0,1), so NO final activation ----
        self.final = nn.Sequential(
            _group_norm(c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(t)

        h = self.init_conv(x)
        skips: list[torch.Tensor] = [h]

        for level_idx, blocks in enumerate(self.down_levels):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, temb)
                    skips.append(h)  # ONLY ResBlocks emit skips; attention
                    continue         # transforms in-place and stores nothing.
                h = block(h)
            if level_idx < len(self.down_levels) - 1:
                h = self.down_samples[level_idx](h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for level_idx, blocks in enumerate(self.up_levels):
            for block in blocks:
                if isinstance(block, ResBlock):
                    # Only ResBlocks consume a skip; attention must NOT pop,
                    # or the stack drifts and decoder widths desync.
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, temb)
                    continue
                h = block(h)
            h = self.up_samples[level_idx](h)

        assert not skips, "runtime skip stack not fully consumed"
        return self.final(h)
