"""Conditional U-Net — the noise predictor eps_theta(x_t, t, text).

Architecture is a scaled-down version of the Stable Diffusion / OpenAI guided-diffusion U-Net:
  - Sinusoidal timestep embedding + a pooled-text embedding, summed and injected into every
    ResBlock (FiLM-style conditioning).
  - Self-attention at low resolutions to capture global structure.
  - Cross-attention to the CLIP token sequence at low resolutions — this is how the *caption*
    steers generation.
  - Symmetric down/up path with skip connections.

The network predicts the noise epsilon that was added to the image (the standard DDPM
parameterization).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def make_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count (<=32) that divides `channels`."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the diffusion timestep, like positional encodings in Transformers.
    t: [B] (long/float) -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # odd dim: pad one column
        emb = F.pad(emb, (0, 1))
    return emb


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[B, N, C] -> [B, heads, N, C//heads]."""
    b, n, c = x.shape
    return x.view(b, n, heads, c // heads).transpose(1, 2)


class ResBlock(nn.Module):
    """Two conv layers with a conditioning embedding added in between (FiLM via bias shift)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = make_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = make_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial positions."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = make_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).permute(1, 0, 3, 2)  # 3 x [B,N,C]
        q, k, v = (split_heads(t, self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v)              # [B,heads,N,hd]
        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class CrossAttention2d(nn.Module):
    """Multi-head cross-attention: image positions (queries) attend to text tokens (keys/values)."""

    def __init__(self, channels: int, text_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = make_norm(channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(self.norm(x)).reshape(b, c, h * w).transpose(1, 2)  # [B, N, C]
        q = split_heads(q, self.heads)                                    # [B,heads,N,hd]
        k = split_heads(self.to_k(context), self.heads)                   # [B,heads,L,hd]
        v = split_heads(self.to_v(context), self.heads)
        # mask [B, L] (1=real,0=pad) -> [B,1,1,L] boolean; True = attend.
        attn_mask = mask[:, None, None, :].bool() if mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,heads,N,hd]
        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CondSequential(nn.ModuleList):
    """A sequence of layers that routes the right extra inputs to each layer type."""

    def __init__(self, *layers):
        super().__init__(layers)

    def forward(self, x, emb, context, mask):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, CrossAttention2d):
                x = layer(x, context, mask)
            else:  # SelfAttention2d, Downsample, Upsample, Conv2d
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        emb_dim = ch * 4
        heads = cfg.num_heads
        attn_res = set(cfg.attn_resolutions)

        # timestep + pooled-text -> shared conditioning embedding
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_embed_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        # ---- down path ----
        self.in_conv = nn.Conv2d(cfg.channels, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        input_block_chans = [ch]
        now = ch
        res = cfg.image_size
        for level, mult in enumerate(cfg.channel_mults):
            out = ch * mult
            for _ in range(cfg.num_res_blocks):
                layers = [ResBlock(now, out, emb_dim, cfg.dropout)]
                now = out
                if res in attn_res:
                    layers += [SelfAttention2d(now, heads),
                               CrossAttention2d(now, cfg.text_embed_dim, heads)]
                self.down_blocks.append(CondSequential(*layers))
                input_block_chans.append(now)
            if level != len(cfg.channel_mults) - 1:
                self.down_blocks.append(CondSequential(Downsample(now)))
                input_block_chans.append(now)
                res //= 2

        # ---- middle ----
        self.middle = CondSequential(
            ResBlock(now, now, emb_dim, cfg.dropout),
            SelfAttention2d(now, heads),
            CrossAttention2d(now, cfg.text_embed_dim, heads),
            ResBlock(now, now, emb_dim, cfg.dropout),
        )

        # ---- up path ----
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channel_mults))):
            out = ch * mult
            for i in range(cfg.num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                layers = [ResBlock(now + skip_ch, out, emb_dim, cfg.dropout)]
                now = out
                if res in attn_res:
                    layers += [SelfAttention2d(now, heads),
                               CrossAttention2d(now, cfg.text_embed_dim, heads)]
                if level != 0 and i == cfg.num_res_blocks:
                    layers.append(Upsample(now))
                    res *= 2
                self.up_blocks.append(CondSequential(*layers))

        self.out_norm = make_norm(now)
        self.out_conv = nn.Conv2d(now, cfg.channels, 3, padding=1)

    def forward(self, x, t, pooled, tokens, mask):
        emb = self.time_mlp(timestep_embedding(t, self.cfg.base_channels)) + self.text_proj(pooled)

        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, emb, tokens, mask)
            skips.append(h)

        h = self.middle(h, emb, tokens, mask)

        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, emb, tokens, mask)

        return self.out_conv(F.silu(self.out_norm(h)))
