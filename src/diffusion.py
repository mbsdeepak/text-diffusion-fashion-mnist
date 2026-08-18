"""The diffusion process itself — separate from the network so the math is easy to read.

Training side:  q_sample adds `t` steps of noise to a clean image (the "forward process").
Sampling side:  DDIM iteratively denoises pure noise back to an image (the "reverse process"),
                steered by classifier-free guidance.
"""
from __future__ import annotations

import torch

from config import Config


def make_beta_schedule(cfg: Config) -> torch.Tensor:
    T = cfg.timesteps
    if cfg.schedule == "linear":
        return torch.linspace(1e-4, 0.02, T)
    # Cosine schedule (Nichol & Dhariwal 2021) — noise is added more gently, which trains better.
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999).float()


def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather a[t] for a batch of timesteps and reshape to broadcast over an image tensor."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        betas = make_beta_schedule(cfg)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)  # signal-to-noise ratio per t

    def to(self, device: str) -> "GaussianDiffusion":
        for name in ("betas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "snr"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def min_snr_weight(self, t: torch.Tensor, gamma: float) -> torch.Tensor:
        """Per-sample Min-SNR-γ loss weight for ε-prediction: min(SNR_t, γ) / SNR_t = min(1, γ/SNR_t).
        Down-weights high-SNR (low-noise) timesteps so training isn't dominated by easy steps."""
        if gamma <= 0:
            return torch.ones(t.shape[0], device=t.device)
        return (gamma / self.snr.gather(0, t)).clamp(max=1.0)

    # ---- forward process (training) ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add `t` steps of noise: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) noise."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    # ---- reverse process (sampling) ----
    @torch.no_grad()
    def ddim_sample(self, model, conditioner, labels: torch.Tensor) -> torch.Tensor:
        """Generate images for the given class labels using DDIM + classifier-free guidance."""
        cfg = self.cfg
        device = labels.device
        b = labels.shape[0]

        # Precompute conditional and unconditional (null-caption) text embeddings.
        pooled_c, tokens_c, mask_c = conditioner.encode(labels)
        null = conditioner.null_labels(b, device)
        pooled_u, tokens_u, mask_u = conditioner.encode(null)

        # Sub-sequence of timesteps to actually visit, high -> low.
        step_indices = torch.linspace(0, cfg.timesteps - 1, cfg.ddim_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=[0])

        x = torch.randn(b, cfg.channels, cfg.image_size, cfg.image_size, device=device)

        for i, step in enumerate(step_indices):
            t = torch.full((b,), step.item(), device=device, dtype=torch.long)

            eps_c = model(x, t, pooled_c, tokens_c, mask_c)
            eps_u = model(x, t, pooled_u, tokens_u, mask_u)
            eps = eps_u + cfg.guidance_scale * (eps_c - eps_u)  # classifier-free guidance

            abar_t = _extract(self.alphas_cumprod, t, x.shape)
            if i < len(step_indices) - 1:
                t_prev = torch.full((b,), step_indices[i + 1].item(), device=device, dtype=torch.long)
                abar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                abar_prev = torch.ones_like(abar_t)  # final step denoises all the way to x0

            x0_pred = ((x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)).clamp(-1, 1)
            sigma = cfg.ddim_eta * torch.sqrt(
                (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev).clamp(min=0)
            )
            dir_xt = torch.sqrt((1 - abar_prev - sigma ** 2).clamp(min=0)) * eps
            noise = torch.randn_like(x) if (cfg.ddim_eta > 0 and i < len(step_indices) - 1) else 0.0
            x = torch.sqrt(abar_prev) * x0_pred + dir_xt + sigma * noise

        return x
