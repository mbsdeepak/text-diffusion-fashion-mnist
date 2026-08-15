"""Fast end-to-end wiring check — NOT training. Runs a couple of forward/backward steps and
a tiny 4-step sampling pass on a tiny model so you can confirm everything is connected before
committing to a multi-hour run:

    python -m scripts.smoke_test

Exits non-zero if anything is miswired. Uses a shrunken config so it finishes in seconds.
"""
from __future__ import annotations

import torch

from config import get_config, FASHION_CLASSES
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet


def main() -> None:
    cfg = get_config()
    # Shrink everything so this runs in seconds even on CPU.
    cfg.base_channels = 16
    cfg.channel_mults = (1, 2)
    cfg.num_res_blocks = 1
    cfg.attn_resolutions = (16,)
    cfg.ddim_steps = 4
    cfg.timesteps = 100
    device = cfg.device
    print(f"Device: {device}")

    conditioner = TextConditioner(cfg).to(device)
    model = UNet(cfg).to(device)
    diffusion = GaussianDiffusion(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net parameters: {n_params:.2f}M")

    # --- one training step ---
    b = 4
    images = torch.randn(b, cfg.channels, cfg.image_size, cfg.image_size, device=device)
    labels = torch.randint(0, len(FASHION_CLASSES), (b,), device=device)
    labels = conditioner.apply_cfg_dropout(labels, cfg.cfg_dropout)
    pooled, tokens, mask = conditioner.encode(labels)

    t = torch.randint(0, cfg.timesteps, (b,), device=device)
    noise = torch.randn_like(images)
    x_t = diffusion.q_sample(images, t, noise)
    pred = model(x_t, t, pooled, tokens, mask)
    assert pred.shape == images.shape, f"pred shape {pred.shape} != {images.shape}"

    loss = torch.nn.functional.mse_loss(pred, noise)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), \
        "no gradients flowed — conditioning path is disconnected"
    print(f"train step OK — loss={loss.item():.4f}, grads flow")

    # --- one sampling pass ---
    with torch.no_grad():
        out = diffusion.ddim_sample(model, conditioner, labels[:2])
    assert out.shape == (2, cfg.channels, cfg.image_size, cfg.image_size), out.shape
    assert torch.isfinite(out).all(), "sampling produced NaNs/Infs"
    print(f"sampling OK — output {tuple(out.shape)}, range [{out.min():.2f}, {out.max():.2f}]")

    print("\nSMOKE TEST PASSED ✅  — the full pipeline is wired correctly.")


if __name__ == "__main__":
    main()
