"""Training entry point.  Run from the project root:

    python -m src.train

Trains eps_theta with the simple DDPM MSE objective, keeps an EMA copy of the weights
(EMA samples are noticeably cleaner), checkpoints each epoch, and dumps a preview grid
every few epochs so you can watch the model learn.
"""
from __future__ import annotations

import argparse
import copy
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm

from config import get_config, FASHION_CLASSES
from src.data import get_dataloader, denormalize
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


@torch.no_grad()
def save_preview(model, ema, diffusion, conditioner, cfg, path: str) -> None:
    """Sample one image per class using the EMA weights, then restore live weights."""
    backup = copy.deepcopy(model.state_dict())
    ema.copy_to(model)
    model.eval()

    labels = torch.arange(len(FASHION_CLASSES), device=cfg.device)
    imgs = diffusion.ddim_sample(model, conditioner, labels)
    save_image(denormalize(imgs), path, nrow=len(FASHION_CLASSES))

    model.load_state_dict(backup)
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="continue training from checkpoints/last.pt")
    parser.add_argument("--epochs", type=int, default=None, help="override total epochs")
    args = parser.parse_args()

    cfg = get_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    print(f"Device: {cfg.device}")

    loader = get_dataloader(cfg, train=True)
    conditioner = TextConditioner(cfg).to(cfg.device)
    model = UNet(cfg).to(cfg.device)
    diffusion = GaussianDiffusion(cfg).to(cfg.device)
    ema = EMA(model, cfg.ema_decay)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net parameters: {n_params:.1f}M")

    # Resume from the last checkpoint if requested (restores weights, EMA, and optimizer state).
    start_epoch = 0
    ckpt_path = os.path.join(cfg.ckpt_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(cfg.device) for k, v in ck["ema"].items()}
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {ckpt_path}: continuing at epoch {start_epoch + 1}/{cfg.epochs}")

    # Loss history CSV (accumulates across resumes) — feeds scripts/plot_curve.py.
    csv_path = os.path.join("assets", "loss_history.csv")
    if not (args.resume and os.path.exists(csv_path)):
        with open(csv_path, "w") as f:
            f.write("epoch,avg_loss\n")

    step = start_epoch * len(loader)
    for epoch in range(cfg.epochs):
        model.train()
        running, count = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for images, labels in pbar:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            # Classifier-free guidance: drop the caption for a fraction of the batch.
            labels = conditioner.apply_cfg_dropout(labels, cfg.cfg_dropout)
            pooled, tokens, mask = conditioner.encode(labels)

            t = torch.randint(0, cfg.timesteps, (images.shape[0],), device=cfg.device)
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise)

            pred = model(x_t, t, pooled, tokens, mask)
            loss = torch.nn.functional.mse_loss(pred, noise)

            if not torch.isfinite(loss):
                print(f"DIVERGED: non-finite loss at step {step}, epoch {epoch + 1}", flush=True)
                return

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            ema.update(model)

            step += 1
            running += loss.item()
            count += 1
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / max(count, 1)
        print(f"[epoch {epoch + 1}/{cfg.epochs}] avg_loss={avg:.4f}", flush=True)
        with open(csv_path, "a") as f:
            f.write(f"{epoch + 1},{avg:.4f}\n")

        ckpt = {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "optim": optim.state_dict(),
            "epoch": epoch,
            "config": cfg.__dict__,
        }
        torch.save(ckpt, os.path.join(cfg.ckpt_dir, "last.pt"))

        if (epoch + 1) % cfg.sample_every_epochs == 0:
            path = os.path.join(cfg.sample_dir, f"epoch_{epoch + 1:03d}.png")
            save_preview(model, ema, diffusion, conditioner, cfg, path)
            print(f"  saved preview -> {path}")

    print("Training complete. Checkpoint at", os.path.join(cfg.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    main()
