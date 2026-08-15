"""Generate images from a trained checkpoint.  Run from the project root, e.g.:

    python -m src.sample --classes all --n 6 --guidance 3.0 --out assets/samples.png
    python -m src.sample --classes sneaker,bag,dress --n 8 --out assets/shoes.png

Uses the EMA weights and DDIM sampling with classifier-free guidance.
"""
from __future__ import annotations

import argparse
import os

import torch
from torchvision.utils import save_image

from config import get_config, FASHION_CLASSES
from src.data import denormalize
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet


def parse_classes(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(len(FASHION_CLASSES)))
    out = []
    for token in spec.split(","):
        token = token.strip().lower()
        if token.isdigit():
            out.append(int(token))
        elif token in FASHION_CLASSES:
            out.append(FASHION_CLASSES.index(token))
        else:
            raise ValueError(f"Unknown class '{token}'. Choose from: {FASHION_CLASSES}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/last.pt")
    p.add_argument("--classes", default="all", help="'all', names, or indices (comma-separated)")
    p.add_argument("--n", type=int, default=6, help="images per class")
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (default: config)")
    p.add_argument("--steps", type=int, default=None, help="DDIM steps (default: config)")
    p.add_argument("--out", default="assets/samples.png")
    p.add_argument("--weights", choices=["ema", "model"], default="ema",
                   help="which weights to sample from; use 'model' (raw) for short runs where EMA lags")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = get_config()
    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
    if args.steps is not None:
        cfg.ddim_steps = args.steps
    if args.seed is not None:
        torch.manual_seed(args.seed)

    conditioner = TextConditioner(cfg).to(cfg.device)
    model = UNet(cfg).to(cfg.device)

    ckpt = torch.load(args.ckpt, map_location=cfg.device)
    key = args.weights if args.weights in ckpt else "model"
    model.load_state_dict(ckpt[key])
    print(f"Loaded '{key}' weights from {args.ckpt} (epoch {ckpt.get('epoch', '?')})")
    model.eval()
    diffusion = GaussianDiffusion(cfg).to(cfg.device)

    class_ids = parse_classes(args.classes)
    labels = torch.tensor([c for c in class_ids for _ in range(args.n)], device=cfg.device)

    print(f"Sampling {len(labels)} images (guidance={cfg.guidance_scale}, steps={cfg.ddim_steps})...")
    imgs = diffusion.ddim_sample(model, conditioner, labels)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(denormalize(imgs), args.out, nrow=args.n)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
