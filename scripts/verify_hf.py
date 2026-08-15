"""Verify the *published* weights: download them from the Hugging Face Hub, load them into the
U-Net, generate a sample grid, and assert the output is valid. Reproduces the "the weights on HF
actually work" check that anyone can run:

    python -m scripts.verify_hf                       # writes assets/hf_verify.png
    python -m scripts.verify_hf --n 3 --guidance 1.5  # 3 images per class

Exits non-zero if the download, the state-dict load, or generation fails.
"""
from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.utils import save_image

from config import get_config, FASHION_CLASSES
from src.data import denormalize
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet

MODEL_REPO = "mbsdeepak/text-diffusion-fashion-mnist"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=MODEL_REPO, help="HF model repo to pull weights from")
    p.add_argument("--n", type=int, default=3, help="images per class")
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (default: config)")
    p.add_argument("--out", default="assets/hf_verify.png")
    args = p.parse_args()

    cfg = get_config()
    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
    device = cfg.device
    print(f"Device: {device}  |  repo: {args.repo}")

    # 1. Download the published weights from the Hub.
    weights_path = hf_hub_download(args.repo, "model.safetensors")
    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Downloaded model.safetensors ({size_mb:.1f} MB)")

    # 2. Load them into the U-Net — assert an exact key match.
    model = UNet(cfg).to(device)
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys when loading weights: {missing}"
    assert not unexpected, f"unexpected keys when loading weights: {unexpected}"
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {len(state)} tensors into U-Net ({n_params:.1f}M params); all keys matched")

    # 3. Generate one grid, `--n` images per class.
    conditioner = TextConditioner(cfg).to(device)
    diffusion = GaussianDiffusion(cfg).to(device)
    labels = torch.tensor(
        [c for c in range(len(FASHION_CLASSES)) for _ in range(args.n)], device=device
    )
    with torch.no_grad():
        imgs = diffusion.ddim_sample(model, conditioner, labels)

    # 4. Assert the output is well-formed.
    expected = (len(FASHION_CLASSES) * args.n, cfg.channels, cfg.image_size, cfg.image_size)
    assert tuple(imgs.shape) == expected, f"bad output shape {tuple(imgs.shape)} != {expected}"
    assert torch.isfinite(imgs).all(), "output contains NaNs/Infs"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(denormalize(imgs), args.out, nrow=args.n)
    print(f"Generated {imgs.shape[0]} images, range [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"\nHF WEIGHTS VERIFIED ✅  grid -> {args.out}")


if __name__ == "__main__":
    main()
