"""Fréchet Inception Distance (FID) between generated samples and the real Fashion-MNIST test set.

    python -m scripts.fid --n-samples 2048            # writes/prints an FID number

Lower is better. NOTE: standard FID uses ImageNet-InceptionV3 on 299x299 RGB, so the *absolute*
value here (32x32 grayscale clothing, upscaled) is not comparable to natural-image FID in the
literature — treat it as a relative quality signal (e.g. across guidance scales / checkpoints).

Weights are pulled from the Hugging Face Hub, so no local training is needed.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets

from config import get_config, FASHION_CLASSES
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet

MODEL_REPO = "mbsdeepak/text-diffusion-fashion-mnist"


def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
    """[N,1,H,W] in [-1,1] -> [N,3,H,W] uint8 in [0,255] (FID/Inception wants 3 channels)."""
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return x.repeat(1, 3, 1, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=MODEL_REPO)
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (default: config)")
    args = p.parse_args()

    cfg = get_config()
    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
    dev = cfg.device
    n = args.n_samples
    print(f"Device: {dev} | samples: {n} | guidance: {cfg.guidance_scale}")

    # FID runs InceptionV3 on CPU here (MPS lacks some ops it uses); features are cheap enough.
    fid = FrechetInceptionDistance(feature=2048, normalize=False)

    # --- real: first n Fashion-MNIST test images, resized to the model's 32x32 ---
    ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True)
    real = ds.data[:n].float() / 255.0
    real = F.interpolate(real[:, None], size=cfg.image_size, mode="bilinear", align_corners=False)
    real = real * 2 - 1  # -> [-1,1]
    # Feed real images through Inception in batches too (a single 2048-image pass spikes RAM).
    for i in range(0, n, args.batch):
        fid.update(to_uint8_rgb(real[i:i + args.batch]), real=True)
    print(f"registered {n} real images")

    # --- generated: n samples spread evenly across the 10 classes ---
    model = UNet(cfg).to(dev)
    model.load_state_dict(load_file(hf_hub_download(args.repo, "model.safetensors")))
    model.eval()
    cond = TextConditioner(cfg).to(dev)
    diff = GaussianDiffusion(cfg).to(dev)

    done = 0
    with torch.no_grad():
        while done < n:
            b = min(args.batch, n - done)
            labels = torch.arange(done, done + b, device=dev) % len(FASHION_CLASSES)
            imgs = diff.ddim_sample(model, cond, labels)
            fid.update(to_uint8_rgb(imgs.cpu()), real=False)
            done += b
            print(f"  generated {done}/{n}", flush=True)

    score = fid.compute().item()
    print(f"\nFID ({n} samples, guidance {cfg.guidance_scale}) = {score:.2f}   (lower is better)")


if __name__ == "__main__":
    main()
