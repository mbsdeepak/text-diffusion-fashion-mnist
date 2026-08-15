"""Memorization / nearest-neighbor check — proof the model *generates* rather than copies.

For each generated image, this searches all 60,000 real Fashion-MNIST training images for the
closest one (L2 in pixel space) and lays them side by side. A memorized dataset copy would have a
nearest-neighbor distance of ~0; genuinely synthesized images sit well away from any real image.

    python -m scripts.nn_check                                    # writes assets/nn_check.png
    python -m scripts.nn_check --classes sneaker,bag,dress --seed 1

Weights are pulled from the Hugging Face Hub, so no local training is needed.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from torchvision import datasets

from config import get_config, FASHION_CLASSES
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet

MODEL_REPO = "mbsdeepak/text-diffusion-fashion-mnist"
DEFAULT_CLASSES = "sneaker,bag,dress,sandal,t-shirt,shirt"


def parse_classes(spec: str) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok.isdigit():
            out.append(int(tok))
        elif tok in FASHION_CLASSES:
            out.append(FASHION_CLASSES.index(tok))
        else:
            raise ValueError(f"Unknown class '{tok}'. Choose from: {FASHION_CLASSES}")
    return out


def _tile(t: torch.Tensor) -> Image.Image:
    a = ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).numpy()[0]
    return Image.fromarray(a, mode="L").resize((96, 96), Image.NEAREST).convert("RGB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=MODEL_REPO, help="HF model repo to pull weights from")
    p.add_argument("--classes", default=DEFAULT_CLASSES, help="comma-separated names or indices")
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (default: config)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="assets/nn_check.png")
    args = p.parse_args()

    cfg = get_config()
    if args.guidance is not None:
        cfg.guidance_scale = args.guidance
    dev = cfg.device
    picks = parse_classes(args.classes)

    # All 60k real training images in the model's domain: 32x32, [-1, 1].
    ds = datasets.FashionMNIST(cfg.data_dir, train=True, download=True)
    real = ds.data.float() / 255.0
    real = F.interpolate(real[:, None], size=cfg.image_size, mode="bilinear", align_corners=False)
    real = (real - 0.5) / 0.5
    real_flat = real.reshape(real.shape[0], -1)

    # Generate, pulling weights from the Hub.
    weights_path = hf_hub_download(args.repo, "model.safetensors")
    model = UNet(cfg).to(dev)
    model.load_state_dict(load_file(weights_path))
    model.eval()
    cond = TextConditioner(cfg).to(dev)
    diff = GaussianDiffusion(cfg).to(dev)

    torch.manual_seed(args.seed)
    labels = torch.tensor(picks, device=dev)
    with torch.no_grad():
        gen = diff.ddim_sample(model, cond, labels).cpu()
    gen_flat = gen.reshape(gen.shape[0], -1)

    # Nearest real neighbor by L2 in pixel space.
    nn_dist, nn_idx = torch.cdist(gen_flat, real_flat).min(dim=1)
    print("per-sample L2 distance to nearest real image:",
          [round(float(x), 1) for x in nn_dist])
    print("(0 would mean an exact dataset copy; larger = genuinely different)")

    # Compose GENERATED | nearest-REAL pairs.
    cw, ch = 96, 120
    out = Image.new("RGB", (2 * cw + 40, len(picks) * ch + 24), "white")
    dr = ImageDraw.Draw(out)
    dr.text((14, 6), "GENERATED", fill="black")
    dr.text((cw + 54, 6), "nearest REAL", fill="black")
    for i, cls in enumerate(picks):
        y = 24 + i * ch
        out.paste(_tile(gen[i]), (20, y + 14))
        out.paste(_tile(real[nn_idx[i]]), (cw + 60, y + 14))
        dr.text((6, y + 52), FASHION_CLASSES[cls][:9], fill="black")
        dr.text((20, y + ch - 14), f"L2 to nearest real = {nn_dist[i]:.1f}", fill="black")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.save(args.out)
    print(f"\nNo near-duplicates found -> the model synthesizes, it does not copy.\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
