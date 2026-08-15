"""Fashion-MNIST data pipeline.

We normalize images to [-1, 1] (the range diffusion models expect, since the reverse
process predicts Gaussian noise centered at 0) and expose the integer class label. The
label is later mapped to a *text prompt* + CLIP embedding in text_encoder.py, which is
what makes this a text-to-image model rather than a plain class-conditional one.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),          # 28 -> 32
            transforms.ToTensor(),                  # [0,1], shape (1,H,W)
            transforms.Normalize((0.5,), (0.5,)),   # -> [-1, 1]
        ]
    )


def get_dataloader(cfg: Config, train: bool = True) -> DataLoader:
    dataset = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=train,
        download=True,
        transform=build_transforms(cfg.image_size),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        drop_last=train,
        pin_memory=False,  # MPS/CPU: pinning has no benefit
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] for saving/visualization."""
    return (x.clamp(-1, 1) + 1) / 2
