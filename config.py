"""Central configuration. Every hyperparameter lives here so experiments are reproducible
and the training/sampling scripts stay free of magic numbers."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch


def pick_device() -> str:
    """Prefer Apple MPS, then CUDA, then CPU. This project is tuned for MPS."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# The 10 Fashion-MNIST classes, in label order (0..9). We turn each into a text prompt
# so the model is conditioned on CLIP *text* embeddings — this is what makes it multimodal.
FASHION_CLASSES = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

# Turn a class name into a natural-language caption for the CLIP text encoder.
PROMPT_TEMPLATE = "a photo of a {label}, product image on white background"


@dataclass
class Config:
    # --- data ---
    image_size: int = 32          # Fashion-MNIST is 28x28; we pad/resize to 32 for clean downsampling
    channels: int = 1             # grayscale
    data_dir: str = "data"

    # --- diffusion ---
    timesteps: int = 1000         # T for training (cosine schedule)
    schedule: str = "cosine"      # "cosine" (Nichol & Dhariwal) or "linear"
    min_snr_gamma: float = 5.0    # Min-SNR-γ loss weighting (Hang et al. 2023); 0 disables

    # --- model (U-Net) ---
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)   # resolutions 32 -> 16 -> 8
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16, 8)  # where cross-/self-attention is applied
    num_heads: int = 4
    dropout: float = 0.1
    text_embed_dim: int = 512     # CLIP ViT-B/32 text hidden size

    # --- text conditioning ---
    clip_model: str = "openai/clip-vit-base-patch32"
    max_text_len: int = 20        # captions are short; keep the token sequence small
    cfg_dropout: float = 0.15     # prob. of dropping the caption during training (enables CFG)

    # --- training ---
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 40
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_every: int = 50
    sample_every_epochs: int = 2  # dump a preview grid every N epochs
    num_workers: int = 0          # 0 is safest on macOS/MPS

    # --- sampling ---
    ddim_steps: int = 50          # DDIM lets us sample in 50 steps instead of 1000
    ddim_eta: float = 0.0         # 0 = deterministic DDIM
    guidance_scale: float = 1.5   # classifier-free guidance; small/short-trained models artifact at high CFG

    # --- io / runtime ---
    ckpt_dir: str = "checkpoints"
    sample_dir: str = "samples"
    seed: int = 42
    device: str = field(default_factory=pick_device)


def get_config() -> Config:
    return Config()
