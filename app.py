"""Gradio demo for text-diffusion-fashion-mnist.

Runs locally (`python app.py`) and doubles as the entry point for the Hugging Face Space.
It pulls the trained weights from the model repo and generates garments on demand. On a free
CPU Space each generation takes a few seconds (the U-Net runs a 2x-batched forward per DDIM step).
"""
from __future__ import annotations

import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config import get_config, FASHION_CLASSES
from src.diffusion import GaussianDiffusion
from src.text_encoder import TextConditioner
from src.unet import UNet

MODEL_REPO = "mbsdeepak/text-diffusion-fashion-mnist"

cfg = get_config()
device = cfg.device

weights_path = hf_hub_download(MODEL_REPO, "model.safetensors")
model = UNet(cfg).to(device)
model.load_state_dict(load_file(weights_path))
model.eval()

conditioner = TextConditioner(cfg).to(device)
diffusion = GaussianDiffusion(cfg).to(device)


def _to_pil(x: torch.Tensor) -> Image.Image:
    """[1,H,W] in [-1,1] -> upscaled grayscale PIL image."""
    arr = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu().numpy()[0]
    return Image.fromarray(arr, mode="L").resize((160, 160), Image.NEAREST)


@torch.no_grad()
def generate(category: str, n_images: int, guidance: float, steps: int, seed: float):
    if seed is not None and seed >= 0:
        torch.manual_seed(int(seed))
    cfg.guidance_scale = float(guidance)
    cfg.ddim_steps = int(steps)

    idx = FASHION_CLASSES.index(category)
    labels = torch.full((int(n_images),), idx, dtype=torch.long, device=device)
    imgs = diffusion.ddim_sample(model, conditioner, labels)
    return [_to_pil(img) for img in imgs]


with gr.Blocks(title="text-diffusion-fashion-mnist") as demo:
    gr.Markdown(
        "# 🧵 text-diffusion-fashion-mnist\n"
        "Generate novel clothing images with a **from-scratch, CLIP-conditioned diffusion model** "
        "(DDPM + DDIM + classifier-free guidance), trained on Fashion-MNIST. Pick a category and "
        "the model paints new garments from pure noise. Runs on CPU, so give it a few seconds.\n\n"
        "Weights: [mbsdeepak/text-diffusion-fashion-mnist](https://huggingface.co/mbsdeepak/text-diffusion-fashion-mnist) · "
        "Code: [GitHub](https://github.com/mbsdeepak/text-diffusion-fashion-mnist)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            category = gr.Dropdown(FASHION_CLASSES, value="sneaker", label="Category")
            n_images = gr.Slider(1, 8, value=4, step=1, label="Number of images")
            guidance = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Guidance scale (CFG)")
            steps = gr.Slider(10, 50, value=40, step=5, label="DDIM steps")
            seed = gr.Number(value=42, label="Seed (-1 = random)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Samples", columns=4, height=360)

    btn.click(generate, [category, n_images, guidance, steps, seed], gallery)
    gr.Examples(
        examples=[["sneaker", 4, 3.0, 40, 42], ["dress", 4, 4.0, 40, 7], ["bag", 4, 3.0, 40, 1]],
        inputs=[category, n_images, guidance, steps, seed],
    )


if __name__ == "__main__":
    demo.queue().launch()
