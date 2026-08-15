"""Frozen CLIP text encoder — the multimodal bridge.

This is the same idea Stable Diffusion uses: freeze a pretrained text encoder and condition
the image model on its embeddings. Because Fashion-MNIST has a *fixed* set of 10 captions
(plus one empty "null" caption for classifier-free guidance), we encode all 11 prompts **once**
at startup and cache them. Training then just indexes into the cache by label — no CLIP forward
pass per step, which keeps things fast on MPS/CPU.

For each prompt we keep two views:
  - `pooled`  [N, 512]      : one vector per caption, injected via FiLM into every ResBlock.
  - `tokens`  [N, L, 512]   : the full token sequence, attended to by cross-attention layers.
"""
from __future__ import annotations

import torch
from transformers import CLIPTokenizer, CLIPTextModel, logging as hf_logging

from config import Config, FASHION_CLASSES, PROMPT_TEMPLATE

# Loading CLIPTextModel from the full CLIP checkpoint warns about unused vision weights; harmless.
hf_logging.set_verbosity_error()


class TextConditioner:
    """Holds cached CLIP embeddings for the fixed prompt bank. Not an nn.Module: the encoder
    is frozen and thrown away after caching, so there are no trainable parameters here."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.null_index = len(FASHION_CLASSES)  # index of the empty "" caption

        prompts = [PROMPT_TEMPLATE.format(label=c) for c in FASHION_CLASSES]
        prompts.append("")  # null caption for classifier-free guidance

        tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model)
        encoder = CLIPTextModel.from_pretrained(cfg.clip_model).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        enc = tokenizer(
            prompts,
            padding="max_length",
            max_length=cfg.max_text_len,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = encoder(**enc)

        # Cache on CPU; .to(device) moves them onto the training device.
        self.tokens = out.last_hidden_state.detach()   # [N, L, 512]
        self.pooled = out.pooler_output.detach()        # [N, 512]
        self.mask = enc.attention_mask.detach()         # [N, L] (1 = real token, 0 = pad)

        del encoder  # free ~60M params; we only needed the cache

    def to(self, device: str) -> "TextConditioner":
        self.tokens = self.tokens.to(device)
        self.pooled = self.pooled.to(device)
        self.mask = self.mask.to(device)
        return self

    def encode(self, labels: torch.Tensor):
        """labels [B] long -> (pooled [B,512], tokens [B,L,512], mask [B,L])."""
        return self.pooled[labels], self.tokens[labels], self.mask[labels]

    def apply_cfg_dropout(self, labels: torch.Tensor, p: float) -> torch.Tensor:
        """Randomly replace a fraction `p` of captions with the null caption. Training the
        model to denoise both conditioned and unconditioned is what enables CFG at sample time."""
        if p <= 0:
            return labels
        drop = torch.rand(labels.shape[0], device=labels.device) < p
        labels = labels.clone()
        labels[drop] = self.null_index
        return labels

    def null_labels(self, n: int, device: str) -> torch.Tensor:
        return torch.full((n,), self.null_index, dtype=torch.long, device=device)
