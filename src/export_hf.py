"""Export a training checkpoint to a Hugging Face-friendly folder:

    python -m src.export_hf --ckpt checkpoints/last.pt --which model --out hf_export

Produces:
    hf_export/model.safetensors   # weights (safetensors, not pickle)
    hf_export/config.json         # the Config used, to rebuild the U-Net

`--which model` exports the raw training weights; `--which ema` exports the EMA weights.
For short runs the raw weights are the better choice (the EMA hasn't caught up).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch
from safetensors.torch import save_file

from config import get_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/last.pt")
    p.add_argument("--which", choices=["model", "ema"], default="model")
    p.add_argument("--out", default="hf_export")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt[args.which]
    state = {k: v.contiguous() for k, v in state.items()}

    os.makedirs(args.out, exist_ok=True)
    save_file(state, os.path.join(args.out, "model.safetensors"))

    # Store the config so the architecture can be rebuilt on load.
    cfg = get_config()
    cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg.__dict__)
    cfg_dict["trained_epochs"] = ckpt.get("epoch", None)
    cfg_dict["exported_weights"] = args.which
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    n = sum(v.numel() for v in state.values()) / 1e6
    print(f"Exported {args.which} weights ({n:.1f}M params) -> {args.out}/model.safetensors")
    print(f"Config -> {args.out}/config.json  (trained_epochs={cfg_dict['trained_epochs']})")


if __name__ == "__main__":
    main()
