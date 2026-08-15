"""Render the training-loss curve from the committed loss history:

    python -m scripts.plot_curve        # reads assets/loss_history.csv -> assets/training_curve.png

The CSV (epoch, avg_loss) is produced during training; keeping it in the repo makes the plot
reproducible without re-running training.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="assets/loss_history.csv")
    p.add_argument("--out", default="assets/training_curve.png")
    args = p.parse_args()

    epochs, losses = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.plot(epochs, losses, marker="o", color="#4f46e5", linewidth=2, markersize=5)
    ax.set_title("Training loss — DDPM ε-prediction (Fashion-MNIST)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean-squared error")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    for x, y in zip(epochs, losses):
        if x in (1, len(epochs)):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"saved -> {args.out}  ({len(epochs)} epochs, final loss {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
