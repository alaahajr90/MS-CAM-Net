"""Minimal plotting utilities for Stage 1 training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_path: Path,
    title: str,
) -> None:
    """Save train/validation total-loss curves without interpreting embeddings as rPPG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Training loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Stage 1 objective")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
