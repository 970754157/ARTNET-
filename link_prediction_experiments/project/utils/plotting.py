"""Plotting helpers for training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _rolling_stats(values: Sequence[float], window: int):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr, arr, arr
    w = max(1, window)
    means = np.zeros_like(arr)
    stds = np.zeros_like(arr)
    vars_ = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        sub = arr[start : i + 1]
        means[i] = float(np.mean(sub))
        stds[i] = float(np.std(sub))
        vars_[i] = float(np.var(sub))
    return means, stds, vars_


def plot_loss_mean_std_var(
    losses: Sequence[float],
    output_path: Path,
    window: int,
    title: str = "Train Loss with Rolling Mean/Std/Var",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(losses))
    means, stds, vars_ = _rolling_stats(losses, window)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, losses, label="loss", color="tab:blue", alpha=0.6)
    axes[0].plot(x, means, label=f"mean(w={window})", color="tab:orange")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, stds, label=f"std(w={window})", color="tab:green")
    axes[1].set_ylabel("std")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(x, vars_, label=f"var(w={window})", color="tab:red")
    axes[2].set_ylabel("var")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_val_curve(
    steps: Iterable[int],
    values: Iterable[float],
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = list(steps)
    y = list(values)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

