from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_loss(log_history: Iterable[dict], output_path: str | Path) -> Path:
    records = [
        {"step": item["step"], "loss": item["loss"]}
        for item in log_history
        if "loss" in item and "step" in item
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return output_path

    df = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["loss"], linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_accuracy_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_df.empty:
        return output_path

    plt.figure(figsize=(10, 5))
    plt.bar(metrics_df["label"], metrics_df["exact_match_accuracy"])
    plt.ylabel("Exact Match Accuracy")
    plt.title("Model / Setting Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
