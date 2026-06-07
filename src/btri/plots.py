"""Plotting helpers for BTRI output tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOT_METRICS = (
    "banach_l1_mean",
    "banach_l2_rms",
    "banach_linf",
    "banach_bv_total_variation_density",
    "banach_w12_gradient_seminorm",
    "step_height",
)


def write_metric_plots(frame_table: pd.DataFrame, output_dir: Path, metrics: Iterable[str] = DEFAULT_PLOT_METRICS) -> None:
    """Write compact trend and histogram figures for available metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in metrics:
        if metric_name not in frame_table.columns:
            continue
        metric_series = frame_table[metric_name].dropna()
        if metric_series.empty:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axes[0].plot(frame_table["sequence_index"], frame_table[metric_name], linewidth=0.8)
        axes[0].set_xlabel("Frame index")
        axes[0].set_ylabel(metric_name)
        axes[0].set_title("Sequence trend")
        axes[1].hist(metric_series, bins=40, color="#4C78A8", edgecolor="white")
        axes[1].set_xlabel(metric_name)
        axes[1].set_ylabel("Count")
        axes[1].set_title("Empirical distribution")
        figure.savefig(output_dir / f"{metric_name}.png", dpi=200)
        plt.close(figure)
