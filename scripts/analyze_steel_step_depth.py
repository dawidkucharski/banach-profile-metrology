"""Statistically evaluate steel step-depth contrast over recorded sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def robust_mad(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if array.size == 0:
        return float("nan")
    median = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - median)))


def summarise(group: pd.DataFrame) -> pd.Series:
    depth = group["step_depth_native"]
    signed = group["step_height"]
    return pd.Series(
        {
            "n": int(depth.count()),
            "mean_signed_step_height": float(signed.mean()),
            "sd_signed_step_height": float(signed.std(ddof=1)),
            "mean_step_depth": float(depth.mean()),
            "median_step_depth": float(depth.median()),
            "sd_step_depth": float(depth.std(ddof=1)),
            "mad_step_depth": robust_mad(depth),
            "sem_step_depth": float(depth.std(ddof=1) / np.sqrt(depth.count())),
            "repeatability_limit_2p8sr_depth": float(2.8 * depth.std(ddof=1)),
            "min_step_depth": float(depth.min()),
            "max_step_depth": float(depth.max()),
            "mean_Rq": float(group["Rq"].mean()),
            "mean_L2_to_mean": float(group["L2_to_mean"].mean()),
        }
    )


def compute_step_depth(input_csv: Path, frames_per_sweep: int, angular_step_deg: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(input_csv)
    data = data[np.isfinite(data["step_height"])].copy()
    data["sweep_index"] = (data["sequence_index"] // frames_per_sweep).astype(int) + 1
    data["angle_in_recorded_sweep_deg"] = (data["sequence_index"] % frames_per_sweep) * angular_step_deg
    data["step_depth_native"] = data["step_height"].abs()
    data["recorded_region"] = np.where(data["angle_in_recorded_sweep_deg"] < 360.0, "useful_360_deg", "overlap_40_deg")
    per_sweep_rows = []
    for sweep_index, group in data.groupby("sweep_index"):
        row = summarise(group)
        row["sweep_index"] = sweep_index
        per_sweep_rows.append(row)
    per_sweep = pd.DataFrame(per_sweep_rows)
    overall = summarise(data).to_frame().T
    overall.insert(0, "sweep_index", "all")
    summary = pd.concat([overall, per_sweep], ignore_index=True)
    return data, summary


def save_figure(figure: plt.Figure, path_stem: Path) -> None:
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


def write_plot(data: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    sweep_summary = summary[summary["sweep_index"] != "all"].copy()
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_sequence = figure.add_subplot(grid[0, :])
    axis_box = figure.add_subplot(grid[1, 0])
    axis_summary = figure.add_subplot(grid[1, 1])

    for sweep, group in data.groupby("sweep_index"):
        axis_sequence.plot(
            group["angle_in_recorded_sweep_deg"],
            group["step_depth_native"],
            linestyle="none",
            marker=".",
            markersize=2.0,
            alpha=0.55,
            label=f"sweep {sweep}",
        )
    axis_sequence.axvline(360.0, color="#D55E00", linestyle="--", linewidth=1.0, label="360 deg / overlap boundary")
    axis_sequence.set_title("Steel step-depth proxy over the recorded 400 deg sweeps")
    axis_sequence.set_xlabel("Angle in recorded sweep, deg")
    axis_sequence.set_ylabel("Step depth, native units")
    axis_sequence.grid(True, alpha=0.25)
    axis_sequence.legend(ncol=3, fontsize=7)

    grouped_depths = [group["step_depth_native"].to_numpy() for _, group in data.groupby("sweep_index")]
    axis_box.boxplot(grouped_depths, labels=[str(sweep) for sweep in sorted(data["sweep_index"].unique())], showfliers=False)
    axis_box.set_title("Depth distribution by recorded sweep")
    axis_box.set_xlabel("Sweep index")
    axis_box.set_ylabel("Step depth, native units")
    axis_box.grid(True, axis="y", alpha=0.25)

    axis_summary.errorbar(
        sweep_summary["sweep_index"].astype(int),
        sweep_summary["mean_step_depth"],
        yerr=sweep_summary["sd_step_depth"],
        fmt="o-",
        color="#0072B2",
        capsize=4,
        label="mean +/- SD",
    )
    axis_summary.set_title("Sweep-level repeatability")
    axis_summary.set_xlabel("Sweep index")
    axis_summary.set_ylabel("Step depth, native units")
    axis_summary.grid(True, alpha=0.25)
    axis_summary.legend(fontsize=7)

    figure.suptitle("Statistical evaluation of steel step depth from repeated recorded sweeps")
    save_figure(figure, output_dir / "steel_step_depth_statistics")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("results/r_btri_stage5_15000/step/step_frame_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/steel_step_depth"))
    parser.add_argument("--frames-per-sweep", type=int, default=20000)
    parser.add_argument("--angular-step-deg", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, summary = compute_step_depth(args.input_csv, args.frames_per_sweep, args.angular_step_deg)
    data.to_csv(args.output_dir / "steel_step_depth_frame_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "steel_step_depth_summary.csv", index=False)
    write_plot(data, summary, args.output_dir)


if __name__ == "__main__":
    main()