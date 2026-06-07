"""Compute acquisition-condition statistics for the interferometric datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = [
    {
        "object": "Ceramic",
        "omega_deg_s": 0.50,
        "fps": 6.0,
        "revolutions": 5,
        "total_frames": 24000,
        "rows": 488,
        "cols": 648,
        "diameter_mm": 29.9588,
    },
    {
        "object": "Steel",
        "omega_deg_s": 0.08,
        "fps": 4.0,
        "revolutions": 5,
        "total_frames": 100000,
        "rows": 480,
        "cols": 640,
        "diameter_mm": np.nan,
    },
]


def compute_rows() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        omega = float(dataset["omega_deg_s"])
        fps = float(dataset["fps"])
        revolutions = int(dataset["revolutions"])
        total_frames = int(dataset["total_frames"])
        angular_step = omega / fps
        frames_per_degree = fps / omega
        useful_frames_per_360 = 360.0 / angular_step
        acquired_frames_per_400 = 400.0 / angular_step
        nominal_total_360 = useful_frames_per_360 * revolutions
        nominal_total_400 = acquired_frames_per_400 * revolutions
        observed_frames_per_run = total_frames / revolutions
        observed_angle_per_run = observed_frames_per_run * angular_step
        overlap_degrees = observed_angle_per_run - 360.0
        overlap_frames = overlap_degrees / angular_step
        useful_duration_s = 360.0 / omega
        acquired_duration_s = observed_angle_per_run / omega
        total_duration_s = total_frames / fps
        pixels_per_frame = int(dataset["rows"]) * int(dataset["cols"])
        pixel_samples_total = pixels_per_frame * total_frames
        angular_pixel_density = pixels_per_frame * frames_per_degree
        nyquist_harmonic_360 = useful_frames_per_360 / 2.0
        min_angular_wavelength_deg = 2.0 * angular_step
        diameter = float(dataset["diameter_mm"])
        if np.isfinite(diameter):
            circumference_mm = np.pi * diameter
            arc_step_um = circumference_mm * angular_step / 360.0 * 1000.0
            nyquist_arc_um = 2.0 * arc_step_um
        else:
            arc_step_um = np.nan
            nyquist_arc_um = np.nan
        rows.append(
            {
                "object": dataset["object"],
                "omega_deg_s": omega,
                "fps": fps,
                "revolutions": revolutions,
                "total_frames": total_frames,
                "angular_step_deg_per_frame": angular_step,
                "frames_per_degree": frames_per_degree,
                "useful_frames_per_360_deg": useful_frames_per_360,
                "acquired_frames_per_400_deg": acquired_frames_per_400,
                "observed_frames_per_recorded_run": observed_frames_per_run,
                "observed_angle_per_recorded_run_deg": observed_angle_per_run,
                "overlap_degrees_per_run": overlap_degrees,
                "overlap_frames_per_run": overlap_frames,
                "nominal_total_frames_360_deg_only": nominal_total_360,
                "nominal_total_frames_400_deg_recorded": nominal_total_400,
                "useful_duration_per_360_s": useful_duration_s,
                "recorded_duration_per_run_s": acquired_duration_s,
                "total_acquisition_duration_s": total_duration_s,
                "pixels_per_frame": pixels_per_frame,
                "total_pixel_samples": pixel_samples_total,
                "angular_pixel_samples_per_degree": angular_pixel_density,
                "nyquist_harmonic_per_360_deg": nyquist_harmonic_360,
                "minimum_angular_wavelength_deg": min_angular_wavelength_deg,
                "arc_step_um_ceramic_equator": arc_step_um,
                "nyquist_arc_wavelength_um_ceramic": nyquist_arc_um,
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_metrics = [
        ("angular_step_deg_per_frame", "Angular step\n(deg/frame)", False),
        ("frames_per_degree", "Frames\nper degree", True),
        ("total_acquisition_duration_s", "Total duration\n(s)", True),
        ("total_pixel_samples", "Total pixel\nsamples", True),
    ]
    figure, axes = plt.subplots(1, len(plot_metrics), figsize=(11, 3.2), constrained_layout=True)
    colours = ["#0072B2", "#D55E00"]
    for axis, (column, title, use_log) in zip(axes, plot_metrics):
        axis.bar(summary["object"], summary[column], color=colours)
        axis.set_title(title)
        if use_log:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Acquisition-condition comparison for ceramic and steel measurements")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/acquisition_conditions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_rows()
    summary.to_csv(args.output_dir / "acquisition_sampling_summary.csv", index=False)
    write_plot(summary, args.output_dir / "acquisition_condition_comparison.png")


if __name__ == "__main__":
    main()