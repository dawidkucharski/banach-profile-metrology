"""Generate manuscript figures for phase, form, eccentricity and step extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.titlesize": 11,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def robust_limits(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite_values, [lower, upper])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(finite_values))
        high = float(np.nanmax(finite_values))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def normalised_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    row_count, column_count = shape
    x_axis = np.linspace(-1.0, 1.0, column_count)
    y_axis = np.linspace(-1.0, 1.0, row_count)
    return np.meshgrid(x_axis, y_axis)


def radial_profile(field: np.ndarray, centre: Tuple[float, float] | None = None) -> Dict[str, np.ndarray]:
    row_count, column_count = field.shape
    if centre is None:
        centre = ((column_count - 1.0) / 2.0, (row_count - 1.0) / 2.0)
    x_centre, y_centre = centre
    y_grid, x_grid = np.indices(field.shape)
    radius_map = np.sqrt((x_grid - x_centre) ** 2 + (y_grid - y_centre) ** 2)
    bin_index = np.floor(radius_map).astype(np.int64)
    max_index = int(np.nanmax(bin_index))
    weights = np.bincount(bin_index.ravel(), weights=field.ravel(), minlength=max_index + 1)
    counts = np.bincount(bin_index.ravel(), minlength=max_index + 1)
    profile = weights / np.maximum(counts, 1)
    return {
        "radius_map": radius_map,
        "radius": np.arange(max_index + 1, dtype=np.float64),
        "profile": profile,
    }


def phase_proxy(field: np.ndarray) -> Dict[str, np.ndarray]:
    smoothed = gaussian_filter(field, sigma=1.0)
    radial = radial_profile(smoothed)
    radius = radial["radius"]
    profile = radial["profile"]
    background = gaussian_filter1d(profile, sigma=20.0, mode="nearest")
    fringe_signal = profile - background
    signal_scale = float(np.nanstd(fringe_signal))
    if signal_scale <= 0 or not np.isfinite(signal_scale):
        signal_scale = 1.0
    fringe_signal = fringe_signal / signal_scale
    analytic_signal = hilbert(fringe_signal)
    wrapped = np.angle(analytic_signal)
    unwrapped = np.unwrap(wrapped)
    unwrapped = unwrapped - np.nanmedian(unwrapped)
    radius_map = radial["radius_map"]
    wrapped_map = np.interp(radius_map.ravel(), radius, wrapped).reshape(field.shape)
    unwrapped_map = np.interp(radius_map.ravel(), radius, unwrapped).reshape(field.shape)
    return {
        "radius": radius,
        "profile": profile,
        "fringe_signal": fringe_signal,
        "wrapped": wrapped,
        "unwrapped": unwrapped,
        "wrapped_map": wrapped_map,
        "unwrapped_map": unwrapped_map,
    }


def sample_fit_positions(mask: np.ndarray, max_pixels: int = 20000) -> np.ndarray:
    finite_count = int(np.sum(mask))
    if finite_count <= max_pixels:
        return np.flatnonzero(mask.ravel())
    stride = int(np.ceil(np.sqrt(finite_count / max_pixels)))
    sampled = np.zeros(mask.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    return np.flatnonzero((sampled & mask).ravel())


def fit_polynomial(field: np.ndarray, degree: int, max_pixels: int = 20000) -> Dict[str, np.ndarray]:
    x_grid, y_grid = normalised_grid(field.shape)
    terms = [np.ones_like(field)]
    if degree >= 1:
        terms.extend([x_grid, y_grid])
    if degree >= 2:
        terms.extend([x_grid**2, x_grid * y_grid, y_grid**2])
    design = np.stack([term.ravel() for term in terms], axis=1)
    mask = np.isfinite(field)
    positions = sample_fit_positions(mask, max_pixels=max_pixels)
    coefficients, *_ = np.linalg.lstsq(design[positions, :], field.ravel()[positions], rcond=None)
    fitted = (design @ coefficients).reshape(field.shape)
    return {"fitted": fitted, "residual": field - fitted, "coefficients": coefficients}


def detect_step(levelled: np.ndarray, edge_band_px: int = 8) -> Dict[str, np.ndarray | float | int]:
    profile = np.nanmean(levelled, axis=0)
    smoothed_profile = gaussian_filter1d(profile, sigma=2.0, mode="nearest")
    gradient = np.gradient(smoothed_profile)
    margin = edge_band_px + 2
    interior = np.arange(margin, len(gradient) - margin)
    edge_index = int(interior[np.argmax(np.abs(gradient[interior]))]) if interior.size else int(np.argmax(np.abs(gradient)))
    column_index = np.arange(levelled.shape[1])[None, :]
    left_mask = np.broadcast_to(column_index < edge_index - edge_band_px, levelled.shape)
    right_mask = np.broadcast_to(column_index > edge_index + edge_band_px, levelled.shape)
    edge_mask = ~(left_mask | right_mask)
    finite = np.isfinite(levelled)
    left_level = float(np.nanmedian(levelled[left_mask & finite]))
    right_level = float(np.nanmedian(levelled[right_mask & finite]))
    model = np.zeros_like(levelled, dtype=np.float64)
    model[left_mask] = left_level
    model[right_mask] = right_level
    model[edge_mask] = 0.5 * (left_level + right_level)
    return {
        "profile": profile,
        "smoothed_profile": smoothed_profile,
        "edge_index": edge_index,
        "edge_band_px": edge_band_px,
        "left_level": left_level,
        "right_level": right_level,
        "step_height": right_level - left_level,
        "model": model,
        "residual": levelled - model,
    }


def image_panel(axis: plt.Axes, data: np.ndarray, title: str, cmap: str = "viridis", full_scale: bool = False) -> None:
    if full_scale:
        low = float(np.nanmin(data))
        high = float(np.nanmax(data))
        if low == high:
            high = low + 1.0
    else:
        low, high = robust_limits(data)
    image = axis.imshow(data, cmap=cmap, vmin=low, vmax=high)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02)


def save_figure(figure: plt.Figure, path_stem: Path) -> None:
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


def write_phase_figure(ceramic_path: Path, steel_path: Path, output_dir: Path) -> None:
    datasets = [("Ceramic sphere", ceramic_path), ("Steel step", steel_path)]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.4), constrained_layout=True)
    for row_index, (label, path) in enumerate(datasets):
        field = load_grayscale(path)
        proxy = phase_proxy(field)
        image_panel(axes[row_index, 0], field, f"{label}: exported interferogram", cmap="gray", full_scale=True)
        image_panel(axes[row_index, 1], proxy["wrapped_map"], "Wrapped phase proxy, full -pi..pi scale", cmap="twilight", full_scale=True)
        image_panel(axes[row_index, 2], proxy["unwrapped_map"], "Unwrapped phase proxy, full scale", cmap="viridis", full_scale=True)
        axes[row_index, 3].plot(proxy["radius"], proxy["wrapped"], color="#444444", linewidth=0.8, label="wrapped")
        twin_axis = axes[row_index, 3].twinx()
        twin_axis.plot(proxy["radius"], proxy["unwrapped"], color="#0072B2", linewidth=0.9, label="unwrapped")
        axes[row_index, 3].set_title("Radial phase graph")
        axes[row_index, 3].set_xlabel("Radius, px")
        axes[row_index, 3].set_ylabel("Wrapped phase, rad")
        twin_axis.set_ylabel("Unwrapped phase proxy, rad")
        axes[row_index, 3].grid(True, alpha=0.25)
    figure.suptitle("Wrapped and unwrapped phase visualisation from representative exported interferograms")
    save_figure(figure, output_dir / "phase_wrapped_unwrapped_examples")
    plt.close(figure)


def write_ceramic_form_figure(ceramic_path: Path, output_dir: Path) -> None:
    field = load_grayscale(ceramic_path)
    proxy = phase_proxy(field)
    unwrapped = proxy["unwrapped_map"]
    plane = fit_polynomial(unwrapped, degree=1)
    levelled = plane["residual"]
    cap = fit_polynomial(levelled, degree=2)
    residual = cap["residual"]
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    image_panel(axes[0], field, "Exported interferogram", cmap="gray")
    image_panel(axes[1], unwrapped, "Unwrapped phase proxy", cmap="viridis")
    image_panel(axes[2], cap["fitted"], "Fitted quadratic cap", cmap="viridis")
    image_panel(axes[3], residual, "Form-removed residual", cmap="coolwarm")
    figure.suptitle("Ceramic object: topography proxy, cap removal and residual field")
    save_figure(figure, output_dir / "ceramic_form_removal")
    plt.close(figure)


def existing_indices(folder: Path, stop_index: int) -> np.ndarray:
    indices = []
    for path in folder.glob("*.png"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if index <= stop_index:
            indices.append(index)
    return np.array(sorted(indices), dtype=int)


def extract_phase_scalar(path: Path) -> float:
    field = load_grayscale(path)
    proxy = phase_proxy(field)
    unwrapped_map = proxy["unwrapped_map"]
    row_count, column_count = unwrapped_map.shape
    y_grid, x_grid = np.indices(unwrapped_map.shape)
    radius_map = np.sqrt((x_grid - (column_count - 1.0) / 2.0) ** 2 + (y_grid - (row_count - 1.0) / 2.0) ** 2)
    lower = 0.30 * np.nanmax(radius_map)
    upper = 0.70 * np.nanmax(radius_map)
    mask = (radius_map >= lower) & (radius_map <= upper)
    return float(np.nanmean(unwrapped_map[mask]))


def write_eccentricity_figure(
    ceramic_dir: Path,
    output_dir: Path,
    frames_per_sweep: int,
    angular_step_deg: float,
    sweep_count: int,
    sample_count: int,
) -> Dict[str, float]:
    selected_indices = []
    for sweep_index in range(sweep_count):
        sweep_start = sweep_index * frames_per_sweep
        sweep_stop = sweep_start + frames_per_sweep - 1
        indices = existing_indices(ceramic_dir, stop_index=sweep_stop)
        indices = indices[(indices >= sweep_start) & (indices <= sweep_stop)]
        if indices.size == 0:
            continue
        selected_positions = np.linspace(0, indices.size - 1, min(sample_count, indices.size), dtype=int)
        selected_indices.extend(indices[selected_positions].tolist())
    selected_indices = np.array(sorted(set(selected_indices)), dtype=int)
    if selected_indices.size == 0:
        raise FileNotFoundError(f"No ceramic PNG frames found for eccentricity analysis: {ceramic_dir}")
    values = np.array([extract_phase_scalar(ceramic_dir / f"{index:05d}.png") for index in selected_indices], dtype=np.float64)
    sweep_index = selected_indices // frames_per_sweep
    angle_deg = (selected_indices % frames_per_sweep) * angular_step_deg
    useful_mask = angle_deg < 360.0
    theta = np.deg2rad(angle_deg[useful_mask])
    sweep_used = sweep_index[useful_mask]
    values_used = np.unwrap(values[useful_mask])
    sweep_levels = np.zeros((values_used.size, sweep_count), dtype=np.float64)
    for column in range(sweep_count):
        sweep_levels[:, column] = sweep_used == column
    design = np.column_stack([sweep_levels, np.cos(theta), np.sin(theta)])
    coefficients, *_ = np.linalg.lstsq(design, values_used, rcond=None)
    sweep_offsets = coefficients[:sweep_count]
    harmonic = coefficients[sweep_count] * np.cos(theta) + coefficients[sweep_count + 1] * np.sin(theta)
    fitted = sweep_offsets[sweep_used] + harmonic
    raw_centred = values_used - sweep_offsets[sweep_used]
    corrected = values_used - fitted
    corrected = corrected - np.nanmean(corrected)
    ront = float(np.nanmax(corrected) - np.nanmin(corrected))
    ronq = float(np.sqrt(np.nanmean(corrected**2)))
    eccentricity_amplitude = float(np.sqrt(coefficients[sweep_count] ** 2 + coefficients[sweep_count + 1] ** 2))

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_profile = figure.add_subplot(grid[0, 0])
    axis_residual = figure.add_subplot(grid[1, 0])
    axis_polar_raw = figure.add_subplot(grid[0, 1], projection="polar")
    axis_polar_corrected = figure.add_subplot(grid[1, 1], projection="polar")

    for sweep in sorted(set(sweep_used)):
        mask = sweep_used == sweep
        axis_profile.plot(np.degrees(theta[mask]), raw_centred[mask], linewidth=0.75, alpha=0.55, label=f"sweep {sweep + 1}")
    angle_line = np.linspace(0.0, 360.0, 720)
    theta_line = np.deg2rad(angle_line)
    harmonic_line = coefficients[sweep_count] * np.cos(theta_line) + coefficients[sweep_count + 1] * np.sin(theta_line)
    axis_profile.plot(angle_line, harmonic_line, color="#D55E00", linewidth=1.5, label="common first-harmonic eccentricity")
    axis_profile.set_title("Eccentricity component fitted before roundness evaluation")
    axis_profile.set_xlabel("Rotation angle, deg")
    axis_profile.set_ylabel("Native phase-proxy units")
    axis_profile.grid(True, alpha=0.25)
    axis_profile.legend(loc="best", fontsize=7, ncol=2)

    for sweep in sorted(set(sweep_used)):
        mask = sweep_used == sweep
        axis_residual.plot(np.degrees(theta[mask]), corrected[mask], linewidth=0.75, alpha=0.65, label=f"sweep {sweep + 1}")
    axis_residual.axhline(0.0, color="#444444", linewidth=0.7)
    axis_residual.set_title("Roundness deviation after eccentricity removal")
    axis_residual.set_xlabel("Rotation angle, deg")
    axis_residual.set_ylabel("Residual, native units")
    axis_residual.grid(True, alpha=0.25)

    scale = 0.35 / max(np.nanmax(np.abs(raw_centred)), np.nanmax(np.abs(corrected)), 1.0)
    axis_polar_raw.plot(theta, 1.0 + scale * raw_centred, color="#444444", linewidth=0.55, alpha=0.65)
    axis_polar_raw.plot(theta_line, 1.0 + scale * harmonic_line, color="#D55E00", linewidth=1.2)
    axis_polar_raw.set_title("Five sweeps before eccentricity removal")
    axis_polar_raw.set_yticklabels([])
    axis_polar_corrected.plot(theta, 1.0 + scale * corrected, color="#0072B2", linewidth=0.55, alpha=0.65)
    axis_polar_corrected.set_title("Five sweeps after eccentricity removal")
    axis_polar_corrected.set_yticklabels([])

    figure.suptitle("Ceramic roundness route: extracted profile, eccentricity removal and roundness deviation")
    save_figure(figure, output_dir / "ceramic_eccentricity_roundness")
    plt.close(figure)
    return {
        "sample_count": float(selected_indices.size),
        "frames_per_sweep": float(frames_per_sweep),
        "sweep_count": float(sweep_count),
        "eccentricity_amplitude_native": eccentricity_amplitude,
        "roundness_deviation_RONt_native": ront,
        "roundness_deviation_RONq_native": ronq,
    }


def write_steel_step_figure(steel_path: Path, output_dir: Path) -> Dict[str, float]:
    field = load_grayscale(steel_path)
    plane = fit_polynomial(field, degree=1)
    levelled = plane["residual"]
    step = detect_step(levelled, edge_band_px=8)
    residual = step["residual"]
    edge_index = int(step["edge_index"])
    edge_band_px = int(step["edge_band_px"])

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    image_panel(axes[0, 0], field, "Exported steel interferogram", cmap="gray")
    image_panel(axes[0, 1], levelled, "After piston/tilt removal", cmap="viridis")
    axes[1, 0].plot(step["profile"], color="#444444", linewidth=0.7, label="column mean")
    axes[1, 0].plot(step["smoothed_profile"], color="#0072B2", linewidth=1.1, label="smoothed profile")
    axes[1, 0].axvline(edge_index, color="#D55E00", linewidth=1.2, label="detected edge")
    axes[1, 0].axvspan(edge_index - edge_band_px, edge_index + edge_band_px, color="#D55E00", alpha=0.18, label="excluded edge band")
    axes[1, 0].axhline(step["left_level"], color="#009E73", linestyle="--", linewidth=0.9, label="left median")
    axes[1, 0].axhline(step["right_level"], color="#CC79A7", linestyle="--", linewidth=0.9, label="right median")
    axes[1, 0].set_title("Step extraction from plateau levels")
    axes[1, 0].set_xlabel("Column index, px")
    axes[1, 0].set_ylabel("Levelled native units")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(loc="best", fontsize=7)
    image_panel(axes[1, 1], residual, "Two-plateau residual field", cmap="coolwarm")
    figure.suptitle("Steel object: edge detection, plateau modelling and residual extraction")
    save_figure(figure, output_dir / "steel_step_extraction")
    plt.close(figure)
    return {
        "edge_index_px": float(edge_index),
        "edge_band_px": float(edge_band_px),
        "left_level_native": float(step["left_level"]),
        "right_level_native": float(step["right_level"]),
        "step_height_native": float(step["step_height"]),
    }


def write_summary(path: Path, payload: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceramic-dir", type=Path, default=Path("ceramic"))
    parser.add_argument("--steel-dir", type=Path, default=Path("steel"))
    parser.add_argument("--ceramic-frame", default="00000.png")
    parser.add_argument("--steel-frame", default="00000.png")
    parser.add_argument("--output-dir", type=Path, default=Path("results/processing_figures"))
    parser.add_argument("--frames-per-sweep", type=int, default=4800)
    parser.add_argument("--angular-step-deg", type=float, default=0.08333333333333333)
    parser.add_argument("--sweep-count", type=int, default=5)
    parser.add_argument("--eccentricity-sample-count", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ceramic_path = args.ceramic_dir / args.ceramic_frame
    steel_path = args.steel_dir / args.steel_frame
    write_phase_figure(ceramic_path, steel_path, args.output_dir)
    write_ceramic_form_figure(ceramic_path, args.output_dir)
    eccentricity_summary = write_eccentricity_figure(
        args.ceramic_dir,
        args.output_dir,
        frames_per_sweep=args.frames_per_sweep,
        angular_step_deg=args.angular_step_deg,
        sweep_count=args.sweep_count,
        sample_count=args.eccentricity_sample_count,
    )
    steel_summary = write_steel_step_figure(steel_path, args.output_dir)
    write_summary(
        args.output_dir / "processing_figure_summary.json",
        {"ceramic_eccentricity": eccentricity_summary, "steel_step": steel_summary},
    )


if __name__ == "__main__":
    main()