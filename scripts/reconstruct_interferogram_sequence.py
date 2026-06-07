"""Reconstruct metric height maps from temporal interferogram sequences.

The input folders are treated as raw interferometric intensity stacks I(x, y, t),
not as phase or height maps. Wrapped phase is retrieved by temporal Fourier
demodulation, spatially unwrapped, converted to height in nanometres, and then
summarised with profile-style and Banach-space descriptors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image
from skimage.restoration import unwrap_phase

from btri.descriptors import banach_descriptors


DEFAULT_WAVELENGTH_NM = 532.0
SUPPORTED_IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MetricValue = Union[float, str]


def numeric_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def discover_images(folder: Path) -> List[Path]:
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if not files:
        raise ValueError(f"No supported interferogram images found in {folder}")
    return sorted(files, key=numeric_sort_key)


def select_frame_paths(paths: Sequence[Path], frame_limit: Optional[int], selection: str) -> List[Path]:
    if frame_limit is None or frame_limit <= 0 or len(paths) <= frame_limit:
        return list(paths)
    if selection == "first":
        return list(paths[:frame_limit])
    if selection == "stratified":
        indices = np.unique(np.linspace(0, len(paths) - 1, frame_limit, dtype=int))
        return [paths[int(index)] for index in indices]
    raise ValueError(f"Unsupported frame selection: {selection}")


def load_grayscale(path: Path, downsample: int = 1) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = np.mean(array[..., : min(3, array.shape[2])], axis=2)
    array = np.squeeze(array).astype(np.float64, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Interferogram must be two-dimensional after grayscale conversion: {path}")
    if downsample > 1:
        array = array[::downsample, ::downsample]
    return array


def deterministic_probe_pixels(shape: Tuple[int, int], target_count: int) -> Tuple[np.ndarray, np.ndarray]:
    row_count, column_count = shape
    grid_count = max(2, int(np.ceil(np.sqrt(target_count))))
    rows = np.linspace(0, row_count - 1, grid_count, dtype=int)
    columns = np.linspace(0, column_count - 1, grid_count, dtype=int)
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    return row_grid.ravel(), column_grid.ravel()


def estimate_temporal_frequency(
    paths: Sequence[Path],
    downsample: int,
    probe_pixels: int,
    frequency_min: int,
    frequency_max: Optional[int],
) -> Tuple[int, Dict[str, float]]:
    first = load_grayscale(paths[0], downsample=downsample)
    probe_rows, probe_columns = deterministic_probe_pixels(first.shape, target_count=probe_pixels)
    traces = np.empty((len(paths), probe_rows.size), dtype=np.float64)
    for frame_index, path in enumerate(paths):
        frame = load_grayscale(path, downsample=downsample)
        if frame.shape != first.shape:
            raise ValueError(f"Image shape changed from {first.shape} to {frame.shape} at {path}")
        traces[frame_index, :] = frame[probe_rows, probe_columns]

    traces -= np.mean(traces, axis=0, keepdims=True)
    spectrum = np.fft.rfft(traces, axis=0)
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    power[0] = 0.0
    max_bin = len(power) - 1 if frequency_max is None else min(int(frequency_max), len(power) - 1)
    min_bin = max(1, int(frequency_min))
    if min_bin > max_bin:
        raise ValueError(f"No valid temporal-frequency bins between {min_bin} and {max_bin}")
    selected_power = power[min_bin : max_bin + 1]
    frequency_bin = int(np.argmax(selected_power) + min_bin)
    total_power = float(np.sum(selected_power))
    confidence = float(power[frequency_bin] / total_power) if total_power > 0 else 0.0
    return frequency_bin, {
        "frequency_bin": float(frequency_bin),
        "frequency_cycles_per_selected_sequence": float(frequency_bin),
        "frequency_confidence_fraction": confidence,
        "probe_pixel_count": float(probe_rows.size),
    }


def retrieve_wrapped_phase(
    paths: Sequence[Path],
    frequency_bin: int,
    downsample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = load_grayscale(paths[0], downsample=downsample)
    shape = first.shape
    mean_sum = np.zeros(shape, dtype=np.float64)
    coefficient = np.zeros(shape, dtype=np.complex128)
    frame_count = len(paths)
    for frame_index, path in enumerate(paths):
        frame = load_grayscale(path, downsample=downsample)
        if frame.shape != shape:
            raise ValueError(f"Image shape changed from {shape} to {frame.shape} at {path}")
        phase_step = 2.0 * np.pi * frequency_bin * frame_index / frame_count
        coefficient += frame * np.exp(-1j * phase_step)
        mean_sum += frame
    mean_intensity = mean_sum / frame_count
    modulation_amplitude = 2.0 * np.abs(coefficient) / frame_count
    wrapped_phase = np.angle(coefficient)
    return wrapped_phase, mean_intensity, modulation_amplitude


def unwrap_wrapped_phase(wrapped_phase: np.ndarray, modulation: np.ndarray, mask_percentile: float) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(wrapped_phase) & np.isfinite(modulation)
    if mask_percentile > 0:
        threshold = float(np.nanpercentile(modulation[finite], mask_percentile))
        finite &= modulation >= threshold
    masked_wrapped = np.ma.array(wrapped_phase, mask=~finite)
    unwrapped = unwrap_phase(masked_wrapped)
    unwrapped_array = np.asarray(unwrapped.filled(np.nan), dtype=np.float64)
    valid = np.isfinite(unwrapped_array)
    if not np.any(valid):
        raise ValueError("Phase unwrapping produced no finite pixels")
    unwrapped_array -= float(np.nanmedian(unwrapped_array[valid]))
    return unwrapped_array, valid


def coordinate_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(shape, dtype=np.float64)
    x_coordinates = columns - 0.5 * (shape[1] - 1)
    y_coordinates = 0.5 * (shape[0] - 1) - rows
    return x_coordinates, y_coordinates


def design_matrix(shape: Tuple[int, int], model: str) -> np.ndarray:
    x_coordinates, y_coordinates = coordinate_grid(shape)
    x_coordinates = x_coordinates / max(1.0, float(np.nanmax(np.abs(x_coordinates))))
    y_coordinates = y_coordinates / max(1.0, float(np.nanmax(np.abs(y_coordinates))))
    terms = [np.ones(shape), x_coordinates, y_coordinates]
    if model == "quadratic":
        terms.extend([x_coordinates**2, x_coordinates * y_coordinates, y_coordinates**2])
    return np.stack([term.ravel() for term in terms], axis=1)


def fit_polynomial_reference(surface: np.ndarray, mask: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    matrix = design_matrix(surface.shape, model=model)
    values = surface.ravel()
    valid_flat = mask.ravel() & np.isfinite(values)
    coefficients, *_ = np.linalg.lstsq(matrix[valid_flat], values[valid_flat], rcond=None)
    fitted = (matrix @ coefficients).reshape(surface.shape)
    residual = surface - fitted
    residual[~mask] = np.nan
    return fitted, residual, {f"reference_coefficient_{index}": float(value) for index, value in enumerate(coefficients)}


def detect_step_edge(levelled: np.ndarray, mask: np.ndarray, edge_band: int) -> int:
    column_profile = np.nanmedian(np.where(mask, levelled, np.nan), axis=0)
    gradient = np.abs(np.gradient(column_profile))
    if gradient.size <= 2 * edge_band + 2:
        return int(gradient.size // 2)
    search = gradient[edge_band : gradient.size - edge_band]
    if np.all(~np.isfinite(search)):
        return int(gradient.size // 2)
    return int(np.nanargmax(search) + edge_band)


def fit_step_reference(surface: np.ndarray, mask: np.ndarray, edge_band: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    plane, levelled, plane_parameters = fit_polynomial_reference(surface, mask, model="plane")
    edge = detect_step_edge(levelled, mask=mask, edge_band=edge_band)
    columns = np.arange(surface.shape[1])
    left_mask = mask & (columns[None, :] < edge - edge_band)
    right_mask = mask & (columns[None, :] > edge + edge_band)
    if not np.any(left_mask) or not np.any(right_mask):
        edge = surface.shape[1] // 2
        left_mask = mask & (columns[None, :] < edge)
        right_mask = mask & (columns[None, :] >= edge)
    left_level = float(np.nanmedian(levelled[left_mask]))
    right_level = float(np.nanmedian(levelled[right_mask]))
    step_model_levelled = np.where(columns[None, :] <= edge, left_level, right_level)
    fitted = plane + step_model_levelled
    residual = surface - fitted
    residual[~mask] = np.nan
    parameters = dict(plane_parameters)
    parameters.update(
        {
            "step_edge_column": float(edge),
            "step_left_level_nm": left_level,
            "step_right_level_nm": right_level,
            "step_height_nm": right_level - left_level,
            "step_depth_nm": abs(right_level - left_level),
        }
    )
    return fitted, residual, parameters


def choose_reference_model(label: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = label.lower()
    if "steel" in lowered or "step" in lowered:
        return "step"
    if "ceramic" in lowered or "sphere" in lowered or "spherical" in lowered:
        return "quadratic"
    return "plane"


def profile_metrics(values: np.ndarray, prefix: str) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_Ra_nm": np.nan, f"{prefix}_Rq_nm": np.nan, f"{prefix}_Rz_nm": np.nan}
    centred = finite - float(np.mean(finite))
    peak_positive = float(np.max(centred))
    peak_negative = float(np.min(centred))
    return {
        f"{prefix}_mean_nm": float(np.mean(finite)),
        f"{prefix}_Ra_nm": float(np.mean(np.abs(centred))),
        f"{prefix}_Rq_nm": float(np.sqrt(np.mean(centred**2))),
        f"{prefix}_Rp_nm": peak_positive,
        f"{prefix}_Rv_nm": float(abs(peak_negative)),
        f"{prefix}_Rz_nm": float(peak_positive - peak_negative),
    }


def compute_metrics(
    height_nm: np.ndarray,
    residual_nm: np.ndarray,
    mask: np.ndarray,
    reference_parameters: Dict[str, float],
    gradient_spacing_px: Tuple[float, float],
) -> Dict[str, float]:
    height_values = height_nm[mask & np.isfinite(height_nm)]
    residual_values = residual_nm[mask & np.isfinite(residual_nm)]
    centre_row = residual_nm[residual_nm.shape[0] // 2, :]
    centre_column = residual_nm[:, residual_nm.shape[1] // 2]
    metrics: Dict[str, float] = {
        "height_mean_nm": float(np.mean(height_values)),
        "height_std_nm": float(np.std(height_values)),
        "height_min_nm": float(np.min(height_values)),
        "height_max_nm": float(np.max(height_values)),
        "residual_median_nm": float(np.median(residual_values)),
        "residual_mad_nm": float(np.median(np.abs(residual_values - np.median(residual_values)))),
    }
    metrics.update(profile_metrics(residual_values, prefix="profile_all_pixels"))
    metrics.update(profile_metrics(centre_row, prefix="profile_centre_row"))
    metrics.update(profile_metrics(centre_column, prefix="profile_centre_column"))
    metrics.update(reference_parameters)
    for key, value in banach_descriptors(residual_nm, gradient_spacing=gradient_spacing_px).items():
        metrics[f"{key}_nm"] = float(value)
    return metrics


def angular_roundness(residual_nm: np.ndarray, mask: np.ndarray, bin_count: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    row_indices, column_indices = np.indices(residual_nm.shape, dtype=np.float64)
    valid = mask & np.isfinite(residual_nm)
    if int(valid.sum()) == 0:
        raise ValueError("No valid residual pixels available for angular roundness extraction")
    row_centre = float(np.mean(row_indices[valid]))
    column_centre = float(np.mean(column_indices[valid]))
    angles = np.mod(np.arctan2(row_centre - row_indices, column_indices - column_centre), 2.0 * np.pi)
    bin_indices = np.floor(angles[valid] / (2.0 * np.pi) * bin_count).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    sums = np.bincount(bin_indices, weights=residual_nm[valid], minlength=bin_count)
    counts = np.bincount(bin_indices, minlength=bin_count)
    profile = np.full(bin_count, np.nan, dtype=np.float64)
    profile[counts > 0] = sums[counts > 0] / counts[counts > 0]

    theta = (np.arange(bin_count, dtype=np.float64) + 0.5) * (2.0 * np.pi / bin_count)
    valid_profile = np.isfinite(profile)
    if int(valid_profile.sum()) < 4:
        raise ValueError("Not enough angular bins to estimate roundness profile")
    design = np.column_stack(
        [np.ones(int(valid_profile.sum())), np.cos(theta[valid_profile]), np.sin(theta[valid_profile])]
    )
    coefficients, *_ = np.linalg.lstsq(design, profile[valid_profile], rcond=None)
    eccentricity_fit = coefficients[0] + coefficients[1] * np.cos(theta) + coefficients[2] * np.sin(theta)
    roundness_residual = profile - eccentricity_fit
    residual_values = roundness_residual[np.isfinite(roundness_residual)]
    metrics = {
        "roundness_bin_count": float(bin_count),
        "roundness_valid_bin_count": float(valid_profile.sum()),
        "roundness_profile_mean_nm": float(coefficients[0]),
        "roundness_eccentricity_cos_nm": float(coefficients[1]),
        "roundness_eccentricity_sin_nm": float(coefficients[2]),
        "roundness_eccentricity_amplitude_nm": float(np.sqrt(coefficients[1] ** 2 + coefficients[2] ** 2)),
        "roundness_RONt_nm": float(np.nanmax(residual_values) - np.nanmin(residual_values)),
        "roundness_RONq_nm": float(np.sqrt(np.nanmean(residual_values**2))),
    }
    profile_table = pd.DataFrame(
        {
            "angle_deg": np.degrees(theta),
            "angular_mean_residual_nm": profile,
            "first_harmonic_eccentricity_fit_nm": eccentricity_fit,
            "roundness_residual_nm": roundness_residual,
            "pixel_count": counts,
        }
    )
    return profile_table, metrics


def robust_limits(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return 0.0, 1.0
    low_value, high_value = np.percentile(values, [lower, upper])
    if low_value == high_value:
        low_value = float(np.min(values))
        high_value = float(np.max(values))
    if low_value == high_value:
        high_value = low_value + 1.0
    return float(low_value), float(high_value)


def downsample_for_vector_plot(array: np.ndarray, max_dimension: int = 180) -> np.ndarray:
    stride = max(1, int(np.ceil(max(array.shape) / max_dimension)))
    return array[::stride, ::stride]


def contour_panel(axis: Axes, array: np.ndarray, title: str, colour_map: str, unit: str) -> None:
    plot_array = downsample_for_vector_plot(np.asarray(array, dtype=np.float64))
    low_value, high_value = robust_limits(plot_array)
    levels = np.linspace(low_value, high_value, 42)
    row_axis = np.arange(plot_array.shape[0])
    column_axis = np.arange(plot_array.shape[1])
    contour = axis.contourf(column_axis, row_axis, plot_array, levels=levels, cmap=colour_map, extend="both")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    axis.invert_yaxis()
    plt.colorbar(contour, ax=axis, shrink=0.78, label=unit)


def write_maps_pdf(
    output_path: Path,
    label: str,
    wrapped_phase: np.ndarray,
    unwrapped_phase: np.ndarray,
    height_nm: np.ndarray,
    residual_nm: np.ndarray,
    modulation: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    panels = [
        (wrapped_phase, "Wrapped phase", "twilight", "rad"),
        (unwrapped_phase, "Unwrapped phase", "viridis", "rad"),
        (height_nm, "Height", "viridis", "nm"),
        (residual_nm, "Reference-removed residual", "coolwarm", "nm"),
        (modulation, "Temporal modulation", "magma", "intensity"),
    ]
    for axis, (array, title, colour_map, unit) in zip(axes.ravel(), panels):
        contour_panel(axis, array=array, title=title, colour_map=colour_map, unit=unit)
    axes.ravel()[-1].axis("off")
    figure.suptitle(f"{label}: interferogram sequence reconstruction")
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_roundness_pdf(output_path: Path, label: str, profile_table: pd.DataFrame, metrics: Dict[str, float]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(profile_table["angle_deg"], profile_table["angular_mean_residual_nm"], color="#555555", linewidth=0.9, label="angular mean")
    axes[0].plot(
        profile_table["angle_deg"],
        profile_table["first_harmonic_eccentricity_fit_nm"],
        color="#D55E00",
        linewidth=0.9,
        label="first harmonic",
    )
    axes[0].plot(profile_table["angle_deg"], profile_table["roundness_residual_nm"], color="#0072B2", linewidth=0.9, label="roundness residual")
    axes[0].set_title("Eccentricity reduction")
    axes[0].set_xlabel("angle, deg")
    axes[0].set_ylabel("nm")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    summary_labels = ["eccentricity", "RONt", "RONq"]
    summary_values = [
        metrics["roundness_eccentricity_amplitude_nm"],
        metrics["roundness_RONt_nm"],
        metrics["roundness_RONq_nm"],
    ]
    axes[1].bar(summary_labels, summary_values, color=["#D55E00", "#0072B2", "#4C78A8"])
    axes[1].set_title("Roundness summary")
    axes[1].set_ylabel("nm")
    axes[1].grid(True, axis="y", alpha=0.25)
    figure.suptitle(f"{label}: nm roundness profile after first-harmonic eccentricity removal")
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_comparison_pdf(output_path: Path, comparison: pd.DataFrame) -> None:
    metric_names = [
        "profile_all_pixels_Ra_nm",
        "profile_all_pixels_Rq_nm",
        "profile_all_pixels_Rz_nm",
        "roundness_RONt_nm",
        "roundness_RONq_nm",
        "banach_l1_mean_nm",
        "banach_l2_rms_nm",
        "banach_linf_nm",
    ]
    available = [metric for metric in metric_names if metric in comparison.columns and comparison[metric].notna().any()]
    if not available:
        return
    figure, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    x_positions = np.arange(len(available))
    width = 0.8 / max(1, len(comparison))
    for index, (_, row) in enumerate(comparison.iterrows()):
        offset = (index - 0.5 * (len(comparison) - 1)) * width
        values = [float(row[metric]) if pd.notna(row[metric]) else np.nan for metric in available]
        axis.bar(x_positions + offset, values, width, label=row["sample"])
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            metric.replace("profile_all_pixels_", "")
            .replace("roundness_", "")
            .replace("banach_", "B ")
            .replace("_nm", "")
            for metric in available
        ],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("nm")
    axis.set_title("Profile-style and Banach descriptors from reconstructed height")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def process_folder(
    folder: Path,
    label: str,
    output_root: Path,
    wavelength_nm: float,
    frame_limit: Optional[int],
    frame_selection: str,
    temporal_frequency: Optional[int],
    frequency_min: int,
    frequency_max: Optional[int],
    downsample: int,
    probe_pixels: int,
    modulation_mask_percentile: float,
    reference_model: str,
    edge_band: int,
    roundness_angular_bins: int,
) -> Dict[str, MetricValue]:
    all_paths = discover_images(folder)
    selected_paths = select_frame_paths(all_paths, frame_limit=frame_limit, selection=frame_selection)
    if len(selected_paths) < 4:
        raise ValueError(f"At least four frames are required for temporal phase retrieval, got {len(selected_paths)} in {folder}")
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    if temporal_frequency is None:
        frequency_bin, frequency_info = estimate_temporal_frequency(
            selected_paths,
            downsample=downsample,
            probe_pixels=probe_pixels,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
        )
    else:
        frequency_bin = int(temporal_frequency)
        frequency_info = {
            "frequency_bin": float(frequency_bin),
            "frequency_cycles_per_selected_sequence": float(frequency_bin),
            "frequency_confidence_fraction": np.nan,
            "probe_pixel_count": float(probe_pixels),
        }

    wrapped_phase, mean_intensity, modulation = retrieve_wrapped_phase(selected_paths, frequency_bin=frequency_bin, downsample=downsample)
    unwrapped_phase, mask = unwrap_wrapped_phase(wrapped_phase, modulation=modulation, mask_percentile=modulation_mask_percentile)
    excess_fraction = unwrapped_phase / (2.0 * np.pi)
    height_nm = unwrapped_phase * wavelength_nm / (4.0 * np.pi)

    selected_reference_model = choose_reference_model(label, requested=reference_model)
    if selected_reference_model == "step":
        reference_nm, residual_nm, reference_parameters = fit_step_reference(height_nm, mask=mask, edge_band=edge_band)
    else:
        reference_nm, residual_nm, reference_parameters = fit_polynomial_reference(
            height_nm,
            mask=mask,
            model=selected_reference_model,
        )

    metrics = compute_metrics(
        height_nm=height_nm,
        residual_nm=residual_nm,
        mask=mask,
        reference_parameters=reference_parameters,
        gradient_spacing_px=(float(downsample), float(downsample)),
    )
    if selected_reference_model == "quadratic":
        roundness_profile, roundness_metrics = angular_roundness(
            residual_nm,
            mask=mask,
            bin_count=roundness_angular_bins,
        )
        metrics.update(roundness_metrics)
    else:
        roundness_profile = None
    metadata: Dict[str, MetricValue] = {
        "sample": label,
        "input_folder": str(folder),
        "total_available_frames": float(len(all_paths)),
        "processed_frames": float(len(selected_paths)),
        "frame_selection": frame_selection,
        "downsample": float(downsample),
        "wavelength_nm": float(wavelength_nm),
        "phase_to_height_scale_nm_per_rad": float(wavelength_nm / (4.0 * np.pi)),
        "fringe_order_to_height_scale_nm": float(wavelength_nm / 2.0),
        "reference_model": selected_reference_model,
        "phase_retrieval_method": "temporal Fourier demodulation of intensity sequence",
        "phase_unwrapping_method": "skimage.restoration.unwrap_phase",
        **frequency_info,
    }
    full_metrics: Dict[str, MetricValue] = {**metadata, **metrics}

    save_array(output_dir / "wrapped_phase_rad.npy", wrapped_phase)
    save_array(output_dir / "unwrapped_phase_rad.npy", unwrapped_phase)
    save_array(output_dir / "excess_fraction.npy", excess_fraction)
    save_array(output_dir / "height_nm.npy", height_nm)
    save_array(output_dir / "reference_model_nm.npy", reference_nm)
    save_array(output_dir / "height_residual_nm.npy", residual_nm)
    save_array(output_dir / "mean_intensity.npy", mean_intensity)
    save_array(output_dir / "modulation_amplitude.npy", modulation)
    pd.DataFrame([full_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(full_metrics, handle, indent=2)
    write_maps_pdf(
        output_dir / "reconstruction_maps.pdf",
        label=label,
        wrapped_phase=wrapped_phase,
        unwrapped_phase=unwrapped_phase,
        height_nm=height_nm,
        residual_nm=residual_nm,
        modulation=modulation,
    )
    if roundness_profile is not None:
        roundness_profile.to_csv(output_dir / "roundness_profile.csv", index=False)
        write_roundness_pdf(
            output_dir / "roundness_profile.pdf",
            label=label,
            profile_table=roundness_profile,
            metrics=metrics,
        )
    return full_metrics


def parse_dataset(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        return label.strip(), Path(path_text).expanduser()
    path = Path(value).expanduser()
    return path.name, path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct phase and nm height maps from interferogram image sequences.")
    parser.add_argument("--dataset", action="append", required=True, help="Dataset as label=folder or folder. Repeat for each sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/interferogram_phase_reconstruction_metric"))
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM)
    parser.add_argument("--frame-limit", type=int, default=512, help="Frames processed per folder; use 0 for all frames.")
    parser.add_argument("--frame-selection", choices=["stratified", "first"], default="stratified")
    parser.add_argument("--temporal-frequency", type=int, default=None, help="DFT frequency bin; auto-estimated when omitted.")
    parser.add_argument("--frequency-min", type=int, default=1)
    parser.add_argument("--frequency-max", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=1, help="Integer image downsampling factor for reconstruction.")
    parser.add_argument("--probe-pixels", type=int, default=256)
    parser.add_argument("--modulation-mask-percentile", type=float, default=1.0)
    parser.add_argument("--reference-model", choices=["auto", "plane", "quadratic", "step"], default="auto")
    parser.add_argument("--edge-band", type=int, default=8)
    parser.add_argument("--roundness-angular-bins", type=int, default=720, help="Angular bins for ceramic RONt/RONq extraction")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_limit = None if args.frame_limit <= 0 else int(args.frame_limit)
    rows = []
    for label, folder in [parse_dataset(item) for item in args.dataset]:
        rows.append(
            process_folder(
                folder=folder,
                label=label,
                output_root=args.output_dir,
                wavelength_nm=float(args.wavelength_nm),
                frame_limit=frame_limit,
                frame_selection=args.frame_selection,
                temporal_frequency=args.temporal_frequency,
                frequency_min=int(args.frequency_min),
                frequency_max=args.frequency_max,
                downsample=max(1, int(args.downsample)),
                probe_pixels=max(4, int(args.probe_pixels)),
                modulation_mask_percentile=max(0.0, float(args.modulation_mask_percentile)),
                reference_model=args.reference_model,
                edge_band=max(1, int(args.edge_band)),
                roundness_angular_bins=max(16, int(args.roundness_angular_bins)),
            )
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "sample_comparison_metrics.csv", index=False)
    with (args.output_dir / "sample_comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_comparison_pdf(args.output_dir / "sample_comparison_metrics.pdf", comparison)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": [row["sample"] for row in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
