"""Reconstruct nm profiles from interferogram images using radial fringe processing.

Each input image is a fringe pattern. The phase/excess fraction is decoded from
that single image by radial fringe analysis. A ring-diameter excess-fraction fit
initialises the phase period, and a full-profile sinusoidal radial fit refines
the wrapped excess fraction. The sequence of decoded values is then unwrapped
over the scan/rotation order and converted to nanometres with the reflection
half-wavelength relation d = epsilon * lambda / 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import t as student_t


DEFAULT_WAVELENGTH_NM = 532.0
MIN_SINUSOIDAL_REFINEMENT_R2 = 0.65
MAX_REFINEMENT_RING_DISTANCE = 0.20
SUPPORTED_IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MetricValue = Union[float, str]
_RADIAL_BIN_CACHE: Dict[Tuple[Tuple[int, int], int, Tuple[float, float]], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def numeric_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def numeric_frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def discover_images(folder: Path) -> List[Path]:
    paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if not paths:
        raise ValueError(f"No supported interferogram images found in {folder}")
    return sorted(paths, key=numeric_sort_key)


def select_frame_paths(
    paths: Sequence[Path],
    frame_limit: Optional[int],
    selection: str,
    sweep_count: int,
    per_sweep_frame_limit: Optional[int] = None,
) -> List[Path]:
    if per_sweep_frame_limit is not None and per_sweep_frame_limit > 0:
        frames_per_sweep = max(1, int(round(len(paths) / max(1, sweep_count))))
        selected: List[Path] = []
        for sweep in range(max(1, sweep_count)):
            start = sweep * frames_per_sweep
            stop = min(len(paths), start + frames_per_sweep)
            sweep_paths = list(paths[start:stop])
            if not sweep_paths:
                continue
            if selection == "first":
                selected.extend(sweep_paths[:per_sweep_frame_limit])
            elif selection == "stratified":
                count = min(per_sweep_frame_limit, len(sweep_paths))
                indices = np.unique(np.linspace(0, len(sweep_paths) - 1, count, dtype=int))
                selected.extend([sweep_paths[int(index)] for index in indices])
            else:
                raise ValueError(f"Unsupported frame selection: {selection}")
        return selected
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
        array = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if downsample > 1:
        array = array[::downsample, ::downsample]
    return array


def radial_profile(field: np.ndarray, max_radius_px: int, centre: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    row_count, column_count = field.shape
    if centre is None:
        centre = ((column_count - 1.0) / 2.0, (row_count - 1.0) / 2.0)
    x_centre, y_centre = centre
    cache_key = (field.shape, int(max_radius_px), (round(x_centre, 6), round(y_centre, 6)))
    cached = _RADIAL_BIN_CACHE.get(cache_key)
    if cached is None:
        rows, columns = np.indices(field.shape, dtype=np.float64)
        radius_map = np.sqrt((columns - x_centre) ** 2 + (rows - y_centre) ** 2)
        bin_index = np.floor(radius_map).astype(np.int64)
        max_index = min(int(np.nanmax(bin_index)), int(max_radius_px))
        valid = bin_index <= max_index
        flat_indices = np.flatnonzero(valid.ravel())
        flat_bin_index = bin_index.ravel()[flat_indices]
        counts = np.bincount(bin_index[valid].ravel(), minlength=max_index + 1)
        _RADIAL_BIN_CACHE[cache_key] = (flat_indices, flat_bin_index, counts)
    else:
        flat_indices, flat_bin_index, counts = cached
        max_index = counts.size - 1
    sums = np.bincount(flat_bin_index, weights=field.ravel()[flat_indices], minlength=max_index + 1)
    profile = sums / np.maximum(counts, 1)
    return {"radius": np.arange(max_index + 1, dtype=np.float64), "profile": profile, "counts": counts}


def decimate_xy(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size <= max_points:
        return x, y
    indices = np.unique(np.linspace(0, x.size - 1, max_points, dtype=int))
    return x[indices], y[indices]


def plot_series(axis: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs: MetricValue) -> None:
    x_plot, y_plot = decimate_xy(np.asarray(x), np.asarray(y))
    axis.plot(x_plot, y_plot, **kwargs)


def plot_sweep_angle_series(axis: plt.Axes, table: pd.DataFrame, y_column: str, **kwargs: MetricValue) -> None:
    if "angle_deg" not in table.columns or "sweep_index" not in table.columns:
        x = table["frame_order"].to_numpy(dtype=np.float64) if "frame_order" in table.columns else np.arange(len(table), dtype=np.float64)
        plot_series(axis, x, table[y_column].to_numpy(dtype=np.float64), **kwargs)
        return
    label = kwargs.pop("label", None)
    for index, (_, group) in enumerate(table.groupby("sweep_index", sort=True)):
        group = group.sort_values("angle_deg")
        local_kwargs = dict(kwargs)
        if label is not None and index == 0:
            local_kwargs["label"] = label
        elif label is not None:
            local_kwargs["label"] = "_nolegend_"
        plot_series(
            axis,
            group["angle_deg"].to_numpy(dtype=np.float64),
            group[y_column].to_numpy(dtype=np.float64),
            **local_kwargs,
        )


def spline_smooth_radial_signal(radius: np.ndarray, signal: np.ndarray, smoothing_factor: float) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(radius) & np.isfinite(signal)
    if valid.sum() < 8:
        return gaussian_filter1d(np.nan_to_num(signal, nan=float(np.nanmedian(signal))), sigma=1.0, mode="nearest")
    variance = float(np.nanvar(signal[valid]))
    if not np.isfinite(variance) or variance <= 0:
        return np.full_like(signal, float(np.nanmean(signal[valid])))
    smoothing = max(1e-12, float(smoothing_factor) * float(valid.sum()) * variance)
    try:
        spline = UnivariateSpline(radius[valid], signal[valid], s=smoothing, k=3)
        smoothed = spline(radius)
    except Exception:
        filled = signal.copy()
        filled[~valid] = np.interp(radius[~valid], radius[valid], signal[valid])
        smoothed = gaussian_filter1d(filled, sigma=1.0, mode="nearest")
    return np.asarray(smoothed, dtype=np.float64)


def circular_distance(a: float, b: float) -> float:
    return abs(((a - b + 0.5) % 1.0) - 0.5)


def unwrap_near(value: float, reference: float) -> float:
    return float(value + np.round(reference - value))


def _fast_linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Closed-form OLS slope and intercept for y = a*x + b."""
    n = float(x.size)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.dot(x, y))
    sum_x2 = float(np.dot(x, x))
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator <= 0:
        return np.nan, np.nan
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def fit_ring_diameter_sequence(diameter_squared: np.ndarray) -> Dict[str, np.ndarray | float]:
    diameter_squared = np.asarray(diameter_squared, dtype=np.float64)
    if diameter_squared.size < 2:
        raise ValueError("At least two ring diameters are required")
    adjacent_delta = np.diff(diameter_squared)
    positive_delta = adjacent_delta[np.isfinite(adjacent_delta) & (adjacent_delta > 0)]
    if positive_delta.size == 0:
        raise ValueError("Ring diameters are not strictly increasing")

    naive_orders = np.arange(1, diameter_squared.size + 1, dtype=np.float64)
    candidate_spacings: set[float] = set()
    for delta in positive_delta:
        candidate_spacings.add(float(delta))
        for divisor in range(2, 7):
            candidate_spacings.add(float(delta) / float(divisor))

    best: Optional[Dict[str, np.ndarray | float]] = None
    for spacing in sorted(candidate_spacings):
        if not np.isfinite(spacing) or spacing <= 0:
            continue
        increments = np.rint(adjacent_delta / spacing).astype(int)
        increments = np.clip(increments, 1, 8)
        orders = np.concatenate([[1], 1 + np.cumsum(increments)]).astype(np.float64)
        slope, intercept = _fast_linear_fit(orders, diameter_squared)
        if not np.isfinite(slope) or slope <= 0:
            continue
        fitted = slope * orders + intercept
        residual = diameter_squared - fitted
        total = diameter_squared - float(np.mean(diameter_squared))
        r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else np.nan
        missed_count = float(np.sum(increments - 1))
        residual_order_rmse = float(np.sqrt(np.mean((residual / slope) ** 2)))
        spacing_error = abs(float(slope) / float(spacing) - 1.0)
        score = float(r_squared) - 0.04 * missed_count - 0.08 * residual_order_rmse - 0.03 * spacing_error
        candidate: Dict[str, np.ndarray | float] = {
            "orders": orders,
            "increments": increments.astype(np.float64),
            "slope": float(slope),
            "intercept": float(intercept),
            "fitted": fitted,
            "r_squared": float(r_squared),
            "missed_count": missed_count,
            "max_increment": float(np.max(increments)),
            "residual_order_rmse": residual_order_rmse,
            "score": score,
            "candidate_spacing": float(spacing),
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate

    if best is None:
        slope, intercept = _fast_linear_fit(naive_orders, diameter_squared)
        fitted = slope * naive_orders + intercept
        total = diameter_squared - float(np.mean(diameter_squared))
        residual = diameter_squared - fitted
        r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else np.nan
        best = {
            "orders": naive_orders,
            "increments": np.ones(max(0, diameter_squared.size - 1), dtype=np.float64),
            "slope": float(slope),
            "intercept": float(intercept),
            "fitted": fitted,
            "r_squared": float(r_squared),
            "missed_count": 0.0,
            "max_increment": 1.0,
            "residual_order_rmse": float(np.sqrt(np.mean((residual / slope) ** 2))) if slope else np.nan,
            "score": float(r_squared),
            "candidate_spacing": float(np.nanmedian(positive_delta)),
        }
    return best


def contiguous_peak_windows(peaks: np.ndarray, min_rings: int, max_window_count: int = 60) -> List[np.ndarray]:
    peaks = np.asarray(peaks, dtype=int)
    windows: List[np.ndarray] = []
    seen: set[Tuple[int, ...]] = set()
    max_rings = min(peaks.size, 7)

    def append_window(indices: Sequence[int]) -> None:
        key = tuple(int(index) for index in indices)
        if key in seen:
            return
        seen.add(key)
        windows.append(peaks[list(key)])

    for count in range(min_rings, max_rings + 1):
        for start in range(0, peaks.size - count + 1):
            append_window(range(start, start + count))

    for count in range(min_rings, max_rings + 1):
        span = count + 1
        if span > peaks.size:
            continue
        for start in range(0, peaks.size - span + 1):
            span_indices = list(range(start, start + span))
            for omitted in range(span):
                append_window(span_indices[:omitted] + span_indices[omitted + 1 :])

    if len(windows) <= max_window_count:
        return windows
    windows = sorted(windows, key=lambda item: (-item.size, int(item[0]), int(item[-1])))
    return windows[:max_window_count]


def add_central_endpoint_candidate(
    peaks: np.ndarray,
    radius: np.ndarray,
    min_radius_px: int,
    peak_distance_px: int,
) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return peaks
    central_exclusion_radius = max(float(min_radius_px + peak_distance_px), float(4 * peak_distance_px))
    if np.any(radius[peaks] <= central_exclusion_radius):
        return peaks
    return np.unique(np.concatenate([np.array([0], dtype=int), peaks])).astype(int)


def score_ring_candidate(
    epsilon: float,
    ring_fit: Dict[str, np.ndarray | float],
    ring_count: int,
    expected_epsilon_unwrapped: Optional[float],
    phase_tracking_weight: float,
) -> Tuple[float, float, float]:
    r_squared = float(ring_fit["r_squared"])
    order_rmse = float(ring_fit["residual_order_rmse"])
    missed_count = float(ring_fit["missed_count"])
    quality_score = r_squared - 0.10 * order_rmse - 0.03 * missed_count + 0.005 * ring_count
    if expected_epsilon_unwrapped is None or not np.isfinite(expected_epsilon_unwrapped):
        return quality_score, np.nan, float(epsilon)
    unwrapped = unwrap_near(float(epsilon), float(expected_epsilon_unwrapped))
    continuity_error = abs(unwrapped - float(expected_epsilon_unwrapped))
    return quality_score - phase_tracking_weight * continuity_error, continuity_error, unwrapped


def refine_epsilon_with_radial_sinusoid(
    radius: np.ndarray,
    smoothed: np.ndarray,
    ring_fit_slope: float,
    ring_epsilon: float,
    min_radius_px: int,
) -> Dict[str, float]:
    diameter_squared = (2.0 * radius) ** 2
    valid = np.isfinite(smoothed) & (radius >= min_radius_px)
    if valid.sum() < 12 or not np.isfinite(ring_fit_slope) or ring_fit_slope <= 0:
        return {"epsilon": ring_epsilon, "r_squared": np.nan, "contrast": np.nan, "phase_shift_rad": np.nan, "used_refinement": 0.0}
    # Decimate to keep sinusoidal refinement fast; the phase argument varies
    # quadratically with radius so the signal is well oversampled.
    MAX_REFINEMENT_POINTS = 4000
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size > MAX_REFINEMENT_POINTS:
        step = max(1, valid_indices.size // MAX_REFINEMENT_POINTS)
        valid_indices = valid_indices[::step]
    phase_argument = 2.0 * np.pi * diameter_squared[valid_indices] / ring_fit_slope
    design = np.column_stack([np.ones(valid_indices.size, dtype=np.float64), np.cos(phase_argument), np.sin(phase_argument)])
    coefficients, *_ = np.linalg.lstsq(design, smoothed[valid_indices], rcond=None)
    fitted = design @ coefficients
    residual = smoothed[valid_indices] - fitted
    total = smoothed[valid_indices] - float(np.mean(smoothed[valid_indices]))
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else np.nan
    cos_coefficient = float(coefficients[1])
    sin_coefficient = float(coefficients[2])
    contrast = float(np.hypot(cos_coefficient, sin_coefficient))
    phase_shift = float(np.arctan2(sin_coefficient, cos_coefficient))
    epsilon_candidate = (phase_shift / (2.0 * np.pi)) % 1.0
    inverted_candidate = (epsilon_candidate + 0.5) % 1.0
    epsilon = epsilon_candidate if circular_distance(epsilon_candidate, ring_epsilon) <= circular_distance(inverted_candidate, ring_epsilon) else inverted_candidate
    used_refinement = 1.0
    if (
        not np.isfinite(r_squared)
        or r_squared < MIN_SINUSOIDAL_REFINEMENT_R2
        or circular_distance(epsilon, ring_epsilon) > MAX_REFINEMENT_RING_DISTANCE
    ):
        epsilon = ring_epsilon
        used_refinement = 0.0
    return {
        "epsilon": float(epsilon),
        "r_squared": float(r_squared),
        "contrast": contrast,
        "phase_shift_rad": phase_shift,
        "used_refinement": used_refinement,
    }


def fit_excess_fraction_from_rings(
    field: np.ndarray,
    max_radius_px: int,
    min_radius_px: int,
    min_rings: int,
    peak_distance_px: int,
    background_sigma_px: float,
    smooth_sigma_px: float,
    peak_spline_smoothing_factor: float,
    peak_prominence_fraction: float,
    expected_epsilon_unwrapped: Optional[float] = None,
    phase_tracking_weight: float = 3.0,
) -> Dict[str, MetricValue]:
    radial = radial_profile(field, max_radius_px=max_radius_px)
    radius = radial["radius"]
    profile = radial["profile"]
    background = gaussian_filter1d(profile, sigma=background_sigma_px, mode="nearest")
    fringe = profile - background
    gaussian_smoothed = gaussian_filter1d(fringe, sigma=smooth_sigma_px, mode="nearest")
    smoothed = spline_smooth_radial_signal(radius, gaussian_smoothed, smoothing_factor=peak_spline_smoothing_factor)
    scale = float(np.nanmax(smoothed) - np.nanmin(smoothed))
    prominence = max(1e-6, peak_prominence_fraction * scale)
    peaks, properties = find_peaks(smoothed, distance=peak_distance_px, prominence=prominence)
    peaks = peaks[radius[peaks] >= min_radius_px]
    if peaks.size < min_rings:
        raise ValueError(f"Only {peaks.size} usable bright rings detected")

    best_candidate: Optional[Dict[str, MetricValue | np.ndarray]] = None
    quality_only_candidate: Optional[Dict[str, MetricValue | np.ndarray]] = None
    candidate_windows = contiguous_peak_windows(peaks, min_rings=min_rings)
    for candidate_peaks in candidate_windows:
        ring_radii_candidate = radius[candidate_peaks]
        ring_diameters_candidate = 2.0 * ring_radii_candidate
        diameter_squared_candidate = ring_diameters_candidate**2
        try:
            ring_fit = fit_ring_diameter_sequence(diameter_squared_candidate)
        except ValueError:
            continue
        slope_candidate = float(ring_fit["slope"])
        if not np.isfinite(slope_candidate) or slope_candidate <= 0:
            continue
        intercept_candidate = float(ring_fit["intercept"])
        axis_candidate = -intercept_candidate / slope_candidate
        ring_epsilon_candidate = (1.0 - axis_candidate) % 1.0
        epsilon_candidate = float(ring_epsilon_candidate)
        score, continuity_error, unwrapped_candidate = score_ring_candidate(
            epsilon=epsilon_candidate,
            ring_fit=ring_fit,
            ring_count=int(candidate_peaks.size),
            expected_epsilon_unwrapped=expected_epsilon_unwrapped,
            phase_tracking_weight=phase_tracking_weight,
        )
        quality_score, _, _ = score_ring_candidate(
            epsilon=epsilon_candidate,
            ring_fit=ring_fit,
            ring_count=int(candidate_peaks.size),
            expected_epsilon_unwrapped=None,
            phase_tracking_weight=0.0,
        )
        candidate: Dict[str, MetricValue | np.ndarray] = {
            "score": float(score),
            "quality_score": float(quality_score),
            "continuity_error": float(continuity_error),
            "tracked_unwrapped": float(unwrapped_candidate),
            "epsilon": epsilon_candidate,
            "ring_epsilon": float(ring_epsilon_candidate),
            "ring_fit": ring_fit,
            "peaks": candidate_peaks,
            "ring_radii": ring_radii_candidate,
            "ring_diameters": ring_diameters_candidate,
            "diameter_squared": diameter_squared_candidate,
        }
        if best_candidate is None or float(candidate["score"]) > float(best_candidate["score"]):
            best_candidate = candidate
        if quality_only_candidate is None or float(candidate["quality_score"]) > float(quality_only_candidate["quality_score"]):
            quality_only_candidate = candidate

    if best_candidate is None:
        raise ValueError("No valid ring-diameter candidate could be fitted")

    ring_fit = best_candidate["ring_fit"]
    ring_radii = np.asarray(best_candidate["ring_radii"], dtype=np.float64)
    ring_diameters = np.asarray(best_candidate["ring_diameters"], dtype=np.float64)
    diameter_squared = np.asarray(best_candidate["diameter_squared"], dtype=np.float64)
    ring_numbers = np.asarray(ring_fit["orders"], dtype=np.float64)
    slope = float(ring_fit["slope"])
    intercept = float(ring_fit["intercept"])
    fitted = np.asarray(ring_fit["fitted"], dtype=np.float64)
    r_squared = float(ring_fit["r_squared"])
    axis_intersection = -float(intercept) / float(slope)
    ring_epsilon = float(best_candidate["ring_epsilon"])
    refined = refine_epsilon_with_radial_sinusoid(
        radius=radius,
        smoothed=smoothed,
        ring_fit_slope=slope,
        ring_epsilon=ring_epsilon,
        min_radius_px=min_radius_px,
    )
    epsilon = float(refined["epsilon"])
    quality_epsilon = float(quality_only_candidate["epsilon"]) if quality_only_candidate is not None else epsilon
    if expected_epsilon_unwrapped is None or not np.isfinite(expected_epsilon_unwrapped):
        tracked_unwrapped = epsilon
        continuity_error = np.nan
    else:
        tracked_unwrapped = unwrap_near(epsilon, float(expected_epsilon_unwrapped))
        continuity_error = abs(tracked_unwrapped - float(expected_epsilon_unwrapped))
    if not np.isfinite(tracked_unwrapped):
        tracked_unwrapped = epsilon
    return {
        "epsilon_wrapped": float(epsilon),
        "epsilon_ring_wrapped": float(ring_epsilon),
        "epsilon_quality_wrapped": float(quality_epsilon),
        "epsilon_tracked_unwrapped": float(tracked_unwrapped),
        "phase_tracking_continuity_error": float(continuity_error),
        "phase_tracking_candidate_count": float(len(candidate_windows)),
        "epsilon_sinusoidal_r_squared": float(refined["r_squared"]),
        "epsilon_sinusoidal_contrast": float(refined["contrast"]),
        "epsilon_sinusoidal_phase_shift_rad": float(refined["phase_shift_rad"]),
        "epsilon_sinusoidal_refinement_used": float(refined["used_refinement"]),
        "ring_count": float(ring_diameters.size),
        "ring_fit_slope": float(slope),
        "ring_fit_intercept": float(intercept),
        "ring_fit_axis_intersection": float(axis_intersection),
        "ring_fit_r_squared": float(r_squared),
        "ring_fit_missed_order_count": float(ring_fit["missed_count"]),
        "ring_fit_max_order_increment": float(ring_fit["max_increment"]),
        "ring_fit_residual_order_rmse": float(ring_fit["residual_order_rmse"]),
        "ring_fit_candidate_spacing_px2": float(ring_fit["candidate_spacing"]),
        "radius": radius,
        "profile": profile,
        "background": background,
        "fringe": fringe,
        "gaussian_smoothed": gaussian_smoothed,
        "smoothed": smoothed,
        "peak_radii": ring_radii,
        "diameter_squared": diameter_squared,
        "ring_numbers": ring_numbers,
        "fitted_diameter_squared": fitted,
    }


def fill_missing(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("Not enough valid phase values to unwrap the sequence")
    if valid.all():
        return values.copy(), valid
    indices = np.arange(values.size, dtype=np.float64)
    filled = values.copy()
    filled[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
    return filled, valid


def unwrap_epsilon_by_sweep(epsilon_wrapped: np.ndarray, sweep_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    epsilon_wrapped = np.asarray(epsilon_wrapped, dtype=np.float64)
    sweep_index = np.asarray(sweep_index, dtype=int)
    epsilon_unwrapped = np.full(epsilon_wrapped.size, np.nan, dtype=np.float64)
    valid_all = np.isfinite(epsilon_wrapped)
    for sweep in np.unique(sweep_index):
        mask = sweep_index == sweep
        values = epsilon_wrapped[mask]
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        filled, _ = fill_missing(values)
        epsilon_unwrapped[mask] = np.unwrap(2.0 * np.pi * filled) / (2.0 * np.pi)
    if np.isfinite(epsilon_unwrapped).sum() < 2:
        filled, valid_all = fill_missing(epsilon_wrapped)
        epsilon_unwrapped = np.unwrap(2.0 * np.pi * filled) / (2.0 * np.pi)
    return epsilon_unwrapped, valid_all


def unwrap_epsilon_globally(epsilon_wrapped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    epsilon_wrapped = np.asarray(epsilon_wrapped, dtype=np.float64)
    filled, valid = fill_missing(epsilon_wrapped)
    return np.unwrap(2.0 * np.pi * filled) / (2.0 * np.pi), valid


def fill_unwrapped_by_sweep(values: np.ndarray, sweep_index: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sweep_index = np.asarray(sweep_index, dtype=int)
    filled = values.copy()
    for sweep in np.unique(sweep_index):
        mask = sweep_index == sweep
        local = filled[mask]
        valid = np.isfinite(local)
        if valid.sum() == 0:
            continue
        if valid.sum() == 1:
            local[~valid] = float(local[valid][0])
        elif not valid.all():
            x = np.arange(local.size, dtype=np.float64)
            local[~valid] = np.interp(x[~valid], x[valid], local[valid])
        filled[mask] = local
    return filled


def robust_spline_smooth(values: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < max(8, degrees_of_freedom):
        return gaussian_filter1d(np.where(valid, values, np.nanmedian(values[valid])), sigma=2.0, mode="nearest")
    spline_smoothing = max(1e-9, 0.01 * valid.sum() * float(np.nanvar(values[valid])))
    spline = UnivariateSpline(x[valid], values[valid], s=spline_smoothing, k=3)
    return spline(x)


def profile_metrics(values: np.ndarray, prefix: str) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
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


def banach_profile_descriptors(values: np.ndarray, spacing: float = 1.0) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("Banach profile descriptors require at least two finite values")
    centred = finite - float(np.mean(finite))
    gradient = np.gradient(centred, spacing)
    total_variation = float(np.sum(np.abs(np.diff(centred))))
    l2 = float(np.sqrt(np.mean(centred**2)))
    gradient_seminorm = float(np.sqrt(np.mean(gradient**2)))
    return {
        "banach_l1_mean_nm": float(np.mean(np.abs(centred))),
        "banach_l2_rms_nm": l2,
        "banach_linf_nm": float(np.max(np.abs(centred))),
        "banach_bv_total_variation_nm": total_variation,
        "banach_bv_total_variation_density_nm_per_sample": total_variation / max(1.0, float(centred.size - 1)),
        "banach_w12_gradient_seminorm_nm_per_sample": gradient_seminorm,
        "banach_w12_norm_nm": float(np.sqrt(l2**2 + gradient_seminorm**2)),
    }


def build_sequence_geometry(frame_indices: np.ndarray, total_available_frames: int, sweep_count: int) -> Dict[str, np.ndarray | float]:
    frames_per_sweep = int(round(total_available_frames / max(1, sweep_count)))
    frames_per_sweep = max(1, frames_per_sweep)
    angle_step_deg = 400.0 / frames_per_sweep
    sweep_index = frame_indices // frames_per_sweep
    sweep_angle_deg = (frame_indices % frames_per_sweep) * angle_step_deg
    continuous_angle_deg = frame_indices * angle_step_deg
    angle_deg = continuous_angle_deg % 360.0
    useful_mask = sweep_angle_deg < 360.0
    return {
        "frames_per_sweep": float(frames_per_sweep),
        "angle_step_deg": float(angle_step_deg),
        "sweep_index": sweep_index,
        "angle_deg": angle_deg,
        "sweep_angle_deg": sweep_angle_deg,
        "continuous_angle_deg": continuous_angle_deg,
        "useful_mask": useful_mask,
    }


def fit_ceramic_profile(
    height_nm: np.ndarray,
    angle_deg: np.ndarray,
    sweep_index: np.ndarray,
    useful_mask: np.ndarray,
    spline_degrees_of_freedom: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    valid = useful_mask & np.isfinite(height_nm)
    theta = np.deg2rad(angle_deg[valid])
    values = height_nm[valid]
    sweeps = sweep_index[valid].astype(int)
    unique_sweeps = np.array(sorted(set(sweeps.tolist())), dtype=int)
    sweep_terms = np.column_stack([sweeps == sweep for sweep in unique_sweeps]).astype(np.float64)
    design = np.column_stack([sweep_terms, np.cos(theta), np.sin(theta)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    sweep_offsets = coefficients[: unique_sweeps.size]
    cos_coefficient = float(coefficients[unique_sweeps.size])
    sin_coefficient = float(coefficients[unique_sweeps.size + 1])
    sweep_lookup = {int(sweep): float(offset) for sweep, offset in zip(unique_sweeps, sweep_offsets)}
    eccentricity_fit = np.array([sweep_lookup[int(sweep)] for sweep in sweeps]) + cos_coefficient * np.cos(theta) + sin_coefficient * np.sin(theta)
    residual = values - eccentricity_fit
    residual -= float(np.nanmean(residual))
    order = np.arange(residual.size, dtype=np.float64)
    linear_coefficients = np.polyfit(order, residual, deg=1)
    linear_trend = np.polyval(linear_coefficients, order)
    detrended = residual - linear_trend
    smoothed = robust_spline_smooth(detrended, degrees_of_freedom=spline_degrees_of_freedom)
    smoothed -= float(np.nanmean(smoothed))
    table = pd.DataFrame(
        {
            "frame_order": order,
            "angle_deg": angle_deg[valid],
            "sweep_index": sweeps,
            "height_nm": values,
            "eccentricity_fit_nm": eccentricity_fit,
            "roundness_residual_nm": residual,
            "detrended_residual_nm": detrended,
            "smoothed_profile_nm": smoothed,
        }
    )
    metrics = {
        "eccentricity_cos_nm": cos_coefficient,
        "eccentricity_sin_nm": sin_coefficient,
        "eccentricity_amplitude_nm": float(np.sqrt(cos_coefficient**2 + sin_coefficient**2)),
        "linear_trend_slope_nm_per_sample": float(linear_coefficients[0]),
        "linear_trend_intercept_nm": float(linear_coefficients[1]),
        "roundness_RONt_nm": float(np.nanmax(detrended) - np.nanmin(detrended)),
        "roundness_RONq_nm": float(np.sqrt(np.nanmean(detrended**2))),
    }
    metrics.update(profile_metrics(smoothed, prefix="profile_smoothed"))
    metrics.update(banach_profile_descriptors(smoothed))
    return table, metrics


def sweep_descriptor_rows(sequence_table: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    if "sweep_index" not in sequence_table.columns:
        return pd.DataFrame(rows)
    for sweep_index, group in sequence_table.groupby("sweep_index"):
        if len(group) < 8:
            continue
        row: Dict[str, float] = {"sweep_index": float(sweep_index), "sweep_sample_count": float(len(group))}
        smoothed = group["smoothed_profile_nm"].to_numpy(dtype=np.float64)
        row.update(profile_metrics(smoothed, prefix="profile_smoothed"))
        row.update(banach_profile_descriptors(smoothed))
        if model == "ceramic-roundness" and "detrended_residual_nm" in group:
            residual = group["detrended_residual_nm"].to_numpy(dtype=np.float64)
            residual = residual[np.isfinite(residual)]
            if residual.size:
                row["roundness_RONt_nm"] = float(np.nanmax(residual) - np.nanmin(residual))
                row["roundness_RONq_nm"] = float(np.sqrt(np.nanmean(residual**2)))
        if model == "step" and "step_reference_nm" in group:
            reference = group["step_reference_nm"].to_numpy(dtype=np.float64)
            reference = reference[np.isfinite(reference)]
            if reference.size:
                row["step_depth_nm"] = float(np.nanmax(reference) - np.nanmin(reference))
        rows.append(row)
    return pd.DataFrame(rows)


def uncertainty_from_sweeps(sweep_metrics: pd.DataFrame) -> Dict[str, float]:
    uncertainty: Dict[str, float] = {}
    if sweep_metrics.empty:
        return uncertainty
    for column in sweep_metrics.columns:
        if column in {"sweep_index", "sweep_sample_count"}:
            continue
        values = sweep_metrics[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean_value = float(np.mean(values))
        uncertainty[f"{column}_sweep_mean"] = mean_value
        uncertainty[f"{column}_sweep_n"] = float(values.size)
        if values.size >= 2:
            sd_value = float(np.std(values, ddof=1))
            standard_uncertainty = sd_value / float(np.sqrt(values.size))
            expanded_uncertainty = float(student_t.ppf(0.975, values.size - 1) * standard_uncertainty)
        else:
            sd_value = 0.0
            standard_uncertainty = 0.0
            expanded_uncertainty = 0.0
        uncertainty[f"{column}_sweep_sd"] = sd_value
        uncertainty[f"{column}_standard_uncertainty"] = standard_uncertainty
        uncertainty[f"{column}_expanded_uncertainty_95"] = expanded_uncertainty
        uncertainty[f"{column}_relative_standard_uncertainty"] = float(standard_uncertainty / abs(mean_value)) if mean_value else np.nan
    return uncertainty


def fit_step_profile(
    height_nm: np.ndarray,
    spline_degrees_of_freedom: int,
    sweep_index: Optional[np.ndarray] = None,
    angle_deg: Optional[np.ndarray] = None,
    edge_margin_fraction: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    valid = np.isfinite(height_nm)
    values = height_nm[valid]
    sweeps = sweep_index[valid].astype(int) if sweep_index is not None else np.zeros(values.size, dtype=int)
    angles = angle_deg[valid] if angle_deg is not None else np.full(values.size, np.nan, dtype=np.float64)
    order = np.arange(values.size, dtype=np.float64)

    # Step 1: remove first-harmonic eccentricity (steel artefact is also mounted on a rotation stage)
    unique_sweeps = np.array(sorted(set(sweeps.tolist())), dtype=int)
    theta = np.deg2rad(angles)
    sweep_terms = np.column_stack([sweeps == sweep for sweep in unique_sweeps]).astype(np.float64)
    design = np.column_stack([sweep_terms, np.cos(theta), np.sin(theta)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    sweep_offsets = coefficients[: unique_sweeps.size]
    cos_coefficient = float(coefficients[unique_sweeps.size])
    sin_coefficient = float(coefficients[unique_sweeps.size + 1])
    sweep_lookup = {int(sweep): float(offset) for sweep, offset in zip(unique_sweeps, sweep_offsets)}
    eccentricity_fit = np.array([sweep_lookup[int(sweep)] for sweep in sweeps]) + cos_coefficient * np.cos(theta) + sin_coefficient * np.sin(theta)
    eccentricity_amplitude = float(np.sqrt(cos_coefficient**2 + sin_coefficient**2))
    # R-squared of eccentricity fit
    ss_res = float(np.sum((values - eccentricity_fit)**2))
    ss_tot = float(np.sum((values - np.mean(values))**2))
    eccentricity_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    eccentricity_corrected = values - eccentricity_fit

    # Step 2: remove residual linear drift
    linear_coefficients = np.polyfit(order, eccentricity_corrected, deg=1)
    levelled = eccentricity_corrected - np.polyval(linear_coefficients, order)

    # Step 3: find step edge from smoothed gradient
    smooth_sigma = max(2.0, values.size / 2000.0)
    smooth_for_edge = gaussian_filter1d(levelled, sigma=smooth_sigma, mode="nearest")
    gradient = np.abs(np.gradient(smooth_for_edge))
    margin = int(max(5, edge_margin_fraction * values.size))
    if values.size <= 2 * margin + 2:
        edge_index = int(values.size // 2)
    else:
        search = gradient[margin : values.size - margin]
        edge_index = int(np.nanargmax(search) + margin)

    # Step 4: two-plateau model
    left_mask = order < edge_index
    right_mask = order >= edge_index
    left_level = float(np.nanmedian(levelled[left_mask])) if np.any(left_mask) else 0.0
    right_level = float(np.nanmedian(levelled[right_mask])) if np.any(right_mask) else 0.0
    model = np.where(left_mask, left_level, right_level)
    residual = levelled - model
    residual -= float(np.nanmean(residual))
    smoothed = robust_spline_smooth(residual, degrees_of_freedom=spline_degrees_of_freedom)
    smoothed -= float(np.nanmean(smoothed))

    table = pd.DataFrame(
        {
            "frame_order": order,
            "sweep_index": sweeps,
            "angle_deg": angles,
            "height_nm": values,
            "eccentricity_fit_nm": eccentricity_fit,
            "linear_levelled_height_nm": levelled,
            "step_reference_nm": model,
            "step_residual_nm": residual,
            "smoothed_profile_nm": smoothed,
        }
    )
    metrics = {
        "eccentricity_cos_nm": cos_coefficient,
        "eccentricity_sin_nm": sin_coefficient,
        "eccentricity_amplitude_nm": eccentricity_amplitude,
        "eccentricity_r2_nm": eccentricity_r2,
        "linear_trend_slope_nm_per_sample": float(linear_coefficients[0]),
        "linear_trend_intercept_nm": float(linear_coefficients[1]),
        "step_edge_sample": float(edge_index),
        "step_left_level_nm": left_level,
        "step_right_level_nm": right_level,
        "step_height_nm": float(right_level - left_level),
        "step_depth_nm": float(abs(right_level - left_level)),
    }
    metrics.update(profile_metrics(smoothed, prefix="profile_smoothed"))
    metrics.update(banach_profile_descriptors(smoothed))
    return table, metrics


def choose_sample_model(label: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = label.lower()
    if "steel" in lowered or "step" in lowered:
        return "step"
    return "ceramic-roundness"


def write_radial_extraction_pdf(output_path: Path, label: str, image: np.ndarray, extraction: Dict[str, MetricValue]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"{label}: interferogram")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    radius = extraction["radius"]
    axes[1].plot(radius, extraction["profile"], color="#777777", linewidth=0.8, label="radial intensity")
    axes[1].plot(radius, extraction["background"], color="#D55E00", linewidth=0.9, label="background")
    axes[1].plot(radius, extraction["smoothed"], color="#0072B2", linewidth=0.9, label="fringe signal")
    axes[1].scatter(extraction["peak_radii"], np.interp(extraction["peak_radii"], radius, extraction["smoothed"]), s=28, color="#009E73", edgecolors="#004D40", linewidths=0.6, zorder=5, label="bright rings")
    axes[1].set_title("Radial fringe profile")
    axes[1].set_xlabel("radius, px")
    axes[1].set_ylabel("normalised intensity / signal")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].scatter(extraction["ring_numbers"], extraction["diameter_squared"], s=14, color="#444444", label="rings")
    axes[2].plot(extraction["ring_numbers"], extraction["fitted_diameter_squared"], color="#D55E00", linewidth=1.0, label="linear fit")
    axes[2].axvline(extraction["ring_fit_axis_intersection"], color="#0072B2", linestyle="--", linewidth=0.9, label="1 - epsilon")
    axes[2].set_title(f"epsilon = {float(extraction['epsilon_wrapped']):.3f}")
    axes[2].set_xlabel("ring number")
    axes[2].set_ylabel("diameter squared, px$^2$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=7)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_profile_pdf(output_path: Path, label: str, model: str, sequence_table: pd.DataFrame, metrics: Dict[str, MetricValue]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    if "angle_deg" in sequence_table.columns:
        x = sequence_table["angle_deg"].to_numpy()
        x_label = "angle, deg"
    else:
        x = sequence_table["frame_order"].to_numpy()
        x_label = "image order"
    plot_series(axes[0, 0], sequence_table["frame_order"].to_numpy(), sequence_table["height_nm"].to_numpy(), color="#444444", linewidth=0.65)
    axes[0, 0].set_title("Unwrapped displacement profile")
    axes[0, 0].set_xlabel("image order")
    axes[0, 0].set_ylabel("nm")
    axes[0, 0].grid(True, alpha=0.25)

    if model == "ceramic-roundness":
        plot_sweep_angle_series(axes[0, 1], sequence_table, "height_nm", color="#999999", linewidth=0.55, label="height")
        plot_sweep_angle_series(axes[0, 1], sequence_table, "eccentricity_fit_nm", color="#D55E00", linewidth=0.9, label="eccentricity fit")
        axes[0, 1].set_title("First-harmonic eccentricity reduction")
        plot_sweep_angle_series(axes[1, 0], sequence_table, "detrended_residual_nm", color="#777777", linewidth=0.55, label="roundness residual")
        plot_sweep_angle_series(axes[1, 0], sequence_table, "smoothed_profile_nm", color="#0072B2", linewidth=0.9, label="smoothed profile")
        axes[1, 0].set_title("Residual profile for roughness and Banach descriptors")
        bars = [
            metrics["profile_smoothed_Ra_nm"],
            metrics["profile_smoothed_Rq_nm"],
            metrics["profile_smoothed_Rz_nm"],
            metrics["roundness_RONt_nm"],
            metrics["roundness_RONq_nm"],
            metrics["banach_l2_rms_nm"],
            metrics["banach_linf_nm"],
        ]
        labels = ["Ra", "Rq", "Rz", "RONt", "RONq", "B L2", "B Linf"]
    else:
        plot_series(axes[0, 1], x, sequence_table["linear_levelled_height_nm"].to_numpy(), color="#999999", linewidth=0.55, label="levelled")
        plot_series(axes[0, 1], x, sequence_table["step_reference_nm"].to_numpy(), color="#D55E00", linewidth=0.9, label="two-plateau model")
        axes[0, 1].set_title("Step reference reduction")
        plot_series(axes[1, 0], x, sequence_table["step_residual_nm"].to_numpy(), color="#777777", linewidth=0.55, label="step residual")
        plot_series(axes[1, 0], x, sequence_table["smoothed_profile_nm"].to_numpy(), color="#0072B2", linewidth=0.9, label="smoothed profile")
        axes[1, 0].set_title("Residual profile for roughness and Banach descriptors")
        bars = [
            metrics["profile_smoothed_Ra_nm"],
            metrics["profile_smoothed_Rq_nm"],
            metrics["profile_smoothed_Rz_nm"],
            metrics["step_depth_nm"],
            metrics["banach_l2_rms_nm"],
            metrics["banach_linf_nm"],
        ]
        labels = ["Ra", "Rq", "Rz", "step", "B L2", "B Linf"]
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("nm")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("nm")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].bar(labels, bars, color="#4C78A8")
    axes[1, 1].set_title("Classical and Banach descriptors")
    axes[1, 1].set_ylabel("nm")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    figure.suptitle(f"{label}: radial image processing sequence")
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_roundness_polar_pdf(output_path: Path, label: str, sequence_table: pd.DataFrame, metrics: Dict[str, MetricValue]) -> None:
    """Polar plot of the five rotation sweeps showing roundness deviation with RONt and RONq."""
    if "angle_deg" not in sequence_table.columns or "sweep_index" not in sequence_table.columns:
        return
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 7.0), subplot_kw={"projection": "polar"}, constrained_layout=True)
    ront = float(metrics.get("roundness_RONt_nm", 0))
    ronq = float(metrics.get("roundness_RONq_nm", 0))
    eccentricity_amp = float(metrics.get("eccentricity_amplitude_nm", 0))
    # Reference radius: place the profile far enough out that RONt is clearly visible
    ref_radius = max(ront * 2.5, 500.0)
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]

    # Plot the five sweeps
    for sweep_index, group in sequence_table.groupby("sweep_index", sort=True):
        group = group.sort_values("angle_deg")
        theta = np.deg2rad(group["angle_deg"].to_numpy(dtype=np.float64))
        residual = group["detrended_residual_nm"].to_numpy(dtype=np.float64)
        valid = np.isfinite(residual)
        r_values = ref_radius + residual[valid]
        color = colours[int(sweep_index) % len(colours)]
        axis.plot(theta[valid], r_values, linewidth=0.5, color=color, alpha=0.85,
                  label=f"sweep {int(sweep_index)}")

    # RONt envelope — peak-to-valley roundness band
    ront_upper = ref_radius + ront / 2.0
    ront_lower = ref_radius - ront / 2.0
    axis.fill_between(np.linspace(0, 2 * np.pi, 360), ront_lower, ront_upper,
                       color="#333333", alpha=0.12, linewidth=0, label=f"$RONt={ront:.0f}$ nm")
    axis.axhline(ront_upper, color="#333333", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.axhline(ront_lower, color="#333333", linestyle=":", linewidth=0.6, alpha=0.6)

    # RONq band — RMS roundness about the mean
    ronq_upper = ref_radius + ronq
    ronq_lower = ref_radius - ronq
    axis.fill_between(np.linspace(0, 2 * np.pi, 360), ronq_lower, ronq_upper,
                       color="#D55E00", alpha=0.08, linewidth=0, label=f"$RONq={ronq:.0f}$ nm")

    # Reference circle (ideal)
    axis.axhline(ref_radius, color="#D62728", linestyle="-", linewidth=0.9, alpha=0.7, label="mean circle")

    # Radial scale bar to indicate magnification
    bar_angle = np.deg2rad(30)
    bar_r = ref_radius + ront * 0.85
    bar_length = 200.0  # 200 nm scale bar
    axis.annotate("", xy=(bar_angle, bar_r), xytext=(bar_angle, bar_r + bar_length),
                  arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
    axis.text(bar_angle + 0.08, bar_r + bar_length / 2, f"{bar_length:.0f} nm",
              fontsize=7, color="#555555", va="center")

    axis.set_title(f"{label}: roundness deviation\n$RONt={ront:.0f}$ nm, $RONq={ronq:.0f}$ nm, eccentricity removed $E={eccentricity_amp/1000:.1f}$ µm",
                   pad=18, fontsize=11)
    axis.legend(loc="upper right", bbox_to_anchor=(1.32, 1.02), frameon=False, fontsize=7.5)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_phase_processing_steps_pdf(
    output_path: Path,
    label: str,
    model: str,
    phase_table: pd.DataFrame,
    sequence_table: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    frame_order = np.arange(len(phase_table), dtype=np.float64)
    plot_series(axes[0, 0], frame_order, phase_table["epsilon_wrapped"].to_numpy(), color="#0072B2", linewidth=0.65)
    axes[0, 0].set_title("Wrapped phase fraction from individual interferograms")
    axes[0, 0].set_xlabel("image order")
    axes[0, 0].set_ylabel("wrapped epsilon, fringe order")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(True, alpha=0.25)

    plot_series(axes[0, 1], frame_order, phase_table["epsilon_unwrapped"].to_numpy(), color="#009E73", linewidth=0.65)
    axes[0, 1].set_title("Unwrapped phase sequence")
    axes[0, 1].set_xlabel("image order")
    axes[0, 1].set_ylabel("unwrapped epsilon, fringe order")
    axes[0, 1].grid(True, alpha=0.25)

    plot_series(axes[1, 0], frame_order, phase_table["height_nm"].to_numpy(), color="#444444", linewidth=0.65)
    axes[1, 0].set_title("Topography/profile after half-wavelength conversion")
    axes[1, 0].set_xlabel("image order")
    axes[1, 0].set_ylabel("height/displacement, nm")
    axes[1, 0].grid(True, alpha=0.25)

    if model == "ceramic-roundness":
        plot_sweep_angle_series(axes[1, 1], sequence_table, "height_nm", color="#999999", linewidth=0.55, label="profile")
        plot_sweep_angle_series(axes[1, 1], sequence_table, "eccentricity_fit_nm", color="#D55E00", linewidth=0.9, label="first harmonic")
        plot_sweep_angle_series(axes[1, 1], sequence_table, "detrended_residual_nm", color="#0072B2", linewidth=0.65, label="residual")
        axes[1, 1].set_xlabel("angle, deg")
        axes[1, 1].set_title("Removal of object-motion eccentricity")
    else:
        x = sequence_table["frame_order"].to_numpy(dtype=np.float64)
        plot_series(axes[1, 1], x, sequence_table["linear_levelled_height_nm"].to_numpy(), color="#999999", linewidth=0.55, label="profile")
        plot_series(axes[1, 1], x, sequence_table["step_reference_nm"].to_numpy(), color="#D55E00", linewidth=0.9, label="step reference")
        plot_series(axes[1, 1], x, sequence_table["step_residual_nm"].to_numpy(), color="#0072B2", linewidth=0.65, label="residual")
        axes[1, 1].set_xlabel("image order")
        axes[1, 1].set_title("Removal of step/form component")
    axes[1, 1].set_ylabel("nm")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{label}: phase decoding and form-removal steps")
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def publication_metric_rows(metrics_rows: Sequence[Dict[str, MetricValue]]) -> pd.DataFrame:
    selected = [
        ("ISO/profile", "Ra", "profile_smoothed_Ra_nm"),
        ("ISO/profile", "Rq", "profile_smoothed_Rq_nm"),
        ("ISO/profile", "Rz", "profile_smoothed_Rz_nm"),
        ("ISO/roundness", "RONt", "roundness_RONt_nm"),
        ("ISO/roundness", "RONq", "roundness_RONq_nm"),
        ("ISO/step", "step depth", "step_depth_nm"),
        ("Banach", "L1 mean", "banach_l1_mean_nm"),
        ("Banach", "L2 RMS", "banach_l2_rms_nm"),
        ("Banach", "L infinity", "banach_linf_nm"),
        ("Banach", "BV total variation", "banach_bv_total_variation_nm"),
        ("Banach", "BV density", "banach_bv_total_variation_density_nm_per_sample"),
        ("Banach", "W12 gradient", "banach_w12_gradient_seminorm_nm_per_sample"),
    ]
    rows: List[Dict[str, MetricValue]] = []
    for metrics in metrics_rows:
        sample = str(metrics["sample"])
        for family, descriptor, key in selected:
            if key not in metrics or pd.isna(metrics[key]):
                continue
            rows.append(
                {
                    "sample": sample,
                    "family": family,
                    "descriptor": descriptor,
                    "value": float(metrics[key]),
                    "unit": "nm" if key != "banach_bv_total_variation_density_nm_per_sample" and key != "banach_w12_gradient_seminorm_nm_per_sample" else "nm/sample",
                    "sweep_mean": float(metrics.get(f"{key}_sweep_mean", np.nan)),
                    "standard_uncertainty": float(metrics.get(f"{key}_standard_uncertainty", np.nan)),
                    "expanded_uncertainty_95": float(metrics.get(f"{key}_expanded_uncertainty_95", np.nan)),
                    "relative_standard_uncertainty": float(metrics.get(f"{key}_relative_standard_uncertainty", np.nan)),
                    "repeat_count": float(metrics.get(f"{key}_sweep_n", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def latex_descriptor_label(descriptor: str) -> str:
    labels = {
        "Ra": "$R_a$",
        "Rq": "$R_q$",
        "Rz": "$R_z$",
        "RONt": "$RONt$",
        "RONq": "$RONq$",
        "step depth": "step depth",
        "L1 mean": "$L^1$ mean",
        "L2 RMS": "$L^2$ RMS",
        "L infinity": "$L^\\infty$",
        "BV total variation": "BV total variation",
        "BV density": "BV density",
        "W12 gradient": "$W^{1,2}$ gradient",
    }
    return labels.get(descriptor, descriptor.replace("_", "\\_"))


def compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    absolute = abs(value)
    if absolute >= 1000.0:
        return f"{value:.0f}"
    if absolute >= 100.0:
        return f"{value:.1f}"
    if absolute >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def latex_value_with_uncertainty(row: pd.Series) -> str:
    value = float(row.get("value", np.nan))
    expanded = float(row.get("expanded_uncertainty_95", np.nan))
    unit = str(row.get("unit", ""))
    if not np.isfinite(value):
        return "--"
    value_text = compact_number(value)
    if np.isfinite(expanded) and expanded > 0.0:
        text = f"${value_text} \\pm {compact_number(expanded)}$"
    else:
        text = value_text
    return f"{text} {unit}".strip()


def write_latex_metric_rows(output_path: Path, publication_table: pd.DataFrame) -> None:
    if publication_table.empty:
        output_path.write_text("", encoding="utf-8")
        return
    samples = [sample for sample in ["ceramic", "steel"] if sample in set(publication_table["sample"].astype(str))]
    for sample in publication_table["sample"].astype(str):
        if sample not in samples:
            samples.append(sample)
    ordered_descriptors = publication_table[["family", "descriptor"]].drop_duplicates().itertuples(index=False, name=None)
    lines: List[str] = []
    for family, descriptor in ordered_descriptors:
        subset = publication_table[(publication_table["family"] == family) & (publication_table["descriptor"] == descriptor)]
        label = latex_descriptor_label(str(descriptor))
        values = []
        for sample in samples:
            sample_rows = subset[subset["sample"].astype(str) == sample]
            values.append(latex_value_with_uncertainty(sample_rows.iloc[0]) if not sample_rows.empty else "--")
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_uncertainty_comparison_pdf(output_path: Path, publication_table: pd.DataFrame) -> None:
    if publication_table.empty:
        return
    figure, axis = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    table = publication_table.copy()
    table["label"] = table["sample"] + " " + table["descriptor"]
    x_positions = np.arange(len(table), dtype=np.float64)
    values = table["value"].to_numpy(dtype=np.float64)
    errors = table["expanded_uncertainty_95"].to_numpy(dtype=np.float64)
    errors = np.where(np.isfinite(errors), errors, 0.0)
    colours = table["family"].map(lambda value: "#4C78A8" if str(value).startswith("ISO") else "#F58518").to_list()
    axis.bar(x_positions, values, yerr=errors, capsize=2.5, color=colours)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(table["label"], rotation=35, ha="right")
    axis.set_ylabel("value with 95% expanded uncertainty, nm or nm/sample")
    axis.set_title("ISO/profile metrics versus Banach descriptors with sweep-repeatability uncertainty")
    axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_comparison_pdf(output_path: Path, comparison: pd.DataFrame) -> None:
    metric_names = [
        "profile_smoothed_Ra_nm",
        "profile_smoothed_Rq_nm",
        "profile_smoothed_Rz_nm",
        "roundness_RONt_nm",
        "roundness_RONq_nm",
        "step_depth_nm",
        "banach_l1_mean_nm",
        "banach_l2_rms_nm",
        "banach_linf_nm",
    ]
    available = [metric for metric in metric_names if metric in comparison.columns and comparison[metric].notna().any()]
    figure, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    x_positions = np.arange(len(available))
    width = 0.8 / max(1, len(comparison))
    for index, (_, row) in enumerate(comparison.iterrows()):
        offset = (index - 0.5 * (len(comparison) - 1)) * width
        values = [float(row[metric]) if pd.notna(row[metric]) else np.nan for metric in available]
        axis.bar(x_positions + offset, values, width, label=row["sample"])
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [
            metric.replace("profile_smoothed_", "")
            .replace("roundness_", "")
            .replace("banach_", "B ")
            .replace("step_depth", "step")
            .replace("_nm", "")
            for metric in available
        ],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("nm")
    axis.set_title("Classical profile/roundness and Banach descriptors from radial EFM profiles")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def process_folder(
    folder: Path,
    label: str,
    output_root: Path,
    wavelength_nm: float,
    frame_limit: Optional[int],
    frame_selection: str,
    downsample: int,
    sweep_count: int,
    per_sweep_frame_limit: Optional[int],
    sample_model: str,
    max_radius_px: int,
    min_radius_px: int,
    min_rings: int,
    peak_distance_px: int,
    background_sigma_px: float,
    smooth_sigma_px: float,
    peak_spline_smoothing_factor: float,
    peak_prominence_fraction: float,
    spline_degrees_of_freedom: int,
    progress_every: int,
) -> Dict[str, MetricValue]:
    all_paths = discover_images(folder)
    selected_paths = select_frame_paths(
        all_paths,
        frame_limit=frame_limit,
        selection=frame_selection,
        sweep_count=sweep_count,
        per_sweep_frame_limit=per_sweep_frame_limit,
    )
    if len(selected_paths) < 8:
        raise ValueError(f"At least eight images are required for sequence unwrapping, got {len(selected_paths)}")
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = np.array([numeric_frame_index(path) for path in selected_paths], dtype=int)
    geometry = build_sequence_geometry(frame_indices, total_available_frames=len(all_paths), sweep_count=sweep_count)
    sweep_index_values = np.asarray(geometry["sweep_index"], dtype=int)
    epsilon_wrapped = np.full(len(selected_paths), np.nan, dtype=np.float64)
    epsilon_ring_wrapped = np.full(len(selected_paths), np.nan, dtype=np.float64)
    epsilon_quality_wrapped = np.full(len(selected_paths), np.nan, dtype=np.float64)
    epsilon_tracked_unwrapped = np.full(len(selected_paths), np.nan, dtype=np.float64)
    phase_tracking_continuity_error = np.full(len(selected_paths), np.nan, dtype=np.float64)
    ring_counts = np.full(len(selected_paths), np.nan, dtype=np.float64)
    fit_r_squared = np.full(len(selected_paths), np.nan, dtype=np.float64)
    ring_missed_order_counts = np.full(len(selected_paths), np.nan, dtype=np.float64)
    ring_max_order_increments = np.full(len(selected_paths), np.nan, dtype=np.float64)
    ring_fit_residual_order_rmse = np.full(len(selected_paths), np.nan, dtype=np.float64)
    sinusoidal_fit_r_squared = np.full(len(selected_paths), np.nan, dtype=np.float64)
    sinusoidal_refinement_used = np.full(len(selected_paths), np.nan, dtype=np.float64)
    first_extraction = None
    first_image = None
    failures = 0
    last_unwrapped_by_sweep: Dict[int, float] = {}
    deltas_by_sweep: Dict[int, List[float]] = {}
    for index, path in enumerate(selected_paths):
        if progress_every > 0 and index > 0 and index % progress_every == 0:
            print(f"{label}: processed {index}/{len(selected_paths)} images", flush=True)
        image = load_grayscale(path, downsample=downsample)
        current_sweep = int(sweep_index_values[index])
        last_unwrapped = last_unwrapped_by_sweep.get(current_sweep)
        recent_deltas = deltas_by_sweep.get(current_sweep, [])
        if last_unwrapped is None:
            expected_epsilon_unwrapped = None
        elif recent_deltas:
            expected_epsilon_unwrapped = last_unwrapped + float(np.nanmedian(recent_deltas[-5:]))
        else:
            expected_epsilon_unwrapped = last_unwrapped
        try:
            extraction = fit_excess_fraction_from_rings(
                image,
                max_radius_px=max_radius_px,
                min_radius_px=min_radius_px,
                min_rings=min_rings,
                peak_distance_px=peak_distance_px,
                background_sigma_px=background_sigma_px,
                smooth_sigma_px=smooth_sigma_px,
                peak_spline_smoothing_factor=peak_spline_smoothing_factor,
                peak_prominence_fraction=peak_prominence_fraction,
                expected_epsilon_unwrapped=expected_epsilon_unwrapped,
            )
        except ValueError:
            failures += 1
            continue
        epsilon_wrapped[index] = float(extraction["epsilon_wrapped"])
        epsilon_ring_wrapped[index] = float(extraction["epsilon_ring_wrapped"])
        epsilon_quality_wrapped[index] = float(extraction["epsilon_quality_wrapped"])
        epsilon_tracked_unwrapped[index] = float(extraction["epsilon_tracked_unwrapped"])
        phase_tracking_continuity_error[index] = float(extraction["phase_tracking_continuity_error"])
        ring_counts[index] = float(extraction["ring_count"])
        fit_r_squared[index] = float(extraction["ring_fit_r_squared"])
        ring_missed_order_counts[index] = float(extraction["ring_fit_missed_order_count"])
        ring_max_order_increments[index] = float(extraction["ring_fit_max_order_increment"])
        ring_fit_residual_order_rmse[index] = float(extraction["ring_fit_residual_order_rmse"])
        sinusoidal_fit_r_squared[index] = float(extraction["epsilon_sinusoidal_r_squared"])
        sinusoidal_refinement_used[index] = float(extraction["epsilon_sinusoidal_refinement_used"])
        if np.isfinite(epsilon_tracked_unwrapped[index]):
            if current_sweep in last_unwrapped_by_sweep:
                delta = float(epsilon_tracked_unwrapped[index] - last_unwrapped_by_sweep[current_sweep])
                deltas_by_sweep.setdefault(current_sweep, []).append(delta)
            last_unwrapped_by_sweep[current_sweep] = float(epsilon_tracked_unwrapped[index])
        if first_extraction is None:
            first_extraction = extraction
            first_image = image
    if progress_every > 0:
        print(f"{label}: processed {len(selected_paths)}/{len(selected_paths)} images", flush=True)

    epsilon_raw_unwrapped, valid_epsilon = unwrap_epsilon_by_sweep(epsilon_wrapped, sweep_index_values)
    is_contiguous_sequence = bool(len(frame_indices) > 1 and np.all(np.diff(frame_indices) == 1))
    if is_contiguous_sequence:
        epsilon_unwrapped, valid_epsilon = unwrap_epsilon_globally(epsilon_wrapped)
        phase_unwrapping_method = "global unwrap of the continuous chronological epsilon sequence"
    elif np.isfinite(epsilon_tracked_unwrapped).sum() >= 2:
        epsilon_unwrapped = fill_unwrapped_by_sweep(epsilon_tracked_unwrapped, sweep_index_values)
        phase_unwrapping_method = "tracked continuous epsilon sequence within each selected sweep"
    else:
        epsilon_unwrapped = epsilon_raw_unwrapped
        phase_unwrapping_method = "raw sweep-wise unwrap of the selected epsilon sequence"
    height_nm = epsilon_unwrapped * (wavelength_nm / 2.0)
    height_nm -= float(np.nanmedian(height_nm[valid_epsilon]))
    selected_model = choose_sample_model(label, requested=sample_model)
    if selected_model == "ceramic-roundness":
        sequence_table, metrics = fit_ceramic_profile(
            height_nm=height_nm,
            angle_deg=np.asarray(geometry["angle_deg"], dtype=np.float64),
            sweep_index=np.asarray(geometry["sweep_index"], dtype=int),
            useful_mask=np.asarray(geometry["useful_mask"], dtype=bool),
            spline_degrees_of_freedom=spline_degrees_of_freedom,
        )
    elif selected_model == "step":
        sequence_table, metrics = fit_step_profile(
            height_nm=height_nm,
            spline_degrees_of_freedom=spline_degrees_of_freedom,
            sweep_index=np.asarray(geometry["sweep_index"], dtype=int),
            angle_deg=np.asarray(geometry["angle_deg"], dtype=np.float64),
        )
    else:
        raise ValueError(f"Unsupported sample model: {selected_model}")
    sweep_metrics = sweep_descriptor_rows(sequence_table, model=selected_model)
    uncertainty_metrics = uncertainty_from_sweeps(sweep_metrics)
    metrics.update(uncertainty_metrics)

    metadata: Dict[str, MetricValue] = {
        "sample": label,
        "input_folder": str(folder),
        "total_available_frames": float(len(all_paths)),
        "processed_frames": float(len(selected_paths)),
        "valid_phase_frames": float(valid_epsilon.sum()),
        "failed_phase_frames": float(failures),
        "frame_selection": frame_selection,
        "per_sweep_frame_limit": float(per_sweep_frame_limit) if per_sweep_frame_limit else np.nan,
        "downsample": float(downsample),
        "wavelength_nm": float(wavelength_nm),
        "fringe_order_to_height_scale_nm": float(wavelength_nm / 2.0),
        "frames_per_sweep": float(geometry["frames_per_sweep"]),
        "angle_step_deg": float(geometry["angle_step_deg"]),
        "max_radius_px": float(max_radius_px),
        "min_radius_px": float(min_radius_px),
        "phase_retrieval_method": "per-image radial excess-fraction fringe analysis with spline-smoothed maxima detection",
        "phase_refinement_method": "continuity-aware peak-subset EFM with optional full radial-profile sinusoidal least-squares refinement",
        "phase_unwrapping_method": phase_unwrapping_method,
        "continuous_frame_sequence": float(is_contiguous_sequence),
        "phase_tracking_method": "continuity-aware EFM peak-subset selection within each sweep",
        "sample_model": selected_model,
        "peak_spline_smoothing_factor": float(peak_spline_smoothing_factor),
        "mean_detected_rings": float(np.nanmean(ring_counts)),
        "median_ring_fit_r_squared": float(np.nanmedian(fit_r_squared)),
        "mean_corrected_missing_ring_orders": float(np.nanmean(ring_missed_order_counts)),
        "fraction_images_with_corrected_missing_orders": float(np.nanmean(ring_missed_order_counts > 0)),
        "max_corrected_order_increment": float(np.nanmax(ring_max_order_increments)),
        "median_ring_fit_residual_order_rmse": float(np.nanmedian(ring_fit_residual_order_rmse)),
        "median_phase_tracking_continuity_error": float(np.nanmedian(phase_tracking_continuity_error)),
        "median_sinusoidal_fit_r_squared": float(np.nanmedian(sinusoidal_fit_r_squared)),
        "sinusoidal_refinement_used_fraction": float(np.nanmean(sinusoidal_refinement_used)),
    }
    full_metrics: Dict[str, MetricValue] = {**metadata, **metrics}

    np.save(output_dir / "epsilon_wrapped.npy", epsilon_wrapped.astype(np.float32))
    np.save(output_dir / "epsilon_ring_wrapped.npy", epsilon_ring_wrapped.astype(np.float32))
    np.save(output_dir / "epsilon_unwrapped.npy", epsilon_unwrapped.astype(np.float32))
    np.save(output_dir / "epsilon_raw_unwrapped.npy", epsilon_raw_unwrapped.astype(np.float32))
    np.save(output_dir / "height_profile_nm.npy", height_nm.astype(np.float32))
    phase_table = pd.DataFrame(
        {
            "frame_index": frame_indices,
            "angle_deg": np.asarray(geometry["angle_deg"], dtype=np.float64),
            "sweep_angle_deg": np.asarray(geometry["sweep_angle_deg"], dtype=np.float64),
            "continuous_angle_deg": np.asarray(geometry["continuous_angle_deg"], dtype=np.float64),
            "epsilon_wrapped": epsilon_wrapped,
            "epsilon_ring_wrapped": epsilon_ring_wrapped,
            "epsilon_quality_wrapped": epsilon_quality_wrapped,
            "epsilon_raw_unwrapped": epsilon_raw_unwrapped,
            "epsilon_unwrapped": epsilon_unwrapped,
            "epsilon_tracked_unwrapped": epsilon_tracked_unwrapped,
            "height_nm": height_nm,
            "phase_tracking_continuity_error": phase_tracking_continuity_error,
            "ring_count": ring_counts,
            "ring_fit_r_squared": fit_r_squared,
            "ring_fit_missed_order_count": ring_missed_order_counts,
            "ring_fit_max_order_increment": ring_max_order_increments,
            "ring_fit_residual_order_rmse": ring_fit_residual_order_rmse,
            "sinusoidal_fit_r_squared": sinusoidal_fit_r_squared,
            "sinusoidal_refinement_used": sinusoidal_refinement_used,
        }
    )
    phase_table.to_csv(output_dir / "phase_sequence.csv", index=False)
    sequence_table.to_csv(output_dir / "profile_analysis.csv", index=False)
    sweep_metrics.to_csv(output_dir / "sweep_metric_uncertainty.csv", index=False)
    pd.DataFrame([full_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(full_metrics, handle, indent=2)
    if first_extraction is not None and first_image is not None:
        write_radial_extraction_pdf(output_dir / "radial_phase_extraction.pdf", label=label, image=first_image, extraction=first_extraction)
    write_phase_processing_steps_pdf(output_dir / "phase_processing_steps.pdf", label=label, model=selected_model, phase_table=phase_table, sequence_table=sequence_table)
    write_profile_pdf(output_dir / "profile_reconstruction.pdf", label=label, model=selected_model, sequence_table=sequence_table, metrics=full_metrics)
    if selected_model == "ceramic-roundness" and "angle_deg" in sequence_table.columns:
        write_roundness_polar_pdf(output_dir / "roundness_polar.pdf", label=label, sequence_table=sequence_table, metrics=full_metrics)
    return full_metrics


def parse_dataset(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        return label.strip(), Path(path_text).expanduser()
    path = Path(value).expanduser()
    return path.name, path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Decode per-image radial fringe phase and reconstruct nm profiles.")
    parser.add_argument("--dataset", action="append", required=True, help="Dataset as label=folder or folder. Repeat for each sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/radial_profile_reconstruction_metric"))
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM)
    parser.add_argument("--frame-limit", type=int, default=4800, help="Images processed per folder; use 0 for all images.")
    parser.add_argument("--frame-selection", choices=["first", "stratified"], default="first")
    parser.add_argument("--per-sweep-frame-limit", type=int, default=0, help="If >0, process this many frames from each sweep instead of a single global frame limit.")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--sweep-count", type=int, default=5)
    parser.add_argument("--sample-model", choices=["auto", "ceramic-roundness", "step"], default="auto")
    parser.add_argument("--max-radius-px", type=int, default=240)
    parser.add_argument("--min-radius-px", type=int, default=5)
    parser.add_argument("--ceramic-min-radius-px", type=int, default=None, help="Optional minimum radial peak radius used only for ceramic-roundness datasets.")
    parser.add_argument("--steel-min-radius-px", type=int, default=None, help="Optional minimum radial peak radius used only for step/steel datasets.")
    parser.add_argument("--min-rings", type=int, default=3)
    parser.add_argument("--peak-distance-px", type=int, default=5)
    parser.add_argument("--background-sigma-px", type=float, default=20.0)
    parser.add_argument("--smooth-sigma-px", type=float, default=1.0)
    parser.add_argument("--peak-spline-smoothing-factor", type=float, default=0.03)
    parser.add_argument("--peak-prominence-fraction", type=float, default=0.08)
    parser.add_argument("--spline-degrees-of-freedom", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_limit = None if args.frame_limit <= 0 else int(args.frame_limit)
    rows = []
    for label, folder in [parse_dataset(item) for item in args.dataset]:
        inferred_model = choose_sample_model(label, requested=args.sample_model)
        min_radius_px = int(args.min_radius_px)
        if inferred_model == "ceramic-roundness" and args.ceramic_min_radius_px is not None:
            min_radius_px = int(args.ceramic_min_radius_px)
        if inferred_model == "step" and args.steel_min_radius_px is not None:
            min_radius_px = int(args.steel_min_radius_px)
        rows.append(
            process_folder(
                folder=folder,
                label=label,
                output_root=args.output_dir,
                wavelength_nm=float(args.wavelength_nm),
                frame_limit=frame_limit,
                frame_selection=args.frame_selection,
                downsample=max(1, int(args.downsample)),
                sweep_count=max(1, int(args.sweep_count)),
                per_sweep_frame_limit=None if args.per_sweep_frame_limit <= 0 else int(args.per_sweep_frame_limit),
                sample_model=args.sample_model,
                max_radius_px=max(10, int(args.max_radius_px)),
                min_radius_px=max(0, min_radius_px),
                min_rings=max(3, int(args.min_rings)),
                peak_distance_px=max(1, int(args.peak_distance_px)),
                background_sigma_px=max(1.0, float(args.background_sigma_px)),
                smooth_sigma_px=max(0.25, float(args.smooth_sigma_px)),
                peak_spline_smoothing_factor=max(0.0, float(args.peak_spline_smoothing_factor)),
                peak_prominence_fraction=max(0.0, float(args.peak_prominence_fraction)),
                spline_degrees_of_freedom=max(6, int(args.spline_degrees_of_freedom)),
                progress_every=max(0, int(args.progress_every)),
            )
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "sample_comparison_metrics.csv", index=False)
    publication_table = publication_metric_rows(rows)
    publication_table.to_csv(args.output_dir / "iso_banach_uncertainty_table.csv", index=False)
    write_latex_metric_rows(args.output_dir / "metric_reconstruction_table_rows.tex", publication_table)
    with (args.output_dir / "sample_comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_comparison_pdf(args.output_dir / "sample_comparison_metrics.pdf", comparison)
    write_uncertainty_comparison_pdf(args.output_dir / "iso_banach_uncertainty_table.pdf", publication_table)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": [row["sample"] for row in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())