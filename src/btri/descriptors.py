"""Classical and Banach-space descriptors for residual fields."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def _finite_values(field: np.ndarray) -> np.ndarray:
    array = np.asarray(field, dtype=np.float64)
    values = array[np.isfinite(array)]
    if values.size == 0:
        raise ValueError("Descriptor input contains no finite values")
    return values


def _filled_field(field: np.ndarray) -> np.ndarray:
    array = np.asarray(field, dtype=np.float64)
    values = _finite_values(array)
    fill_value = float(np.median(values))
    return np.where(np.isfinite(array), array, fill_value)


def _safe_skewness(values: np.ndarray) -> float:
    centred = values - float(np.mean(values))
    std_value = float(np.std(values))
    if std_value <= 0:
        return 0.0
    return float(np.mean((centred / std_value) ** 3))


def _safe_excess_kurtosis(values: np.ndarray) -> float:
    centred = values - float(np.mean(values))
    std_value = float(np.std(values))
    if std_value <= 0:
        return 0.0
    return float(np.mean((centred / std_value) ** 4) - 3.0)


def classical_metrics(field: np.ndarray, residual: np.ndarray) -> Dict[str, float]:
    """Compute conventional scalar descriptors for comparison."""

    field_values = _finite_values(field)
    residual_values = _finite_values(residual)
    return {
        "classical_field_mean": float(np.mean(field_values)),
        "classical_field_std": float(np.std(field_values)),
        "classical_field_min": float(np.min(field_values)),
        "classical_field_max": float(np.max(field_values)),
        "classical_residual_mean": float(np.mean(residual_values)),
        "classical_residual_sa": float(np.mean(np.abs(residual_values))),
        "classical_residual_sq": float(np.sqrt(np.mean(residual_values**2))),
        "classical_residual_sp": float(np.max(residual_values)),
        "classical_residual_sv": float(np.min(residual_values)),
        "classical_residual_sz": float(np.max(residual_values) - np.min(residual_values)),
        "classical_residual_skewness": _safe_skewness(residual_values),
        "classical_residual_excess_kurtosis": _safe_excess_kurtosis(residual_values),
    }


def banach_descriptors(
    residual: np.ndarray,
    gradient_spacing: Tuple[float, float] = (1.0, 1.0),
    scale_sigmas_px: Iterable[float] = (1.0, 2.0, 4.0, 8.0),
) -> Dict[str, float]:
    """Compute normed residual descriptors in discrete function spaces."""

    residual_values = _finite_values(residual)
    filled_residual = _filled_field(residual)
    row_spacing, column_spacing = gradient_spacing
    row_gradient, column_gradient = np.gradient(filled_residual, row_spacing, column_spacing)
    gradient_magnitude = np.sqrt(row_gradient**2 + column_gradient**2)
    pixel_area = float(row_spacing * column_spacing)

    descriptors = {
        "banach_l1_mean": float(np.mean(np.abs(residual_values))),
        "banach_l2_rms": float(np.sqrt(np.mean(residual_values**2))),
        "banach_linf": float(np.max(np.abs(residual_values))),
        "banach_bv_total_variation": float(np.sum(gradient_magnitude) * pixel_area),
        "banach_bv_total_variation_density": float(np.mean(gradient_magnitude)),
        "banach_w12_gradient_seminorm": float(np.sqrt(np.mean(gradient_magnitude**2))),
    }
    descriptors["banach_w12_norm"] = float(
        np.sqrt(descriptors["banach_l2_rms"] ** 2 + descriptors["banach_w12_gradient_seminorm"] ** 2)
    )

    previous_scale = filled_residual
    for sigma_value in sorted(float(sigma) for sigma in scale_sigmas_px if float(sigma) > 0):
        smoothed = gaussian_filter(filled_residual, sigma=sigma_value, mode="nearest")
        band = previous_scale - smoothed
        band_values = _finite_values(band)
        sigma_label = str(sigma_value).replace(".", "p")
        descriptors[f"banach_scale_l1_sigma_{sigma_label}"] = float(np.mean(np.abs(band_values)))
        descriptors[f"banach_scale_l2_sigma_{sigma_label}"] = float(np.sqrt(np.mean(band_values**2)))
        previous_scale = smoothed
    descriptors["banach_lowpass_l2_last_scale"] = float(np.sqrt(np.mean(previous_scale**2)))
    return descriptors
