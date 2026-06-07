"""Reference-model fitting for residual topography fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import DatasetConfig


@dataclass
class ReferenceResult:
    """Result of fitting a physical or surrogate reference model."""

    model_name: str
    fitted: np.ndarray
    residual: np.ndarray
    parameters: Dict[str, float]


def coordinate_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalised coordinate grids in the interval [-1, 1]."""

    row_count, column_count = shape
    column_axis = np.linspace(-1.0, 1.0, column_count)
    row_axis = np.linspace(-1.0, 1.0, row_count)
    x_grid, y_grid = np.meshgrid(column_axis, row_axis)
    return x_grid, y_grid


def _polynomial_terms(x_grid: np.ndarray, y_grid: np.ndarray, degree: int) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if degree not in (0, 1, 2):
        raise ValueError("Only polynomial degrees 0, 1, and 2 are supported")
    terms = [np.ones_like(x_grid)]
    names = ["intercept"]
    if degree >= 1:
        terms.extend([x_grid, y_grid])
        names.extend(["x", "y"])
    if degree >= 2:
        terms.extend([x_grid * x_grid, x_grid * y_grid, y_grid * y_grid])
        names.extend(["x2", "xy", "y2"])
    return np.stack([term.ravel() for term in terms], axis=1), tuple(names)


def _sample_mask(mask: Optional[np.ndarray], shape: Tuple[int, int], max_pixels: int) -> np.ndarray:
    if mask is None:
        candidate_mask = np.ones(shape, dtype=bool)
    else:
        candidate_mask = np.asarray(mask, dtype=bool).copy()
    finite_count = int(candidate_mask.sum())
    if finite_count <= max_pixels:
        return candidate_mask
    sampling_stride = int(np.ceil(np.sqrt(finite_count / max_pixels)))
    grid_mask = np.zeros(shape, dtype=bool)
    grid_mask[::sampling_stride, ::sampling_stride] = True
    return candidate_mask & grid_mask


def fit_polynomial_form(
    field: np.ndarray,
    degree: int,
    max_pixels: int,
    mask: Optional[np.ndarray] = None,
    model_name: str = "polynomial",
) -> ReferenceResult:
    """Fit a polynomial form model by deterministic least squares."""

    clean_field = np.asarray(field, dtype=np.float64)
    finite_mask = np.isfinite(clean_field)
    if mask is not None:
        finite_mask &= np.asarray(mask, dtype=bool)
    sampled_mask = _sample_mask(finite_mask, clean_field.shape, max_pixels)
    if int(sampled_mask.sum()) < 6 and degree == 2:
        raise ValueError("Not enough finite pixels to fit a quadratic reference model")

    x_grid, y_grid = coordinate_grid(clean_field.shape)
    design_matrix, coefficient_names = _polynomial_terms(x_grid, y_grid, degree)
    sampled_design = design_matrix[sampled_mask.ravel(), :]
    sampled_values = clean_field.ravel()[sampled_mask.ravel()]
    coefficients, residual_sum, matrix_rank, singular_values = np.linalg.lstsq(
        sampled_design, sampled_values, rcond=None
    )
    fitted = (design_matrix @ coefficients).reshape(clean_field.shape)
    residual = clean_field - fitted
    parameters = {
        f"coef_{name}": float(value) for name, value in zip(coefficient_names, coefficients)
    }
    parameters.update(
        {
            "fit_degree": float(degree),
            "fit_pixel_count": float(sampled_values.size),
            "fit_rank": float(matrix_rank),
            "fit_residual_sum": float(residual_sum[0]) if residual_sum.size else 0.0,
            "fit_condition_proxy": float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > 0
            else float("nan"),
        }
    )
    return ReferenceResult(model_name=model_name, fitted=fitted, residual=residual, parameters=parameters)


def _step_masks(shape: Tuple[int, int], edge_index: int, axis: str, edge_band_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count, column_count = shape
    if axis == "x":
        coordinate_indices = np.arange(column_count)[None, :]
        left_mask = np.broadcast_to(coordinate_indices < edge_index - edge_band_px, shape)
        right_mask = np.broadcast_to(coordinate_indices > edge_index + edge_band_px, shape)
    elif axis == "y":
        coordinate_indices = np.arange(row_count)[:, None]
        left_mask = np.broadcast_to(coordinate_indices < edge_index - edge_band_px, shape)
        right_mask = np.broadcast_to(coordinate_indices > edge_index + edge_band_px, shape)
    else:
        raise ValueError("step_axis must be 'x' or 'y'")
    edge_mask = ~(left_mask | right_mask)
    return left_mask, right_mask, edge_mask


def _detect_step_edge(field: np.ndarray, axis: str) -> int:
    if axis == "x":
        profile = np.nanmean(field, axis=0)
    elif axis == "y":
        profile = np.nanmean(field, axis=1)
    else:
        raise ValueError("step_axis must be 'x' or 'y'")
    smoothed_profile = gaussian_filter1d(profile, sigma=2.0, mode="nearest")
    gradient_profile = np.gradient(smoothed_profile)
    return int(np.argmax(np.abs(gradient_profile)))


def fit_step_form(field: np.ndarray, dataset: DatasetConfig) -> ReferenceResult:
    """Fit a two-plateau step reference with an interpolated edge band."""

    clean_field = np.asarray(field, dtype=np.float64)
    axis = dataset.step_axis.lower()
    edge_index = _detect_step_edge(clean_field, axis=axis)
    left_mask, right_mask, edge_mask = _step_masks(
        clean_field.shape, edge_index, axis=axis, edge_band_px=dataset.step_edge_band_px
    )
    finite_mask = np.isfinite(clean_field)
    left_fit_mask = left_mask & finite_mask
    right_fit_mask = right_mask & finite_mask
    if int(left_fit_mask.sum()) == 0 or int(right_fit_mask.sum()) == 0:
        raise ValueError("Detected step edge leaves an empty plateau region")

    plateau_model = dataset.step_plateau_model.lower()
    if plateau_model == "constant":
        left_level = float(np.median(clean_field[left_fit_mask]))
        right_level = float(np.median(clean_field[right_fit_mask]))
        fitted = np.where(left_mask, left_level, right_level).astype(np.float64)
        if int(edge_mask.sum()) > 0:
            fitted[edge_mask] = 0.5 * (left_level + right_level)
        parameters = {
            "edge_index_px": float(edge_index),
            "edge_band_px": float(dataset.step_edge_band_px),
            "left_level": left_level,
            "right_level": right_level,
            "step_height": right_level - left_level,
            "left_pixel_count": float(left_fit_mask.sum()),
            "right_pixel_count": float(right_fit_mask.sum()),
        }
        return ReferenceResult(
            model_name="step_constant",
            fitted=fitted,
            residual=clean_field - fitted,
            parameters=parameters,
        )

    if plateau_model != "plane":
        raise ValueError("step_plateau_model must be 'constant' or 'plane'")

    left_result = fit_polynomial_form(
        clean_field, degree=1, max_pixels=dataset.fit_max_pixels, mask=left_fit_mask, model_name="left_plane"
    )
    right_result = fit_polynomial_form(
        clean_field, degree=1, max_pixels=dataset.fit_max_pixels, mask=right_fit_mask, model_name="right_plane"
    )
    fitted = np.where(left_mask, left_result.fitted, right_result.fitted).astype(np.float64)
    if int(edge_mask.sum()) > 0:
        fitted[edge_mask] = 0.5 * (left_result.fitted[edge_mask] + right_result.fitted[edge_mask])
    left_level = float(np.median(left_result.fitted[left_fit_mask]))
    right_level = float(np.median(right_result.fitted[right_fit_mask]))
    parameters = {
        "edge_index_px": float(edge_index),
        "edge_band_px": float(dataset.step_edge_band_px),
        "left_level": left_level,
        "right_level": right_level,
        "step_height": right_level - left_level,
        "left_pixel_count": float(left_fit_mask.sum()),
        "right_pixel_count": float(right_fit_mask.sum()),
    }
    for key, value in left_result.parameters.items():
        parameters[f"left_{key}"] = value
    for key, value in right_result.parameters.items():
        parameters[f"right_{key}"] = value
    return ReferenceResult(model_name="step_plane", fitted=fitted, residual=clean_field - fitted, parameters=parameters)


def fit_reference(field: np.ndarray, dataset: DatasetConfig) -> ReferenceResult:
    """Dispatch to the configured reference model."""

    model_name = dataset.reference_model.lower()
    if model_name in {"quadratic", "quadratic_cap", "sphere_surrogate"}:
        return fit_polynomial_form(
            field, degree=2, max_pixels=dataset.fit_max_pixels, model_name="quadratic_cap"
        )
    if model_name in {"plane", "linear"}:
        return fit_polynomial_form(field, degree=1, max_pixels=dataset.fit_max_pixels, model_name="plane")
    if model_name in {"constant", "mean"}:
        return fit_polynomial_form(field, degree=0, max_pixels=dataset.fit_max_pixels, model_name="constant")
    if model_name in {"step", "step_plane", "step_constant"}:
        return fit_step_form(field, dataset=dataset)
    raise ValueError(f"Unsupported reference_model: {dataset.reference_model}")
