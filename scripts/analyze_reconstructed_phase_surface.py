"""Downstream surface analysis for already reconstructed phase maps.

This script starts after PSI phase retrieval and phase unwrapping. It consumes
an existing unwrapped phase map or calibrated height map, converts it to a
surface in nanometres when possible, fits a best-fit reference sphere, computes
the deviation map, and reports profile-style residual and Banach-space descriptors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from PIL import Image

from btri.descriptors import banach_descriptors


DEFAULT_WAVELENGTH_NM = 532.0


def load_numeric_map(path: Path, npz_key: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = npz_key if npz_key is not None else sorted(archive.files)[0]
        array = archive[key]
    elif suffix == ".csv":
        array = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    elif suffix in {".txt", ".dat"}:
        array = np.loadtxt(path, dtype=np.float64)
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim == 3:
            image_array = np.mean(image_array[..., : min(3, image_array.shape[2])], axis=2)
        array = image_array
    else:
        raise ValueError(f"Unsupported input map extension: {suffix}")

    array = np.asarray(array, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Input map must be two-dimensional after squeezing, got shape {array.shape}")
    return array


def load_optional_mask(path: Optional[Path], expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    mask = load_numeric_map(path) > 0
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match input shape {expected_shape}")
    return mask


def convert_to_surface_units(input_map: np.ndarray, input_unit: str, wavelength_nm: float) -> Tuple[np.ndarray, str, bool, float]:
    unit = input_unit.lower()
    if unit == "phase-rad":
        scale = wavelength_nm / (4.0 * np.pi)
        return input_map * scale, "nm", True, scale
    if unit in {"fringe-order", "phase-cycle", "phase-cycles"}:
        scale = wavelength_nm / 2.0
        return input_map * scale, "nm", True, scale
    if unit == "height-nm":
        return input_map.copy(), "nm", True, 1.0
    if unit == "height-um":
        return input_map * 1000.0, "nm", True, 1000.0
    if unit == "height-mm":
        return input_map * 1_000_000.0, "nm", True, 1_000_000.0
    if unit == "native":
        return input_map.copy(), "native units", False, 1.0
    raise ValueError(f"Unsupported input unit: {input_unit}")


def finite_mask_for(array: np.ndarray, user_mask: Optional[np.ndarray] = None) -> np.ndarray:
    mask = np.isfinite(array)
    if user_mask is not None:
        mask &= user_mask
    if int(mask.sum()) == 0:
        raise ValueError("No finite input pixels remain after applying the mask")
    return mask


def deterministic_sample_mask(mask: np.ndarray, max_pixels: int) -> np.ndarray:
    finite_count = int(mask.sum())
    if finite_count <= max_pixels:
        return mask.copy()
    stride = int(np.ceil(np.sqrt(finite_count / max_pixels)))
    sampled = np.zeros(mask.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    sampled &= mask
    if int(sampled.sum()) < max(16, max_pixels // 4):
        positions = np.flatnonzero(mask.ravel())
        count = min(max_pixels, positions.size)
        selected = positions[np.unique(np.linspace(0, positions.size - 1, count, dtype=int))]
        sampled = np.zeros(mask.size, dtype=bool)
        sampled[selected] = True
        sampled = sampled.reshape(mask.shape)
    return sampled


def coordinate_grid(shape: Tuple[int, int], pixel_size_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    row_indices, column_indices = np.indices(shape, dtype=np.float64)
    column_centre = 0.5 * (shape[1] - 1)
    row_centre = 0.5 * (shape[0] - 1)
    x_coordinates = (column_indices - column_centre) * pixel_size_nm
    y_coordinates = (row_centre - row_indices) * pixel_size_nm
    return x_coordinates, y_coordinates


def fit_best_fit_sphere(
    surface: np.ndarray,
    mask: np.ndarray,
    pixel_size_nm: float,
    max_fit_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    sampled_mask = deterministic_sample_mask(mask, max_pixels=max_fit_pixels)
    x_coordinates, y_coordinates = coordinate_grid(
        (int(surface.shape[0]), int(surface.shape[1])),
        pixel_size_nm=pixel_size_nm,
    )
    surface_offset = float(np.nanmedian(surface[mask]))
    centred_surface = surface - surface_offset

    x_sample = x_coordinates[sampled_mask]
    y_sample = y_coordinates[sampled_mask]
    z_sample = centred_surface[sampled_mask]
    design_matrix = np.column_stack([x_sample, y_sample, z_sample, np.ones_like(z_sample)])
    right_hand_side = -(x_sample**2 + y_sample**2 + z_sample**2)
    coefficients, residual_sum, matrix_rank, singular_values = np.linalg.lstsq(design_matrix, right_hand_side, rcond=None)

    sphere_centre_x = -0.5 * coefficients[0]
    sphere_centre_y = -0.5 * coefficients[1]
    sphere_centre_z_centred = -0.5 * coefficients[2]
    radius_squared = sphere_centre_x**2 + sphere_centre_y**2 + sphere_centre_z_centred**2 - coefficients[3]
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        raise ValueError("Algebraic best-fit sphere produced a non-positive radius")
    sphere_radius = float(np.sqrt(radius_squared))

    branch_sign = 1.0 if float(np.nanmedian(z_sample - sphere_centre_z_centred)) >= 0 else -1.0
    radial_argument = sphere_radius**2 - (x_coordinates - sphere_centre_x) ** 2 - (y_coordinates - sphere_centre_y) ** 2
    fitted_centred = np.full(surface.shape, np.nan, dtype=np.float64)
    valid_argument = radial_argument >= 0
    fitted_centred[valid_argument] = sphere_centre_z_centred + branch_sign * np.sqrt(radial_argument[valid_argument])
    fitted_sphere = fitted_centred + surface_offset
    deviation = surface - fitted_sphere

    parameters = {
        "sphere_centre_x_nm": float(sphere_centre_x),
        "sphere_centre_y_nm": float(sphere_centre_y),
        "sphere_centre_z_nm": float(sphere_centre_z_centred + surface_offset),
        "sphere_radius_nm": sphere_radius,
        "sphere_branch_sign": float(branch_sign),
        "sphere_fit_pixel_count": float(sampled_mask.sum()),
        "sphere_fit_rank": float(matrix_rank),
        "sphere_fit_residual_sum": float(residual_sum[0]) if residual_sum.size else 0.0,
        "sphere_fit_condition_proxy": float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else float("nan"),
    }
    return fitted_sphere, deviation, parameters


def compute_profile_residual_metrics(deviation: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = deviation[mask & np.isfinite(deviation)]
    if values.size == 0:
        raise ValueError("Deviation map contains no finite pixels for profile residual metrics")
    centred = values - float(np.mean(values))
    peak_positive = float(np.max(centred))
    peak_negative = float(np.min(centred))
    return {
        "profile_mean": float(np.mean(values)),
        "profile_Ra": float(np.mean(np.abs(centred))),
        "profile_Rq": float(np.sqrt(np.mean(centred**2))),
        "profile_Rp": peak_positive,
        "profile_Rv": float(abs(peak_negative)),
        "profile_Rz": float(peak_positive - peak_negative),
    }


def angular_roundness(deviation: np.ndarray, mask: np.ndarray, bin_count: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    row_indices, column_indices = np.indices(deviation.shape, dtype=np.float64)
    valid_rows = row_indices[mask]
    valid_columns = column_indices[mask]
    row_centre = float(np.mean(valid_rows))
    column_centre = float(np.mean(valid_columns))
    angles = np.mod(np.arctan2(row_centre - row_indices, column_indices - column_centre), 2.0 * np.pi)

    valid = mask & np.isfinite(deviation)
    bin_indices = np.floor(angles[valid] / (2.0 * np.pi) * bin_count).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    sums = np.bincount(bin_indices, weights=deviation[valid], minlength=bin_count)
    counts = np.bincount(bin_indices, minlength=bin_count)
    profile = np.full(bin_count, np.nan, dtype=np.float64)
    profile[counts > 0] = sums[counts > 0] / counts[counts > 0]

    theta = (np.arange(bin_count, dtype=np.float64) + 0.5) * (2.0 * np.pi / bin_count)
    valid_profile = np.isfinite(profile)
    if int(valid_profile.sum()) < 4:
        raise ValueError("Not enough angular bins to estimate roundness profile")
    design_matrix = np.column_stack(
        [np.ones(int(valid_profile.sum())), np.cos(theta[valid_profile]), np.sin(theta[valid_profile])]
    )
    coefficients, *_ = np.linalg.lstsq(design_matrix, profile[valid_profile], rcond=None)
    fitted_eccentricity = coefficients[0] + coefficients[1] * np.cos(theta) + coefficients[2] * np.sin(theta)
    residual_profile = profile - fitted_eccentricity
    residual_values = residual_profile[np.isfinite(residual_profile)]
    metrics = {
        "roundness_bin_count": float(bin_count),
        "roundness_valid_bin_count": float(valid_profile.sum()),
        "roundness_profile_mean": float(coefficients[0]),
        "roundness_eccentricity_cos": float(coefficients[1]),
        "roundness_eccentricity_sin": float(coefficients[2]),
        "roundness_eccentricity_amplitude": float(np.sqrt(coefficients[1] ** 2 + coefficients[2] ** 2)),
        "RONt": float(np.nanmax(residual_values) - np.nanmin(residual_values)),
        "RONq": float(np.sqrt(np.nanmean(residual_values**2))),
    }
    profile_table = pd.DataFrame(
        {
            "angle_deg": np.degrees(theta),
            "angular_mean_deviation": profile,
            "first_harmonic_eccentricity_fit": fitted_eccentricity,
            "roundness_residual": residual_profile,
            "pixel_count": counts,
        }
    )
    return profile_table, metrics


def robust_limits(array: np.ndarray, mask: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    values = array[mask & np.isfinite(array)]
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


def contour_panel(axis: Axes, data: np.ndarray, mask: np.ndarray, title: str, colourbar_label: str, colour_map: str) -> None:
    plot_data = np.where(mask & np.isfinite(data), data, np.nan)
    plot_data = downsample_for_vector_plot(plot_data)
    plot_mask = np.isfinite(plot_data)
    low_value, high_value = robust_limits(plot_data, plot_mask)
    levels = np.linspace(low_value, high_value, 42)
    row_axis = np.arange(plot_data.shape[0])
    column_axis = np.arange(plot_data.shape[1])
    contour = axis.contourf(column_axis, row_axis, plot_data, levels=levels, cmap=colour_map, extend="both")
    axis.set_title(title)
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    plt.colorbar(contour, ax=axis, fraction=0.046, pad=0.04, label=colourbar_label)


def write_step_pdf(
    output_path: Path,
    input_map: np.ndarray,
    surface: np.ndarray,
    fitted_sphere: np.ndarray,
    deviation: np.ndarray,
    mask: np.ndarray,
    profile_table: pd.DataFrame,
    metrics: Mapping[str, Any],
    input_unit: str,
    surface_unit: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    contour_panel(axes[0, 0], input_map, mask, f"Input map ({input_unit})", input_unit, "viridis")
    contour_panel(axes[0, 1], surface, mask, "Converted surface", surface_unit, "viridis")
    contour_panel(axes[0, 2], fitted_sphere, mask & np.isfinite(fitted_sphere), "Best-fit reference sphere", surface_unit, "cividis")
    contour_panel(axes[1, 0], deviation, mask & np.isfinite(deviation), "Deviation map", surface_unit, "coolwarm")

    axes[1, 1].plot(profile_table["angle_deg"], profile_table["angular_mean_deviation"], color="#555555", linewidth=0.9, label="angular mean")
    axes[1, 1].plot(profile_table["angle_deg"], profile_table["first_harmonic_eccentricity_fit"], color="#D55E00", linewidth=0.9, label="eccentricity fit")
    axes[1, 1].plot(profile_table["angle_deg"], profile_table["roundness_residual"], color="#0072B2", linewidth=0.9, label="residual")
    axes[1, 1].set_title("Angular profile and eccentricity removal")
    axes[1, 1].set_xlabel("angle, deg")
    axes[1, 1].set_ylabel(surface_unit)
    axes[1, 1].legend(frameon=False, fontsize=8)

    comparison_names = ["Ra", "Rq", "Rz", "RONt", "RONq", "L1", "L2", "Linf"]
    comparison_values = [
        metrics["profile_Ra"],
        metrics["profile_Rq"],
        metrics["profile_Rz"],
        metrics["RONt"],
        metrics["RONq"],
        metrics["banach_l1_mean"],
        metrics["banach_l2_rms"],
        metrics["banach_linf"],
    ]
    axes[1, 2].barh(comparison_names, comparison_values, color="#4C78A8")
    axes[1, 2].set_title("Profile-style and Banach descriptors")
    axes[1, 2].set_xlabel(surface_unit)
    axes[1, 2].invert_yaxis()
    figure.suptitle("Downstream analysis from reconstructed phase or height map", fontsize=13)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def write_roundness_pdf(output_path: Path, profile_table: pd.DataFrame, metrics: Mapping[str, Any], surface_unit: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(profile_table["angle_deg"], profile_table["angular_mean_deviation"], color="#555555", linewidth=0.9, label="angular mean")
    axes[0].plot(profile_table["angle_deg"], profile_table["first_harmonic_eccentricity_fit"], color="#D55E00", linewidth=0.9, label="first harmonic")
    axes[0].plot(profile_table["angle_deg"], profile_table["roundness_residual"], color="#0072B2", linewidth=0.9, label="roundness residual")
    axes[0].set_xlabel("angle, deg")
    axes[0].set_ylabel(surface_unit)
    axes[0].set_title("Eccentricity removal")
    axes[0].legend(frameon=False)

    bars = [metrics["roundness_eccentricity_amplitude"], metrics["RONt"], metrics["RONq"]]
    labels = ["eccentricity", "RONt", "RONq"]
    axes[1].bar(labels, bars, color=["#D55E00", "#0072B2", "#009E73"])
    axes[1].set_ylabel(surface_unit)
    axes[1].set_title("Roundness summary")
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def analyse(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_map = load_numeric_map(args.input, npz_key=args.npz_key)
    user_mask = load_optional_mask(args.mask, expected_shape=(int(input_map.shape[0]), int(input_map.shape[1])))
    surface, surface_unit, calibrated_to_nm, conversion_scale = convert_to_surface_units(
        input_map,
        input_unit=args.input_unit,
        wavelength_nm=args.wavelength_nm,
    )
    mask = finite_mask_for(surface, user_mask=user_mask)

    effective_pixel_size_nm = float(args.pixel_size_nm) if args.pixel_size_nm is not None else 1.0
    fitted_sphere, deviation, sphere_parameters = fit_best_fit_sphere(
        surface,
        mask=mask,
        pixel_size_nm=effective_pixel_size_nm,
        max_fit_pixels=args.max_fit_pixels,
    )
    deviation_mask = mask & np.isfinite(deviation)
    profile_metrics = compute_profile_residual_metrics(deviation, mask=deviation_mask)
    profile_table, roundness_metrics = angular_roundness(deviation, mask=deviation_mask, bin_count=args.angular_bins)
    banach_metrics = banach_descriptors(
        deviation,
        gradient_spacing=(effective_pixel_size_nm, effective_pixel_size_nm),
        scale_sigmas_px=args.scale_sigmas_px,
    )

    metrics: Dict[str, object] = {
        **profile_metrics,
        **roundness_metrics,
        **banach_metrics,
        **sphere_parameters,
        "input_rows": float(input_map.shape[0]),
        "input_columns": float(input_map.shape[1]),
        "valid_pixel_count": float(deviation_mask.sum()),
        "wavelength_nm": float(args.wavelength_nm),
        "input_to_surface_scale": float(conversion_scale),
        "surface_calibrated_to_nm": bool(calibrated_to_nm),
        "pixel_size_nm": float(args.pixel_size_nm) if args.pixel_size_nm is not None else float("nan"),
        "effective_pixel_size_for_fit": effective_pixel_size_nm,
    }

    np.save(output_dir / "surface_map.npy", surface)
    np.save(output_dir / "best_fit_sphere.npy", fitted_sphere)
    np.save(output_dir / "deviation_map.npy", deviation)

    summary_payload = {"input": str(args.input), "input_unit": args.input_unit, "surface_unit": surface_unit, **metrics}
    summary_csv = output_dir / "phase_surface_downstream_summary.csv"
    pd.DataFrame([summary_payload]).to_csv(summary_csv, index=False)

    profile_csv = output_dir / "phase_surface_roundness_profile.csv"
    profile_table.to_csv(profile_csv, index=False)

    summary_json = output_dir / "phase_surface_downstream_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({key: clean_json_value(value) for key, value in summary_payload.items()}, handle, indent=2)
        handle.write("\n")

    step_pdf = output_dir / "phase_surface_downstream_steps.pdf"
    roundness_pdf = output_dir / "phase_surface_roundness_profile.pdf"
    write_step_pdf(
        step_pdf,
        input_map=input_map,
        surface=surface,
        fitted_sphere=fitted_sphere,
        deviation=deviation,
        mask=mask,
        profile_table=profile_table,
        metrics=metrics,
        input_unit=args.input_unit,
        surface_unit=surface_unit,
    )
    write_roundness_pdf(roundness_pdf, profile_table=profile_table, metrics=metrics, surface_unit=surface_unit)

    return {
        "summary_csv": str(summary_csv),
        "profile_csv": str(profile_csv),
        "summary_json": str(summary_json),
        "step_pdf": str(step_pdf),
        "roundness_pdf": str(roundness_pdf),
        "calibrated_to_nm": calibrated_to_nm,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse an already reconstructed unwrapped phase or height map downstream of PSI/unwrapping."
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing reconstructed phase or height map")
    parser.add_argument("--mask", type=Path, default=None, help="Optional finite/valid mask map")
    parser.add_argument("--npz-key", default=None, help="Array key for .npz inputs; first sorted key is used if omitted")
    parser.add_argument(
        "--input-unit",
        choices=["phase-rad", "fringe-order", "phase-cycle", "phase-cycles", "height-nm", "height-um", "height-mm", "native"],
        default="phase-rad",
        help="Physical unit of the input map before downstream conversion",
    )
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM, help="Interferometer wavelength")
    parser.add_argument("--pixel-size-nm", type=float, default=None, help="Lateral object-plane pixel size for geometric sphere fitting")
    parser.add_argument("--max-fit-pixels", type=int, default=30000, help="Deterministic sample size for sphere fitting")
    parser.add_argument("--angular-bins", type=int, default=720, help="Angular bins for roundness-profile extraction")
    parser.add_argument("--scale-sigmas-px", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0], help="Gaussian scale descriptors")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase_surface_downstream"), help="Output directory")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = analyse(args)
    print(json.dumps(outputs, indent=2))
    if args.pixel_size_nm is None:
        print("Warning: --pixel-size-nm was not supplied; the sphere fit used pixel-index spacing for lateral coordinates.")
    if not outputs["calibrated_to_nm"]:
        print("Warning: input-unit=native, so outputs are not calibrated nanometre results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())