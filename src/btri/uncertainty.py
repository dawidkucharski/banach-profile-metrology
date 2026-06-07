"""ISO 5725 and GUM-style statistical summaries."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import DatasetConfig, UncertaintyConfig


NON_METRIC_COLUMNS = {
    "dataset",
    "file",
    "path",
    "reference_model",
    "frame_index",
    "sequence_index",
    "row_count",
    "column_count",
}


def metric_columns(frame_table: pd.DataFrame) -> List[str]:
    """Return numeric columns that represent measured quantities."""

    columns: List[str] = []
    for column_name in frame_table.columns:
        if column_name in NON_METRIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame_table[column_name]):
            columns.append(column_name)
    return columns


def _anova_reproducibility(values: np.ndarray, block_size: int) -> Dict[str, float]:
    if block_size <= 1 or values.size < block_size * 2:
        return {
            "block_count": float("nan"),
            "within_block_sd": float("nan"),
            "between_block_sd": float("nan"),
            "reproducibility_sd": float("nan"),
        }
    block_ids = np.arange(values.size) // block_size
    unique_blocks = np.unique(block_ids)
    if unique_blocks.size < 2:
        return {
            "block_count": float(unique_blocks.size),
            "within_block_sd": float("nan"),
            "between_block_sd": float("nan"),
            "reproducibility_sd": float("nan"),
        }

    grand_mean = float(np.mean(values))
    within_sum_squares = 0.0
    between_sum_squares = 0.0
    block_sizes = []
    for block_id in unique_blocks:
        block_values = values[block_ids == block_id]
        block_sizes.append(block_values.size)
        block_mean = float(np.mean(block_values))
        within_sum_squares += float(np.sum((block_values - block_mean) ** 2))
        between_sum_squares += float(block_values.size * (block_mean - grand_mean) ** 2)

    within_degrees = values.size - unique_blocks.size
    between_degrees = unique_blocks.size - 1
    if within_degrees <= 0 or between_degrees <= 0:
        return {
            "block_count": float(unique_blocks.size),
            "within_block_sd": float("nan"),
            "between_block_sd": float("nan"),
            "reproducibility_sd": float("nan"),
        }
    mean_square_within = within_sum_squares / within_degrees
    mean_square_between = between_sum_squares / between_degrees
    mean_block_size = float(np.mean(block_sizes))
    between_variance = max((mean_square_between - mean_square_within) / mean_block_size, 0.0)
    reproducibility_variance = mean_square_within + between_variance
    return {
        "block_count": float(unique_blocks.size),
        "within_block_sd": float(np.sqrt(max(mean_square_within, 0.0))),
        "between_block_sd": float(np.sqrt(between_variance)),
        "reproducibility_sd": float(np.sqrt(max(reproducibility_variance, 0.0))),
    }


def summarise_iso_gum(
    frame_table: pd.DataFrame,
    dataset: DatasetConfig,
    uncertainty: UncertaintyConfig,
    columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Summarise repeatability, block reproducibility, and GUM uncertainty."""

    selected_columns = list(columns) if columns else metric_columns(frame_table)
    rows = []
    for metric_name in selected_columns:
        values = frame_table[metric_name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        sample_count = values.size
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if sample_count > 1 else 0.0
        type_a_uncertainty = std_value / float(np.sqrt(sample_count)) if sample_count > 0 else float("nan")
        type_b_absolute = uncertainty.type_b_absolute_by_metric.get(metric_name, uncertainty.type_b_absolute)
        type_b_relative = uncertainty.type_b_relative_by_metric.get(metric_name, uncertainty.type_b_relative)
        type_b_relative_component = abs(mean_value) * float(type_b_relative)
        quantization_component = float(uncertainty.quantization_step) / np.sqrt(12.0) if uncertainty.quantization_step > 0 else 0.0
        combined_uncertainty = float(
            np.sqrt(
                type_a_uncertainty**2
                + float(type_b_absolute) ** 2
                + type_b_relative_component**2
                + quantization_component**2
            )
        )
        nominal_value = dataset.nominal_values.get(metric_name, float("nan"))
        bias_value = mean_value - nominal_value if np.isfinite(nominal_value) else float("nan")
        anova_payload = _anova_reproducibility(values, uncertainty.block_size)
        rows.append(
            {
                "dataset": dataset.name,
                "metric": metric_name,
                "n": sample_count,
                "mean": mean_value,
                "standard_deviation": std_value,
                "iso5725_repeatability_sd": std_value,
                "iso5725_repeatability_limit_r": 2.8 * std_value,
                "gum_type_a_standard_uncertainty": type_a_uncertainty,
                "gum_type_b_absolute": float(type_b_absolute),
                "gum_type_b_relative_component": float(type_b_relative_component),
                "gum_quantization_component": float(quantization_component),
                "gum_combined_standard_uncertainty": combined_uncertainty,
                "gum_coverage_factor": float(uncertainty.coverage_factor),
                "gum_expanded_uncertainty": float(uncertainty.coverage_factor * combined_uncertainty),
                "nominal_value": nominal_value,
                "bias": bias_value,
                **anova_payload,
            }
        )
    return pd.DataFrame(rows)
