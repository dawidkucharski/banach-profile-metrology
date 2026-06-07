"""Preprocessing for interferometric frame fields."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import PreprocessConfig


def apply_preprocess(field: np.ndarray, config: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply deterministic preprocessing and return processed field plus audit data."""

    processed = np.asarray(field, dtype=np.float64)
    audit: Dict[str, float] = {}

    if config.crop is not None:
        if len(config.crop) != 4:
            raise ValueError("crop must be [row_start, row_stop, column_start, column_stop]")
        row_start, row_stop, column_start, column_stop = config.crop
        processed = processed[row_start:row_stop, column_start:column_stop]

    if config.height_scale is not None:
        processed = processed * float(config.height_scale) + float(config.height_offset)
        audit["height_scale"] = float(config.height_scale)
        audit["height_offset"] = float(config.height_offset)

    if config.clip_quantiles is not None:
        if len(config.clip_quantiles) != 2:
            raise ValueError("clip_quantiles must contain two probabilities")
        lower_probability, upper_probability = config.clip_quantiles
        lower_value, upper_value = np.quantile(processed, [lower_probability, upper_probability])
        processed = np.clip(processed, lower_value, upper_value)
        audit["clip_lower"] = float(lower_value)
        audit["clip_upper"] = float(upper_value)

    normalisation = config.normalisation.lower()
    if normalisation == "none":
        return processed, audit
    if normalisation == "centre":
        centre_value = float(np.median(processed))
        audit["normalisation_centre"] = centre_value
        return processed - centre_value, audit
    if normalisation == "standard":
        mean_value = float(np.mean(processed))
        std_value = float(np.std(processed))
        scale_value = std_value if std_value > 0 else 1.0
        audit["normalisation_mean"] = mean_value
        audit["normalisation_scale"] = scale_value
        return (processed - mean_value) / scale_value, audit
    if normalisation == "robust_zscore":
        median_value = float(np.median(processed))
        lower_quartile, upper_quartile = np.quantile(processed, [0.25, 0.75])
        interquartile_range = float(upper_quartile - lower_quartile)
        fallback_scale = float(np.std(processed))
        scale_value = interquartile_range if interquartile_range > 0 else fallback_scale
        if scale_value <= 0:
            scale_value = 1.0
        audit["normalisation_median"] = median_value
        audit["normalisation_scale"] = scale_value
        return (processed - median_value) / scale_value, audit
    if normalisation == "minmax":
        min_value = float(np.min(processed))
        max_value = float(np.max(processed))
        scale_value = max_value - min_value
        if scale_value <= 0:
            scale_value = 1.0
        audit["normalisation_min"] = min_value
        audit["normalisation_scale"] = scale_value
        return (processed - min_value) / scale_value, audit
    raise ValueError(f"Unsupported normalisation: {config.normalisation}")
