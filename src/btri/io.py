"""Image discovery, selection, and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    """A discovered PNG frame with its parsed sequence index."""

    index: int
    path: Path


def discover_pngs(folder: Path) -> List[ImageRecord]:
    """Discover numerically named PNG files without printing directory contents."""

    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    records: List[ImageRecord] = []
    for image_path in folder.glob("*.png"):
        try:
            sequence_index = int(image_path.stem)
        except ValueError:
            continue
        records.append(ImageRecord(index=sequence_index, path=image_path))
    return sorted(records, key=lambda record: record.index)


def continuity_report(records: Sequence[ImageRecord]) -> Dict[str, object]:
    """Report sequence continuity for a discovered dataset."""

    if not records:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "min_index": None,
            "max_index": None,
            "missing_count": None,
            "duplicate_numeric_stems": None,
        }
    indices = [record.index for record in records]
    unique_indices = set(indices)
    expected_count = max(indices) - min(indices) + 1
    return {
        "count": len(records),
        "first": records[0].path.name,
        "last": records[-1].path.name,
        "min_index": min(indices),
        "max_index": max(indices),
        "expected_consecutive_count": expected_count,
        "missing_count": expected_count - len(unique_indices),
        "duplicate_numeric_stems": len(indices) - len(unique_indices),
    }


def select_records(
    records: Sequence[ImageRecord],
    stride: int = 1,
    max_frames: Optional[int] = None,
    frame_selection: str = "all",
    start_index: Optional[int] = None,
    stop_index: Optional[int] = None,
) -> List[ImageRecord]:
    """Select frames deterministically for full or pilot analyses."""

    if stride < 1:
        raise ValueError("stride must be at least 1")
    filtered = [
        record
        for record in records
        if (start_index is None or record.index >= start_index)
        and (stop_index is None or record.index <= stop_index)
    ]
    strided = filtered[::stride]
    if max_frames is None or max_frames >= len(strided):
        return list(strided)
    if max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    if frame_selection == "first":
        return list(strided[:max_frames])
    if frame_selection == "stratified":
        selected_positions = np.linspace(0, len(strided) - 1, max_frames, dtype=int)
        return [strided[int(position)] for position in selected_positions]
    if frame_selection == "all":
        return list(strided[:max_frames])
    raise ValueError(f"Unsupported frame_selection: {frame_selection}")


def load_grayscale(path: Path) -> np.ndarray:
    """Load a PNG frame as a float64 grayscale array."""

    with Image.open(path) as image:
        grayscale = image.convert("L")
        return np.asarray(grayscale, dtype=np.float64)


def image_metadata(path: Path) -> Dict[str, object]:
    """Return compact metadata and intensity statistics for one image."""

    with Image.open(path) as image:
        array = np.asarray(image)
        return {
            "file": path.name,
            "mode": image.mode,
            "width": image.size[0],
            "height": image.size[1],
            "array_shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
            "mean": float(np.nanmean(array)),
            "std": float(np.nanstd(array)),
        }


def representative_metadata(records: Sequence[ImageRecord]) -> List[Dict[str, object]]:
    """Inspect only first, middle, and last frames."""

    if not records:
        return []
    positions = sorted(set([0, len(records) // 2, len(records) - 1]))
    metadata = []
    for position in positions:
        payload = image_metadata(records[position].path)
        payload["sequence_index"] = records[position].index
        payload["position"] = position
        metadata.append(payload)
    return metadata


def iter_paths(records: Iterable[ImageRecord]) -> Iterable[Path]:
    """Yield paths from records."""

    for record in records:
        yield record.path
