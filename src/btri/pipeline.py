"""End-to-end BTRI analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

import pandas as pd

from .config import DatasetConfig, PipelineConfig, resolve_project_path
from .descriptors import banach_descriptors, classical_metrics
from .io import continuity_report, discover_pngs, load_grayscale, representative_metadata, select_records
from .plots import write_metric_plots
from .preprocess import apply_preprocess
from .reference import fit_reference
from .uncertainty import summarise_iso_gum


@dataclass
class DatasetResult:
    """Output paths and summary table for one dataset."""

    dataset: str
    frame_table_path: Path
    inventory_path: Path
    summary_path: Path
    frame_count: int


@dataclass
class PipelineResult:
    """Output paths for a pipeline run."""

    output_dir: Path
    dataset_results: List[DatasetResult]
    combined_summary_path: Optional[Path]


def _frame_record(dataset: DatasetConfig, image_path: Path, sequence_index: int, config: PipelineConfig) -> Dict[str, object]:
    raw_field = load_grayscale(image_path)
    processed_field, preprocess_audit = apply_preprocess(raw_field, dataset.preprocess)
    reference_result = fit_reference(processed_field, dataset)
    gradient_spacing = tuple(float(value) for value in config.descriptors.gradient_spacing)
    if len(gradient_spacing) != 2:
        raise ValueError("descriptors.gradient_spacing must contain [row_spacing, column_spacing]")
    record: Dict[str, object] = {
        "dataset": dataset.name,
        "file": image_path.name,
        "path": str(image_path),
        "sequence_index": int(sequence_index),
        "row_count": int(processed_field.shape[0]),
        "column_count": int(processed_field.shape[1]),
        "reference_model": reference_result.model_name,
    }
    for key, value in preprocess_audit.items():
        record[f"preprocess_{key}"] = value
    record.update(classical_metrics(processed_field, reference_result.residual))
    record.update(
        banach_descriptors(
            reference_result.residual,
            gradient_spacing=(gradient_spacing[0], gradient_spacing[1]),
            scale_sigmas_px=config.descriptors.scale_sigmas_px,
        )
    )
    for key, value in reference_result.parameters.items():
        record[key] = value
    return record


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def process_dataset(dataset: DatasetConfig, config: PipelineConfig, output_dir: Path) -> DatasetResult:
    """Process one dataset and write frame-level and summary outputs."""

    dataset_dir = resolve_project_path(config.project_root, dataset.path)
    discovered_records = discover_pngs(dataset_dir)
    selected_records = select_records(
        discovered_records,
        stride=dataset.stride,
        max_frames=dataset.max_frames,
        frame_selection=dataset.frame_selection,
        start_index=dataset.start_index,
        stop_index=dataset.stop_index,
    )
    dataset_output_dir = output_dir / dataset.name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = dataset_output_dir / "inventory.json"
    inventory_payload = {
        "dataset": dataset.name,
        "path": str(dataset_dir),
        "discovered": continuity_report(discovered_records),
        "selected": continuity_report(selected_records),
        "representative_images": representative_metadata(discovered_records),
        "selection": {
            "stride": dataset.stride,
            "max_frames": dataset.max_frames,
            "frame_selection": dataset.frame_selection,
            "start_index": dataset.start_index,
            "stop_index": dataset.stop_index,
        },
    }
    _write_json(inventory_path, inventory_payload)

    records: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    for selected_position, image_record in enumerate(selected_records, start=1):
        record = _frame_record(dataset, image_record.path, image_record.index, config)
        record["frame_index"] = int(selected_position - 1)
        records.append(record)
    elapsed_seconds = time.perf_counter() - start_time

    frame_table = pd.DataFrame(records)
    frame_table_path = dataset_output_dir / "frame_descriptors.csv"
    frame_table.to_csv(frame_table_path, index=False)

    summary_table = summarise_iso_gum(frame_table, dataset=dataset, uncertainty=config.uncertainty)
    summary_table["processing_elapsed_seconds"] = elapsed_seconds
    summary_path = dataset_output_dir / "iso5725_gum_summary.csv"
    summary_table.to_csv(summary_path, index=False)

    if config.make_plots and not frame_table.empty:
        write_metric_plots(frame_table, dataset_output_dir / "figures")

    return DatasetResult(
        dataset=dataset.name,
        frame_table_path=frame_table_path,
        inventory_path=inventory_path,
        summary_path=summary_path,
        frame_count=len(selected_records),
    )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run all configured datasets and write combined outputs."""

    output_dir = resolve_project_path(config.project_root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(output_dir / "config_resolved.json")
    dataset_results = []
    summary_tables = []
    for dataset in config.datasets:
        dataset_result = process_dataset(dataset, config=config, output_dir=output_dir)
        dataset_results.append(dataset_result)
        summary_tables.append(pd.read_csv(dataset_result.summary_path))

    combined_summary_path: Optional[Path] = None
    if summary_tables:
        combined_summary = pd.concat(summary_tables, ignore_index=True)
        combined_summary_path = output_dir / "iso5725_gum_summary.csv"
        combined_summary.to_csv(combined_summary_path, index=False)

    return PipelineResult(
        output_dir=output_dir,
        dataset_results=dataset_results,
        combined_summary_path=combined_summary_path,
    )
