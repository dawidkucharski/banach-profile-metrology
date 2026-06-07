"""Command line interface for the BTRI pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .pipeline import run_pipeline


def _parse_dataset_names(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    names: List[str] = []
    for value in values:
        names.extend([name.strip() for name in value.split(",") if name.strip()])
    return names or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Banach-space texture and roundness imaging analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/btri_default.json"), help="JSON config path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dataset", action="append", help="Dataset name to run; may be repeated or comma-separated")
    parser.add_argument("--max-frames", type=int, default=None, help="Override maximum frames per dataset")
    parser.add_argument("--stride", type=int, default=None, help="Override frame stride per dataset")
    parser.add_argument(
        "--frame-selection",
        choices=["all", "first", "stratified"],
        default=None,
        help="Override frame selection strategy when --max-frames is used",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG diagnostic figures")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = PipelineConfig.from_json(arguments.config)

    selected_names = _parse_dataset_names(arguments.dataset)
    datasets = config.datasets
    if selected_names is not None:
        selected_name_set = set(selected_names)
        datasets = [dataset for dataset in datasets if dataset.name in selected_name_set]
        missing_names = selected_name_set - {dataset.name for dataset in datasets}
        if missing_names:
            parser.error(f"Unknown dataset(s): {', '.join(sorted(missing_names))}")

    overridden_datasets = []
    for dataset in datasets:
        updated_dataset = dataset
        if arguments.max_frames is not None:
            updated_dataset = replace(updated_dataset, max_frames=arguments.max_frames)
        if arguments.stride is not None:
            updated_dataset = replace(updated_dataset, stride=arguments.stride)
        if arguments.frame_selection is not None:
            updated_dataset = replace(updated_dataset, frame_selection=arguments.frame_selection)
        overridden_datasets.append(updated_dataset)

    if arguments.output_dir is not None:
        config = replace(config, output_dir=arguments.output_dir)
    if arguments.no_plots:
        config = replace(config, make_plots=False)
    config = replace(config, datasets=overridden_datasets)

    result = run_pipeline(config)
    print(f"BTRI output directory: {result.output_dir}")
    for dataset_result in result.dataset_results:
        print(
            f"{dataset_result.dataset}: {dataset_result.frame_count} frames, "
            f"descriptors={dataset_result.frame_table_path}, summary={dataset_result.summary_path}"
        )
    if result.combined_summary_path is not None:
        print(f"Combined summary: {result.combined_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
