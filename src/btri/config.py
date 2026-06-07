"""Configuration objects for the BTRI analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessConfig:
    """Frame-level preprocessing options."""

    normalisation: str = "none"
    crop: Optional[List[int]] = None
    clip_quantiles: Optional[List[float]] = None
    height_scale: Optional[float] = None
    height_offset: float = 0.0


@dataclass
class DatasetConfig:
    """Dataset-specific analysis options."""

    name: str
    path: str
    kind: str
    reference_model: str
    stride: int = 1
    max_frames: Optional[int] = None
    frame_selection: str = "all"
    start_index: Optional[int] = None
    stop_index: Optional[int] = None
    fit_max_pixels: int = 20000
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    step_axis: str = "x"
    step_edge_band_px: int = 8
    step_plateau_model: str = "plane"
    nominal_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class DescriptorConfig:
    """Functional descriptor settings."""

    gradient_spacing: List[float] = field(default_factory=lambda: [1.0, 1.0])
    scale_sigmas_px: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])


@dataclass
class UncertaintyConfig:
    """ISO 5725 and GUM-style summary settings."""

    coverage_factor: float = 2.0
    block_size: int = 100
    type_b_absolute: float = 0.0
    type_b_relative: float = 0.0
    quantization_step: float = 0.0
    type_b_absolute_by_metric: Dict[str, float] = field(default_factory=dict)
    type_b_relative_by_metric: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    project_root: str = "."
    output_dir: str = "results/btri"
    make_plots: bool = True
    datasets: List[DatasetConfig] = field(default_factory=list)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        dataset_payloads = payload.get("datasets", [])
        datasets = []
        for dataset_payload in dataset_payloads:
            preprocess_payload = dataset_payload.get("preprocess", {})
            merged_payload = dict(dataset_payload)
            merged_payload["preprocess"] = PreprocessConfig(**preprocess_payload)
            datasets.append(DatasetConfig(**merged_payload))

        descriptor_payload = payload.get("descriptors", {})
        uncertainty_payload = payload.get("uncertainty", {})
        return cls(
            project_root=payload.get("project_root", "."),
            output_dir=payload.get("output_dir", "results/btri"),
            make_plots=payload.get("make_plots", True),
            datasets=datasets,
            descriptors=DescriptorConfig(**descriptor_payload),
            uncertainty=UncertaintyConfig(**uncertainty_payload),
        )

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")


def resolve_project_path(project_root: str, target_path: str) -> Path:
    """Resolve a project-relative or absolute path."""

    candidate = Path(target_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(project_root).expanduser() / candidate).resolve()
