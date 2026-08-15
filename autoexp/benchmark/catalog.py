from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoexp.domain import TemplateManifest
from autoexp.application.templates import TemplateCatalog


PHASE1_TEMPLATE_IDS = (
    "housing-regression-v1",
    "covertype-classification-v1",
    "bank-marketing-classification-v1",
)


@dataclass(frozen=True)
class BenchmarkTask:
    """A reviewed, fixed task contract used by the Phase 1 benchmark."""

    template_id: str
    display_name: str
    description: str
    manifest: TemplateManifest

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    @property
    def metric_name(self) -> str:
        return self.manifest.metric_name

    @property
    def metric_direction(self) -> str:
        return self.manifest.metric_direction

    @property
    def baseline_parameters(self) -> dict[str, Any]:
        return self.manifest.baseline_parameter_values


class OptimizationBenchmarkCatalog:
    """Discover and validate the fixed Phase 1 optimization tasks."""

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.catalog = TemplateCatalog(self.project_root)

    def list(self) -> list[BenchmarkTask]:
        return [self.get(template_id) for template_id in PHASE1_TEMPLATE_IDS]

    def get(self, template_id: str) -> BenchmarkTask:
        try:
            descriptor = self.catalog.get(template_id)
        except ValueError as exc:
            raise KeyError(f"Phase 1 benchmark task not found: {template_id}") from exc
        manifest = descriptor.manifest
        self._validate(manifest)
        return BenchmarkTask(
            template_id=manifest.template_id,
            display_name=descriptor.display_name,
            description=manifest.description or "",
            manifest=manifest,
        )

    @staticmethod
    def _validate(manifest: TemplateManifest) -> None:
        baseline = manifest.baseline_parameter_values
        missing = set(manifest.parameter_policy) - set(baseline)
        if missing:
            raise ValueError(
                f"{manifest.template_id} baseline is missing parameters: {sorted(missing)}"
            )
        invalid = [
            name
            for name, value in baseline.items()
            if name in manifest.parameter_policy
            and not manifest.parameter_policy[name].accepts(value)
        ]
        if invalid:
            raise ValueError(
                f"{manifest.template_id} baseline violates parameter policy: {sorted(invalid)}"
            )
        if not manifest.dataset_id or not manifest.metric_name:
            raise ValueError(
                f"{manifest.template_id} must declare a dataset and metric"
            )
        if not manifest.validation:
            raise ValueError(f"{manifest.template_id} must declare validation gates")
