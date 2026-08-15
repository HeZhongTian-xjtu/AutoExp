from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from autoexp.domain.errors import PreflightReport, ValidationIssue
from autoexp.evaluation.dataset import DatasetIntegrityReport
from autoexp.evaluation.metrics import MetricsDocument
from autoexp.execution.base import ExecutionResult
from autoexp.validation.models import ValidationReport


class TrialOutcome(BaseModel):
    """Persistable result of one candidate experiment execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    trial_id: UUID
    parameters: dict[str, Any]
    status: Literal[
        "succeeded",
        "preflight_failed",
        "validation_failed",
        "execution_failed",
        "metric_failed",
    ]
    workspace: Path
    preflight: PreflightReport | None = None
    validation: ValidationReport | None = None
    dataset_integrity: DatasetIntegrityReport | None = None
    execution: ExecutionResult | None = None
    evaluation_execution: ExecutionResult | None = None
    metrics: MetricsDocument | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
