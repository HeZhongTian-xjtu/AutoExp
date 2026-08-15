from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autoexp.domain.errors import ValidationIssue


class PrimaryMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    direction: Literal["minimize", "maximize"]

    @field_validator("value")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric must be finite")
        return value


class MetricsDocument(BaseModel):
    dataset_id: str | None = None
    dataset_sha256: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    trial_id: UUID | None = None
    primary: PrimaryMetric
    secondary: dict[str, float | int] = Field(default_factory=dict)
    parameters: dict[str, Any]
    seed: int


def load_metrics(
    path: Path,
    expected_metric: str,
    expected_direction: str,
    expected_parameters: dict[str, Any],
    *,
    expected_dataset_id: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_trial_id: UUID | None = None,
) -> tuple[MetricsDocument | None, list[ValidationIssue]]:
    try:
        document = MetricsDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [
            ValidationIssue(
                code="METRIC_MISSING",
                phase="metric_check",
                message=f"metric file does not exist: {path}",
                suggestion="Make the fixed evaluation entrypoint write metrics.json.",
            )
        ]
    except (ValueError, TypeError) as exc:
        return None, [
            ValidationIssue(
                code="METRIC_INVALID",
                phase="metric_check",
                message=f"metric document is invalid: {exc}",
                suggestion="Write the documented metrics schema.",
            )
        ]

    issues: list[ValidationIssue] = []
    if (
        document.primary.name != expected_metric
        or document.primary.direction != expected_direction
    ):
        issues.append(
            ValidationIssue(
                code="METRIC_CONTRACT_MISMATCH",
                phase="metric_check",
                message="primary metric does not match ExperimentSpec",
                suggestion="Use the registered metric and optimization direction.",
            )
        )
    if document.parameters != expected_parameters:
        issues.append(
            ValidationIssue(
                code="METRIC_PARAMETER_MISMATCH",
                phase="metric_check",
                message="reported parameters do not match the executed trial",
                suggestion="Report the exact parameters used by the training script.",
            )
        )
    if expected_dataset_id is not None and document.dataset_id != expected_dataset_id:
        issues.append(
            ValidationIssue(
                code="METRIC_DATASET_MISMATCH",
                phase="metric_check",
                message="reported dataset_id does not match the registered dataset",
                suggestion="Use the immutable dataset manifest identity.",
            )
        )
    if (
        expected_dataset_sha256 is not None
        and document.dataset_sha256 != expected_dataset_sha256
    ):
        issues.append(
            ValidationIssue(
                code="METRIC_DATASET_FINGERPRINT_MISMATCH",
                phase="metric_check",
                message="reported dataset fingerprint does not match the verified asset",
                suggestion="Evaluate the exact dataset registered before execution.",
            )
        )
    if expected_seed is not None and document.seed != expected_seed:
        issues.append(
            ValidationIssue(
                code="METRIC_SEED_MISMATCH",
                phase="metric_check",
                message="reported seed does not match the executed trial",
                suggestion="Report the seed used by the training script.",
            )
        )
    if expected_trial_id is not None and document.trial_id != expected_trial_id:
        issues.append(
            ValidationIssue(
                code="METRIC_TRIAL_MISMATCH",
                phase="metric_check",
                message="reported trial_id does not match the executed trial",
                suggestion="Include AUTOEXP_TRIAL_ID in the fixed evaluator output.",
            )
        )
    return (document if not issues else None), issues
