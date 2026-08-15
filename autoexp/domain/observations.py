from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .models import ParameterRange


class TrialObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: UUID
    ordinal: int = Field(ge=1)
    status: Literal[
        "succeeded",
        "preflight_failed",
        "validation_failed",
        "execution_failed",
        "metric_failed",
    ]
    parameters: dict[str, Any]
    primary_metric: float | None = None
    secondary_metrics: dict[str, float | int] = Field(default_factory=dict)
    metric_details: dict[str, Any] = Field(default_factory=dict)
    dataset_sha256: str | None = None
    preflight_passed: bool = False
    failure_codes: list[str] = Field(default_factory=list)
    failure_messages: list[str] = Field(default_factory=list)


class ExperimentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    template_id: str
    dataset_id: str
    model_id: str
    objective: str
    hypothesis: str
    metric_name: str
    metric_direction: Literal["minimize", "maximize"]
    current_parameters: dict[str, Any] = Field(default_factory=dict)
    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, ParameterRange] = Field(default_factory=dict)
    trials: list[TrialObservation] = Field(default_factory=list)
    latest_trial_id: UUID | None = None
    latest_status: str | None = None
    latest_failure_codes: list[str] = Field(default_factory=list)
    latest_failure_messages: list[str] = Field(default_factory=list)
    failure_fingerprint: str | None = None
    failure_context: dict[str, Any] = Field(default_factory=dict)
    best_trial_id: UUID | None = None
    best_metric: float | None = None
    metric_history: list[float] = Field(default_factory=list)
    dataset_sha256: str | None = None
    remaining_trials: int = Field(ge=0)
    stale_trials: int = Field(ge=0)
    patience: int = Field(ge=1)
    target_metric: float | None = None
    max_repairs_per_trial: int = Field(ge=0)
    repair_count: int = Field(default=0, ge=0)
