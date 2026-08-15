from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .actions import DecisionRecord
from .errors import ValidationIssue
from .models import ExperimentSpec
from .trials import TrialOutcome


class RunEvent(BaseModel):
    """A stable, serializable event in the experiment timeline."""

    model_config = ConfigDict(extra="forbid")

    name: str
    message: str
    trial_id: UUID | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExperimentRun:
    """The domain aggregate shared by workflow, persistence and presenters."""

    run_id: UUID = field(default_factory=uuid4)
    phase: str = "CREATED"
    started_at: datetime | None = None
    elapsed_seconds: float = 0.0
    current_trial_id: UUID | None = None
    next_parameters: dict[str, Any] | None = None
    active_template_root: Path | None = None
    status: str = "CREATED"
    spec: ExperimentSpec | None = None
    planner_metadata: dict[str, Any] = field(default_factory=dict)
    output_root: Path | None = None
    report_path: str | None = None
    ai_summary: str | None = None
    ai_summary_metadata: dict[str, Any] = field(default_factory=dict)
    outcomes: list[TrialOutcome] = field(default_factory=list)
    events: list[RunEvent] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def best(self) -> TrialOutcome | None:
        valid = [outcome for outcome in self.outcomes if outcome.metrics is not None]
        if not valid:
            return None
        direction = valid[0].metrics.primary.direction
        return (max if direction == "maximize" else min)(
            valid,
            key=lambda item: item.metrics.primary.value,
        )
