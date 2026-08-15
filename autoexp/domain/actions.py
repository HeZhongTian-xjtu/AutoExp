from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ParameterRange
from .observations import ExperimentObservation
from .repairs import RepairSpec


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["CONTINUE", "NARROW_SPACE", "REPAIR", "STOP", "HUMAN_REVIEW"]
    parameters: dict[str, Any] | None = None
    search_space: dict[str, ParameterRange] | None = None
    failure_code: str | None = None
    strategy: str | None = None
    repair: RepairSpec | None = None
    reason: str = Field(min_length=1, max_length=2_000)
    conclusion: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ActionDecision":
        if self.action == "CONTINUE" and not self.parameters:
            raise ValueError("CONTINUE action requires parameters")
        if self.action == "NARROW_SPACE" and not self.search_space:
            raise ValueError("NARROW_SPACE action requires search_space")
        if self.action == "REPAIR" and not self.repair:
            raise ValueError("REPAIR action requires a structured repair spec")
        return self


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    trial_id: UUID | None = None
    observation: ExperimentObservation
    decision: ActionDecision
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
