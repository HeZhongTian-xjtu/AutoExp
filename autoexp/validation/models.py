from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal


from pydantic import BaseModel, ConfigDict, Field

from autoexp.domain.errors import ValidationIssue
from autoexp.execution import ExecutionResult


class GateResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    status: Literal["passed", "failed", "skipped"]
    command: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    exit_code: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    execution: ExecutionResult | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)


class ValidationReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    gates: list[GateResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(gate.status in {"passed", "skipped"} for gate in self.gates)

    def issues(self) -> list[ValidationIssue]:
        return [issue for gate in self.gates for issue in gate.issues]
