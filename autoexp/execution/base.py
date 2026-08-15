from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_count: int = Field(default=2, ge=1, le=64)
    memory_mb: int = Field(default=2048, ge=128, le=1_048_576)
    pids_limit: int = Field(default=128, ge=16, le=65_536)


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: UUID
    trial_id: UUID
    workspace: Path
    command: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(ge=1, le=86_400)
    environment: dict[str, str] = Field(default_factory=dict)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    network_policy: Literal["none", "restricted"] = "none"
    output_subdir: str = Field(default="working", min_length=1, max_length=120)
    image: str | None = Field(default=None, max_length=240)
    immutable_paths: list[str] = Field(default_factory=list)


class ClassifiedError(BaseModel):
    code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    status: Literal["succeeded", "failed", "timeout", "cancelled"]
    exit_code: int | None
    started_at: datetime
    finished_at: datetime
    stdout_path: Path
    stderr_path: Path
    output_truncated: bool = False
    resource_usage: dict[str, float | int | str] = Field(default_factory=dict)
    error: ClassifiedError | None = None


class Executor(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

    def cancel(self, trial_id: UUID) -> bool: ...
