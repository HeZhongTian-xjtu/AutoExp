from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepairSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_file: str = Field(min_length=1, max_length=240)
    patch: str = Field(min_length=1, max_length=100_000)
    reason: str = Field(min_length=1, max_length=2_000)
    expected_base_sha256: str | None = None
    verification: str = Field(default="preflight_and_trial", max_length=80)


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    target_file: str
    base_sha256: str | None = None
    patched_sha256: str | None = None
    preflight_passed: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
