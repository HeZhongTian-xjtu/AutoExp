from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ValidationIssue(BaseModel):
    """Stable, serializable validation result used across all boundaries."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warning", "error"] = "error"
    phase: str
    file: str | None = None
    line: int | None = None
    message: str
    suggestion: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    discovered_imports: list[str] = Field(default_factory=list)
    environment_fingerprint: str | None = None

    def format_report(self) -> str:
        if self.passed:
            return "Preflight passed."
        lines = ["Preflight blocked execution:"]
        for issue in self.issues:
            location = f" ({issue.file}:{issue.line})" if issue.file else ""
            lines.append(f"- [{issue.code}] {issue.message}{location}")
            if issue.suggestion:
                lines.append(f"  Suggested action: {issue.suggestion}")
        return "\n".join(lines)
