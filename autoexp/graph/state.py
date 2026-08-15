from __future__ import annotations

from operator import add
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import UUID

from autoexp.domain import (
    DecisionRecord,
    ExperimentRun,
    ExperimentSpec,
    RunEvent,
    TrialOutcome,
)
from autoexp.domain.errors import ValidationIssue


class AutoExpGraphState(TypedDict, total=False):
    """JSON-compatible control state persisted by LangGraph checkpoints.

    Large logs and source files remain in ArtifactStore. The graph only keeps
    references, summaries and the current control decision.
    """

    schema_version: str
    run: dict[str, Any]
    spec: dict[str, Any]
    planning_request: dict[str, Any] | None
    output_root: str
    route: str
    candidates: list[dict[str, Any]]
    completed_keys: list[str]
    best_value: float | None
    stale_trials: int
    current_observation: dict[str, Any] | None
    pending_action: dict[str, Any] | None
    current_trial_id: str | None
    current_parameters: dict[str, Any] | None
    narrowed_space: dict[str, Any]
    retry_current: bool
    human_command: dict[str, Any] | None
    failure_fingerprint: str | None
    failure_context_ref: str | None
    code_context: dict[str, Any] | None
    code_context_ref: str | None
    repair_attempts: list[dict[str, Any]]
    pending_human_review: dict[str, Any] | None
    operation_keys: Annotated[list[str], add]
    graph_events: Annotated[list[dict[str, Any]], add]


GRAPH_SCHEMA_VERSION = "1.0"


def run_to_snapshot(run: ExperimentRun) -> dict[str, Any]:
    """Convert the existing aggregate to a checkpoint-safe JSON snapshot."""

    return {
        "run_id": str(run.run_id),
        "phase": run.phase,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "elapsed_seconds": run.elapsed_seconds,
        "current_trial_id": str(run.current_trial_id) if run.current_trial_id else None,
        "next_parameters": run.next_parameters,
        "active_template_root": (
            str(run.active_template_root) if run.active_template_root else None
        ),
        "status": run.status,
        "spec": run.spec.model_dump(mode="json") if run.spec else None,
        "planner_metadata": run.planner_metadata,
        "output_root": str(run.output_root) if run.output_root else None,
        "report_path": run.report_path,
        "ai_summary": run.ai_summary,
        "ai_summary_metadata": run.ai_summary_metadata,
        "outcomes": [item.model_dump(mode="json") for item in run.outcomes],
        "events": [item.model_dump(mode="json") for item in run.events],
        "decisions": [item.model_dump(mode="json") for item in run.decisions],
        "issues": [item.model_dump(mode="json") for item in run.issues],
    }


def snapshot_to_run(snapshot: dict[str, Any]) -> ExperimentRun:
    """Rehydrate the application aggregate from a checkpoint snapshot."""

    started_at = snapshot.get("started_at")
    return ExperimentRun(
        run_id=UUID(snapshot["run_id"]),
        phase=snapshot.get("phase", "CREATED"),
        started_at=datetime.fromisoformat(started_at) if started_at else None,
        elapsed_seconds=float(snapshot.get("elapsed_seconds", 0.0)),
        current_trial_id=(
            UUID(snapshot["current_trial_id"])
            if snapshot.get("current_trial_id")
            else None
        ),
        next_parameters=snapshot.get("next_parameters"),
        active_template_root=(
            Path(snapshot["active_template_root"])
            if snapshot.get("active_template_root")
            else None
        ),
        status=snapshot.get("status", "CREATED"),
        spec=(
            ExperimentSpec.model_validate(snapshot["spec"])
            if snapshot.get("spec")
            else None
        ),
        planner_metadata=snapshot.get("planner_metadata") or {},
        output_root=(
            Path(snapshot["output_root"]) if snapshot.get("output_root") else None
        ),
        report_path=snapshot.get("report_path"),
        ai_summary=snapshot.get("ai_summary"),
        ai_summary_metadata=snapshot.get("ai_summary_metadata") or {},
        outcomes=[
            TrialOutcome.model_validate(item) for item in snapshot.get("outcomes", [])
        ],
        events=[RunEvent.model_validate(item) for item in snapshot.get("events", [])],
        decisions=[
            DecisionRecord.model_validate(item)
            for item in snapshot.get("decisions", [])
        ],
        issues=[
            ValidationIssue.model_validate(item) for item in snapshot.get("issues", [])
        ],
    )
