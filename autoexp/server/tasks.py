from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import UUID

from autoexp.application import AutoExpApplicationService
from .store import TaskStore


ROOT = Path(
    os.getenv("AUTOEXP_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
TASK_DB = Path(os.getenv("AUTOEXP_TASK_DB", ROOT / "workspaces" / "tasks.sqlite3"))


def execute_run(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
    store = TaskStore(TASK_DB)
    if store.cancelled(run_id):
        store.update(run_id, "CANCELLED")
        return {"run_id": run_id, "status": "CANCELLED"}
    try:
        store.update(run_id, "RUNNING")
        store.event(run_id, "PLANNING", "Worker started AutoExp")
        service = AutoExpApplicationService(
            ROOT,
            template_id=request["template_id"],
            planner_mode=request.get("planner_mode", "llm"),
            action_planner_mode=request.get("planner_mode", "llm"),
            executor_mode=request.get("executor_mode", "docker"),
            tracker_mode=request.get("tracker_mode", "mlflow"),
            summary_mode="auto" if request.get("generate_summary") else "disabled",
        )
        run = service.run(
            request["objective"],
            request.get("hypothesis")
            or "The Agent will improve the registered metric.",
            request.get("max_trials", 3),
            seed=request.get("seed", 42),
            dataset_id=request.get("dataset_id"),
            run_id=UUID(run_id),
            output_root=ROOT / "workspaces" / "worker-runs",
        )
        summary = service.summarize(run)
        status = "CANCELLED" if store.cancelled(run_id) else run.status
        store.update(run_id, status, summary)
        store.event(
            run_id, status, f"Run {status.lower()}", {"trial_count": len(run.outcomes)}
        )
        return summary
    except Exception as exc:
        store.update(run_id, "FAILED", {"error": str(exc)})
        store.event(run_id, "FAILED", str(exc))
        raise
