from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from autoexp.persistence.artifacts import ArtifactRecord


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    spec_json TEXT,
    planner_json TEXT NOT NULL DEFAULT '{}',
    output_root TEXT NOT NULL,
    report_path TEXT,
    ai_summary TEXT,
    ai_summary_metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT,
    name TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT,
    kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, relative_path)
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT,
    observation_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_state (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_run_id ON trials(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_datasets_updated_at ON datasets(updated_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


class SQLiteRepository:
    """Small repository for run state; every operation uses a short-lived connection."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "ai_summary" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN ai_summary TEXT")
            if "ai_summary_metadata_json" not in columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN ai_summary_metadata_json TEXT"
                )

    def initialize_run(self, run_id: UUID, status: str, output_root: Path) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, status, output_root, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at
                """,
                (str(run_id), status, str(output_root), now, now),
            )

    def save_run(self, run: Any) -> None:
        now = _now()
        spec_json = run.spec.model_dump(mode="json") if run.spec is not None else None
        planner_json = (
            run.planner_metadata.model_dump(mode="json")
            if hasattr(run.planner_metadata, "model_dump")
            else run.planner_metadata
        )
        output_root = str(run.output_root) if run.output_root else ""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, status, spec_json, planner_json, output_root, report_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    spec_json=excluded.spec_json,
                    planner_json=excluded.planner_json,
                    output_root=excluded.output_root,
                    report_path=excluded.report_path,
                    updated_at=excluded.updated_at
                """,
                (
                    str(run.run_id),
                    run.status,
                    _json(spec_json) if spec_json is not None else None,
                    _json(planner_json or {}),
                    output_root,
                    run.report_path,
                    now,
                    now,
                ),
            )

        self._save_run_summary(run)
        self.save_state(run)

    def _save_run_summary(self, run: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET ai_summary=?, ai_summary_metadata_json=? WHERE run_id=?",
                (
                    getattr(run, "ai_summary", None),
                    _json(getattr(run, "ai_summary_metadata", {}) or {}),
                    str(run.run_id),
                ),
            )

    def save_state(self, run: Any) -> None:
        state = {
            "phase": getattr(run, "phase", "CREATED"),
            "started_at": (
                run.started_at.isoformat() if getattr(run, "started_at", None) else None
            ),
            "elapsed_seconds": float(getattr(run, "elapsed_seconds", 0.0)),
            "current_trial_id": (
                str(run.current_trial_id)
                if getattr(run, "current_trial_id", None)
                else None
            ),
            "next_parameters": getattr(run, "next_parameters", None),
            "active_template_root": (
                str(run.active_template_root)
                if getattr(run, "active_template_root", None)
                else None
            ),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_state(run_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (str(run.run_id), _json(state), _now()),
            )

    def save_trial(self, run_id: UUID, ordinal: int, outcome: Any) -> None:
        now = _now()
        outcome_json = outcome.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trials(trial_id, run_id, ordinal, status, parameters_json, outcome_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trial_id) DO UPDATE SET
                    status=excluded.status,
                    parameters_json=excluded.parameters_json,
                    outcome_json=excluded.outcome_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(outcome.trial_id),
                    str(run_id),
                    ordinal,
                    outcome.status,
                    _json(outcome.parameters),
                    _json(outcome_json),
                    now,
                    now,
                ),
            )

    def append_event(self, run_id: UUID, event: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, trial_id, name, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(event.trial_id) if event.trial_id else None,
                    event.name,
                    event.message,
                    _json(event.payload),
                    event.created_at.isoformat(),
                ),
            )

    def save_decision(self, decision: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO decisions(decision_id, run_id, trial_id, observation_json, decision_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    observation_json=excluded.observation_json,
                    decision_json=excluded.decision_json,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                (
                    str(decision.decision_id),
                    str(decision.run_id),
                    str(decision.trial_id) if decision.trial_id else None,
                    _json(decision.observation.model_dump(mode="json")),
                    _json(decision.decision.model_dump(mode="json")),
                    _json(decision.metadata),
                    decision.created_at.isoformat(),
                ),
            )

    def register_artifact(self, artifact: ArtifactRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(artifact_id, run_id, trial_id, kind, relative_path, sha256, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, relative_path) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    trial_id=excluded.trial_id,
                    kind=excluded.kind,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    created_at=excluded.created_at
                """,
                (
                    str(artifact.artifact_id),
                    str(artifact.run_id),
                    str(artifact.trial_id) if artifact.trial_id else None,
                    artifact.kind,
                    artifact.relative_path,
                    artifact.sha256,
                    artifact.size_bytes,
                    _now(),
                ),
            )

    def register_dataset(self, dataset: Any) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets(dataset_id, record_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    record_json=excluded.record_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(dataset.dataset_id),
                    _json(dataset.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def load_dataset(self, dataset_id: str) -> Any | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM datasets WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        from autoexp.domain import DatasetRecord

        return DatasetRecord.model_validate(json.loads(row["record_json"]))

    def list_datasets(self) -> list[Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM datasets ORDER BY updated_at DESC, dataset_id"
            ).fetchall()
        from autoexp.domain import DatasetRecord

        return [
            DatasetRecord.model_validate(json.loads(row["record_json"])) for row in rows
        ]

    def list_artifacts(self, run_id: UUID | str) -> list[ArtifactRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id=? ORDER BY artifact_id",
                (str(run_id),),
            ).fetchall()
        return [
            ArtifactRecord(
                artifact_id=UUID(row["artifact_id"]),
                run_id=UUID(row["run_id"]),
                trial_id=UUID(row["trial_id"]) if row["trial_id"] else None,
                kind=row["kind"],
                relative_path=row["relative_path"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
            )
            for row in rows
        ]

    def load_run(self, run_id: UUID | str) -> Any | None:
        with self._connect() as connection:
            run_row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (str(run_id),)
            ).fetchone()
            if run_row is None:
                return None
            trial_rows = connection.execute(
                "SELECT outcome_json FROM trials WHERE run_id=? ORDER BY ordinal",
                (str(run_id),),
            ).fetchall()
            decision_rows = connection.execute(
                "SELECT * FROM decisions WHERE run_id=? ORDER BY created_at, decision_id",
                (str(run_id),),
            ).fetchall()
            state_row = connection.execute(
                "SELECT state_json FROM run_state WHERE run_id=?",
                (str(run_id),),
            ).fetchone()
            event_rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY event_id",
                (str(run_id),),
            ).fetchall()

        from autoexp.domain import (
            DecisionRecord,
            ExperimentRun,
            ExperimentSpec,
            RunEvent,
            TrialOutcome,
        )

        state = json.loads(state_row["state_json"]) if state_row else {}

        spec = (
            ExperimentSpec.model_validate(json.loads(run_row["spec_json"]))
            if run_row["spec_json"]
            else None
        )
        planner = json.loads(run_row["planner_json"] or "{}")
        events = [
            RunEvent(
                name=row["name"],
                message=row["message"],
                trial_id=UUID(row["trial_id"]) if row["trial_id"] else None,
                payload=json.loads(row["payload_json"] or "{}"),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in event_rows
        ]
        outcomes = [
            TrialOutcome.model_validate(json.loads(row["outcome_json"]))
            for row in trial_rows
        ]
        decisions = [
            DecisionRecord(
                decision_id=UUID(row["decision_id"]),
                run_id=UUID(row["run_id"]),
                trial_id=UUID(row["trial_id"]) if row["trial_id"] else None,
                observation=json.loads(row["observation_json"]),
                decision=json.loads(row["decision_json"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in decision_rows
        ]
        return ExperimentRun(
            run_id=UUID(run_row["run_id"]),
            status=run_row["status"],
            spec=spec,
            ai_summary=run_row["ai_summary"],
            ai_summary_metadata=json.loads(run_row["ai_summary_metadata_json"] or "{}"),
            planner_metadata=planner,
            output_root=(
                Path(run_row["output_root"]) if run_row["output_root"] else None
            ),
            report_path=run_row["report_path"],
            outcomes=outcomes,
            decisions=decisions,
            phase=state.get("phase", run_row["status"]),
            started_at=(
                datetime.fromisoformat(state["started_at"])
                if state.get("started_at")
                else None
            ),
            elapsed_seconds=float(state.get("elapsed_seconds", 0.0)),
            current_trial_id=(
                UUID(state["current_trial_id"])
                if state.get("current_trial_id")
                else None
            ),
            next_parameters=state.get("next_parameters"),
            active_template_root=(
                Path(state["active_template_root"])
                if state.get("active_template_root")
                else None
            ),
            events=events,
        )

    def delete_all_runs(self) -> int:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            connection.execute("DELETE FROM runs")
        return int(count)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, status, spec_json, planner_json, report_path, ai_summary, created_at, updated_at "
                "FROM runs ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
            dataset_rows = connection.execute(
                "SELECT dataset_id, record_json FROM datasets"
            ).fetchall()
        datasets = {
            row["dataset_id"]: json.loads(row["record_json"] or "{}")
            for row in dataset_rows
        }
        history: list[dict[str, Any]] = []
        for row in rows:
            spec = json.loads(row["spec_json"] or "{}")
            dataset_id = spec.get("dataset_id")
            dataset = datasets.get(dataset_id, {})
            history.append(
                {
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "planner": json.loads(row["planner_json"] or "{}"),
                    "report_path": row["report_path"],
                    "has_ai_summary": bool(row["ai_summary"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "objective": spec.get("objective"),
                    "template_id": spec.get("template_id"),
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.get("display_name") or dataset_id,
                }
            )
        return history

    def set_report_path(self, run_id: UUID, report_path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET report_path=?, updated_at=? WHERE run_id=?",
                (report_path, _now(), str(run_id)),
            )
