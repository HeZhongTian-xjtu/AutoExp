from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    """Durable task/event control plane, separate from experiment facts."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
            CREATE TABLE IF NOT EXISTS tasks(
              run_id TEXT PRIMARY KEY, status TEXT NOT NULL, request_json TEXT NOT NULL,
              result_json TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_events(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              phase TEXT NOT NULL, message TEXT NOT NULL, payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_task_events_run ON task_events(run_id, sequence);
            """
            )

    def create(self, run_id: str, request: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO tasks(run_id,status,request_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (run_id, "QUEUED", json.dumps(request), now, now),
            )
        self.event(run_id, "QUEUED", "Run queued")

    def update(
        self, run_id: str, status: str, result: dict[str, Any] | None = None
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE tasks SET status=?, result_json=COALESCE(?,result_json), updated_at=? WHERE run_id=?",
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    _now(),
                    run_id,
                ),
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT run_id FROM tasks ORDER BY updated_at DESC LIMIT ?",
                (min(max(limit, 1), 100),),
            ).fetchall()
        return [item for row in rows if (item := self.get(row["run_id"]))]

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE tasks SET cancel_requested=1,status='CANCELLING',updated_at=? WHERE run_id=? AND status NOT IN ('COMPLETED','FAILED','CANCELLED')",
                (_now(), run_id),
            )
        if cursor.rowcount:
            self.event(run_id, "CANCELLING", "Cancellation requested")
        return bool(cursor.rowcount)

    def cancelled(self, run_id: str) -> bool:
        item = self.get(run_id)
        return bool(item and item["cancel_requested"])

    def event(
        self,
        run_id: str,
        phase: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO task_events(run_id,phase,message,payload_json,created_at) VALUES(?,?,?,?,?)",
                (run_id, phase, message, json.dumps(payload or {}), _now()),
            )
            return int(cursor.lastrowid)

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM task_events WHERE run_id=? AND sequence>? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "run_id": run_id,
                "phase": row["phase"],
                "message": row["message"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db
