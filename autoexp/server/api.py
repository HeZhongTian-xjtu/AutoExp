from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .store import TaskStore


ROOT = Path(
    os.getenv("AUTOEXP_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
STORE = TaskStore(os.getenv("AUTOEXP_TASK_DB", ROOT / "workspaces" / "tasks.sqlite3"))
app = FastAPI(title="AutoExp API", version="1.0")


class RunRequest(BaseModel):
    objective: str = Field(min_length=3, max_length=2000)
    hypothesis: str = Field(
        default="The Agent will improve the registered metric.", max_length=2000
    )
    template_id: str = "housing-regression-v1"
    dataset_id: str | None = None
    planner_mode: str = "llm"
    executor_mode: str = "docker"
    tracker_mode: str = "mlflow"
    max_trials: int = Field(default=3, ge=1, le=8)
    seed: int = 42
    generate_summary: bool = False


def _authorize(request: Request) -> None:
    expected = os.getenv("AUTOEXP_API_TOKEN")
    if expected and request.headers.get("authorization") != f"Bearer {expected}":
        raise HTTPException(401, "invalid API token")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", status_code=202)
def create_run(body: RunRequest, request: Request) -> dict[str, str]:
    _authorize(request)
    if len(
        [item for item in STORE.list(100) if item["status"] in {"QUEUED", "RUNNING"}]
    ) >= int(os.getenv("AUTOEXP_MAX_CONCURRENT_RUNS", "2")):
        raise HTTPException(429, "concurrency limit reached")
    run_id = str(uuid4())
    payload = body.model_dump()
    STORE.create(run_id, payload)
    try:
        from redis import Redis
        from rq import Queue

        Queue(
            "autoexp",
            connection=Redis.from_url(
                os.getenv("AUTOEXP_REDIS_URL", "redis://localhost:6379/0")
            ),
        ).enqueue(
            "autoexp.server.tasks.execute_run",
            run_id,
            payload,
            job_timeout=int(os.getenv("AUTOEXP_JOB_TIMEOUT", "3600")),
            job_id=run_id,
        )
    except Exception as exc:
        STORE.update(run_id, "FAILED", {"error": f"queue unavailable: {exc}"})
        raise HTTPException(503, "task queue unavailable") from exc
    return {"run_id": run_id, "status": "QUEUED"}


@app.get("/api/runs")
def list_runs(request: Request):
    _authorize(request)
    return STORE.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request):
    _authorize(request)
    item = STORE.get(run_id)
    if item is None:
        raise HTTPException(404, "run not found")
    return item


@app.post("/api/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str, request: Request):
    _authorize(request)
    if not STORE.request_cancel(run_id):
        raise HTTPException(409, "run cannot be cancelled")
    try:
        from redis import Redis
        from rq.job import Job

        Job.fetch(
            run_id,
            connection=Redis.from_url(
                os.getenv("AUTOEXP_REDIS_URL", "redis://localhost:6379/0")
            ),
        ).cancel()
    except Exception:
        pass
    return {"run_id": run_id, "status": "CANCELLING"}


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request, after: int = 0):
    _authorize(request)
    if STORE.get(run_id) is None:
        raise HTTPException(404, "run not found")

    async def generate():
        cursor = after
        while not await request.is_disconnected():
            events = STORE.events(run_id, cursor)
            for event in events:
                cursor = event["sequence"]
                yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event)}\n\n"
            item = STORE.get(run_id)
            if item and item["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")
