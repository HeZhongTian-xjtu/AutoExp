from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable


class MLflowTracker:
    """Best-effort parent/child MLflow mirror; SQLite remains authoritative."""

    source = "mlflow"

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        artifact_root: Path | str | None = None,
        spool_root: Path | str | None = None,
    ):
        import mlflow

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite:///./workspaces/mlflow.db"
        )
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(
            experiment_name or os.getenv("MLFLOW_EXPERIMENT", "AutoExp")
        )
        self.artifact_root = Path(
            artifact_root
            or os.getenv("AUTOEXP_ARTIFACT_ROOT", "workspaces/autoexp-artifacts")
        ).resolve()
        self.spool_root = Path(
            spool_root or os.getenv("AUTOEXP_MLFLOW_SPOOL", "workspaces/mlflow-spool")
        ).resolve()
        self._parents: dict[str, str] = {}
        self._experiments: dict[str, str] = {}
        self.errors: list[str] = []

    @property
    def run_urls(self) -> dict[str, str]:
        base = os.getenv("MLFLOW_UI_URL")
        return (
            {
                key: f"{base.rstrip('/')}/#/experiments/{self._experiments.get(key, '0')}/runs/{value}"
                for key, value in self._parents.items()
            }
            if base
            else {}
        )

    def start_run(self, run: Any) -> None:
        key = str(run.run_id)
        if key in self._parents:
            return

        def start() -> None:
            active = self.mlflow.start_run(run_name=f"autoexp-{key}")
            self._parents[key] = active.info.run_id
            self._experiments[key] = active.info.experiment_id
            spec = run.spec
            self.mlflow.set_tags(
                {
                    "autoexp_run_id": key,
                    "template_id": spec.template_id if spec else "unknown",
                    "dataset_id": spec.dataset_id if spec else "unknown",
                    "planner_source": (run.planner_metadata or {}).get(
                        "source", "unknown"
                    ),
                    "code_version": _git_revision(),
                }
            )
            if spec:
                self.mlflow.log_params(
                    {
                        "template_id": spec.template_id,
                        "dataset_id": spec.dataset_id,
                        "model_id": spec.model_id,
                        "metric": spec.metric.name,
                        "seed": spec.seed,
                        "max_trials": spec.budget.max_trials,
                    }
                )
            self.mlflow.end_run()

        self._call("start_run", run, start)

    def log_trial(self, run: Any, outcome: Any) -> None:
        def log() -> None:
            self._ensure_parent(run)
            parent_id = self._parents[str(run.run_id)]
            with self.mlflow.start_run(
                run_name=f"trial-{len(run.outcomes):03d}",
                nested=False,
                tags={
                    "mlflow.parentRunId": parent_id,
                    "autoexp_trial_id": str(outcome.trial_id),
                    "status": outcome.status,
                },
            ):
                self.mlflow.log_params(
                    {key: str(value) for key, value in outcome.parameters.items()}
                )
                if outcome.metrics:
                    self.mlflow.log_metric("primary", outcome.metrics.primary.value)
                    for name, value in outcome.metrics.secondary.items():
                        if isinstance(value, (int, float)):
                            self.mlflow.log_metric(name, value)
                    if outcome.metrics.dataset_sha256:
                        self.mlflow.set_tag(
                            "dataset_sha256", outcome.metrics.dataset_sha256
                        )
                if outcome.validation:
                    for gate in outcome.validation.gates:
                        self.mlflow.set_tag(f"gate.{gate.name}", gate.status)

        self._call(
            "log_trial",
            run,
            log,
            {"trial_id": str(outcome.trial_id), "status": outcome.status},
        )

    def log_decision(self, run: Any, decision: Any) -> None:
        def log() -> None:
            self._ensure_parent(run)
            with self.mlflow.start_run(run_id=self._parents[str(run.run_id)]):
                self.mlflow.set_tag(
                    f"decision_{len(run.decisions)}", decision.decision.action
                )

        self._call("log_decision", run, log, {"action": decision.decision.action})

    def log_artifact(self, run: Any, artifact: Any) -> None:
        def log() -> None:
            self._ensure_parent(run)
            path = self.artifact_root / artifact.relative_path
            if path.is_file():
                with self.mlflow.start_run(run_id=self._parents[str(run.run_id)]):
                    self.mlflow.log_artifact(str(path))

        self._call("log_artifact", run, log, {"path": artifact.relative_path})

    def finish_run(self, run: Any) -> None:
        def finish() -> None:
            self._ensure_parent(run)
            with self.mlflow.start_run(run_id=self._parents[str(run.run_id)]):
                self.mlflow.set_tag("final_status", run.status)
                self.mlflow.log_metric("trial_count", len(run.outcomes))
                self.mlflow.log_metric(
                    "elapsed_seconds", float(getattr(run, "elapsed_seconds", 0.0))
                )
                if run.best and run.best.metrics:
                    self.mlflow.log_metric(
                        "best_primary", run.best.metrics.primary.value
                    )

        self._call("finish_run", run, finish)

    def replay_spool(self) -> int:
        """Return pending count; events stay durable until an operator reruns the source Run sync."""
        return (
            len(list(self.spool_root.glob("*.jsonl")))
            if self.spool_root.exists()
            else 0
        )

    def _ensure_parent(self, run: Any) -> None:
        if str(run.run_id) not in self._parents:
            self.start_run(run)
        if str(run.run_id) not in self._parents:
            raise RuntimeError("MLflow parent Run is unavailable")

    def _call(
        self,
        operation: str,
        run: Any,
        callback: Callable[[], None],
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            callback()
        except Exception as exc:
            self.errors.append(str(exc))
            self.spool_root.mkdir(parents=True, exist_ok=True)
            record = {
                "operation": operation,
                "run_id": str(run.run_id),
                "payload": payload or {},
                "error": str(exc),
            }
            with (self.spool_root / f"{run.run_id}.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_mlflow_tracker() -> MLflowTracker:
    return MLflowTracker()


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or "uncommitted"
        )
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"
