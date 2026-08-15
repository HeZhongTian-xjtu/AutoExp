from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml

from autoexp.domain import DatasetRecord, TrialOutcome
from autoexp.domain.errors import ValidationIssue
from autoexp.domain.models import ExperimentSpec, TemplateManifest
from autoexp.application.datasets import DatasetCatalog
from autoexp.domain.policies import validate_parameters
from autoexp.evaluation import verify_dataset_manifest
from autoexp.evaluation.metrics import load_metrics
from autoexp.execution import (
    ExecutionRequest,
    Executor,
    build_executor,
)
from autoexp.preflight import PreflightPipeline
from autoexp.repair import classify_failure
from autoexp.validation import ValidationPipeline


class TrialRunner:
    def __init__(
        self,
        project_root: Path,
        template_root: Path,
        manifest: TemplateManifest,
        executor: Executor | None = None,
        dataset: DatasetRecord | None = None,
        dataset_catalog: DatasetCatalog | None = None,
    ):
        self.project_root = project_root.resolve()
        self.template_root = template_root.resolve()
        self.manifest = manifest
        self.executor = executor or build_executor()
        self.dataset = dataset
        self.dataset_catalog = dataset_catalog
        self.preflight = PreflightPipeline(self.project_root / "requirements.txt")
        self.validation = ValidationPipeline(self.executor, self.manifest)

    def run(
        self,
        spec: ExperimentSpec,
        parameters: dict[str, Any],
        output_root: Path,
        run_id: UUID | None = None,
        trial_id: UUID | None = None,
        timeout_seconds: int | None = None,
        trial_index: int | None = None,
    ) -> TrialOutcome:
        trial_id = trial_id or uuid4()
        issues = validate_parameters(parameters, self.manifest)
        if issues:
            return TrialOutcome(
                trial_id=trial_id,
                parameters=parameters,
                status="preflight_failed",
                workspace=output_root / str(trial_id),
                issues=issues,
            )

        workspace = output_root.resolve() / str(trial_id)
        if workspace.exists():
            raise FileExistsError(f"trial workspace already exists: {workspace}")
        shutil.copytree(self.template_root, workspace)
        if self.dataset is not None and self.dataset_catalog is not None:
            self.dataset_catalog.stage(self.dataset, workspace)
        total_steps = 1
        self._write_progress(
            workspace,
            phase="preparing",
            current=0,
            total=total_steps,
            trial_index=trial_index,
            total_trials=spec.budget.max_trials,
            message="Preparing the trial workspace",
        )
        dataset_integrity = verify_dataset_manifest(workspace, self.manifest)
        if not dataset_integrity.passed:
            self._write_progress(
                workspace,
                "failed",
                total_steps,
                total_steps,
                trial_index,
                spec.budget.max_trials,
                "Dataset validation failed",
                status="failed",
            )
            return TrialOutcome(
                trial_id=trial_id,
                parameters=parameters,
                status="preflight_failed",
                workspace=workspace,
                dataset_integrity=dataset_integrity,
                issues=dataset_integrity.issues,
            )
        config_path = workspace / "configs" / "experiment.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = dict(parameters)
        config["seed"] = spec.seed
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

        code = (workspace / self.manifest.entrypoint).read_text(encoding="utf-8")
        preflight = self.preflight.run(code, self.manifest)
        if not preflight.passed:
            self._write_progress(
                workspace,
                "preflight",
                0,
                total_steps,
                trial_index,
                spec.budget.max_trials,
                "AST preflight failed",
                status="failed",
            )
            return TrialOutcome(
                trial_id=trial_id,
                parameters=parameters,
                status="preflight_failed",
                workspace=workspace,
                dataset_integrity=dataset_integrity,
                preflight=preflight,
                issues=preflight.issues,
            )

        self._write_progress(
            workspace,
            "validation",
            0,
            total_steps,
            trial_index,
            spec.budget.max_trials,
            "Running Pytest and Smoke gates",
        )
        validation = self.validation.run(
            workspace=workspace,
            run_id=run_id or uuid4(),
            trial_id=trial_id,
            timeout_seconds=int(
                timeout_seconds
                or self.manifest.resource_policy.get(
                    "timeout_seconds", spec.budget.timeout_seconds
                )
            ),
        )
        if not validation.passed:
            self._write_progress(
                workspace,
                "validation",
                total_steps,
                total_steps,
                trial_index,
                spec.budget.max_trials,
                "Validation gate failed",
                status="failed",
            )
            return TrialOutcome(
                trial_id=trial_id,
                parameters=parameters,
                status="validation_failed",
                workspace=workspace,
                dataset_integrity=dataset_integrity,
                preflight=preflight,
                validation=validation,
                issues=validation.issues(),
            )

        effective_run_id = run_id or uuid4()
        request = ExecutionRequest(
            environment={
                "PYTHONHASHSEED": "0",
                "AUTOEXP_RUN_ID": str(effective_run_id),
                "AUTOEXP_TRIAL_ID": str(trial_id),
                "AUTOEXP_PROGRESS_PATH": (
                    str(workspace / "working" / "progress.json")
                    if getattr(self.executor, "is_local", False)
                    else "/workspace/working/progress.json"
                ),
            },
            run_id=effective_run_id,
            trial_id=trial_id,
            workspace=workspace,
            command=[sys.executable, self.manifest.entrypoint],
            timeout_seconds=int(
                timeout_seconds
                or self.manifest.resource_policy.get(
                    "timeout_seconds", spec.budget.timeout_seconds
                )
            ),
            resource_limits={
                "cpu_count": int(self.manifest.resource_policy.get("cpu_count", 2)),
                "memory_mb": int(self.manifest.resource_policy.get("memory_mb", 2048)),
                "pids_limit": int(self.manifest.resource_policy.get("pids_limit", 128)),
            },
            network_policy="none",
            immutable_paths=list(self.manifest.immutable_files),
        )
        self._write_progress(
            workspace,
            "training",
            0,
            total_steps,
            trial_index,
            spec.budget.max_trials,
            "Training model",
        )
        execution = self.executor.execute(request)
        if execution.status != "succeeded":
            self._write_progress(
                workspace,
                "training",
                total_steps,
                total_steps,
                trial_index,
                spec.budget.max_trials,
                "Training failed",
                status="failed",
            )
            output_text = ""
            for path in (execution.stdout_path, execution.stderr_path):
                if path.is_file():
                    output_text += path.read_text(encoding="utf-8", errors="replace")[
                        -4_000:
                    ]
            issue = ValidationIssue(
                code=classify_failure(
                    output_text, timeout=execution.status == "timeout"
                ),
                phase="execution",
                message=(
                    execution.error.message
                    if execution.error
                    else "experiment process failed"
                ),
                suggestion="Inspect the captured stderr and classify the failure.",
            )
            return TrialOutcome(
                trial_id=trial_id,
                parameters=parameters,
                status="execution_failed",
                workspace=workspace,
                dataset_integrity=dataset_integrity,
                preflight=preflight,
                validation=validation,
                execution=execution,
                issues=[issue],
            )

        evaluation_execution = None
        if self.manifest.evaluator_entrypoint:
            self._write_progress(
                workspace,
                "evaluation",
                total_steps,
                total_steps,
                trial_index,
                spec.budget.max_trials,
                "Running fixed evaluator",
            )
            evaluation_request = request.model_copy(
                update={
                    "command": [sys.executable, self.manifest.evaluator_entrypoint],
                    "output_subdir": "working/evaluator",
                }
            )
            evaluation_execution = self.executor.execute(evaluation_request)
            if evaluation_execution.status != "succeeded":
                self._write_progress(
                    workspace,
                    "evaluation",
                    total_steps,
                    total_steps,
                    trial_index,
                    spec.budget.max_trials,
                    "Evaluator failed",
                    status="failed",
                )
                output_text = ""
                for path in (
                    evaluation_execution.stdout_path,
                    evaluation_execution.stderr_path,
                ):
                    if path.is_file():
                        output_text += path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-4_000:]
                issue = ValidationIssue(
                    code="EVALUATOR_FAILED",
                    phase="evaluation",
                    message=(
                        evaluation_execution.error.message
                        if evaluation_execution.error
                        else "fixed evaluator failed"
                    ),
                    suggestion="Inspect the evaluator logs; the training result is not accepted without independent evaluation.",
                    details={
                        "error_code": (
                            evaluation_execution.error.code
                            if evaluation_execution.error
                            else "UNKNOWN"
                        ),
                        "output_tail": output_text[-2_000:],
                    },
                )
                return TrialOutcome(
                    trial_id=trial_id,
                    parameters=parameters,
                    status="execution_failed",
                    workspace=workspace,
                    dataset_integrity=dataset_integrity,
                    preflight=preflight,
                    validation=validation,
                    execution=execution,
                    evaluation_execution=evaluation_execution,
                    issues=[issue],
                )

        metrics_path = workspace / self.manifest.metric_file
        metrics, metric_issues = load_metrics(
            metrics_path,
            spec.metric.name,
            spec.metric.direction,
            parameters,
            expected_dataset_id=(
                dataset_integrity.dataset_id if dataset_integrity else None
            ),
            expected_dataset_sha256=(
                dataset_integrity.dataset_sha256 if dataset_integrity else None
            ),
            expected_seed=spec.seed,
            expected_trial_id=trial_id if self.manifest.evaluator_entrypoint else None,
        )
        self._write_progress(
            workspace,
            "completed",
            total_steps,
            total_steps,
            trial_index,
            spec.budget.max_trials,
            "Trial completed" if metrics else "Metric validation failed",
            status="succeeded" if metrics else "failed",
        )
        return TrialOutcome(
            trial_id=trial_id,
            parameters=parameters,
            status="succeeded" if metrics else "metric_failed",
            workspace=workspace,
            dataset_integrity=dataset_integrity,
            preflight=preflight,
            validation=validation,
            execution=execution,
            evaluation_execution=evaluation_execution,
            metrics=metrics,
            issues=metric_issues,
        )

    @staticmethod
    def _write_progress(
        workspace: Path,
        phase: str,
        current: int,
        total: int,
        trial_index: int | None,
        total_trials: int,
        message: str,
        status: str = "running",
    ) -> None:
        path = workspace / "working" / "progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "phase": phase,
            "current": max(0, int(current)),
            "total": max(1, int(total)),
            "unit": "trial",
            "trial_index": trial_index,
            "total_trials": total_trials,
            "message": message,
        }
        temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            for attempt in range(8):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == 7:
                        return
                    time.sleep(0.05 * (attempt + 1))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def recover_existing(
        self,
        spec: ExperimentSpec,
        parameters: dict[str, Any],
        workspace: Path,
        trial_id: UUID,
    ) -> TrialOutcome:
        """Recover a checkpointed workspace without launching duplicate code."""
        code_path = workspace / self.manifest.entrypoint
        preflight = None
        validation = None
        issues: list[ValidationIssue] = []
        dataset_integrity = verify_dataset_manifest(workspace, self.manifest)
        issues.extend(dataset_integrity.issues)
        if code_path.is_file():
            preflight = self.preflight.run(
                code_path.read_text(encoding="utf-8"), self.manifest
            )
            issues.extend(preflight.issues)
        else:
            issues.append(
                ValidationIssue(
                    code="RECOVERY_ENTRYPOINT_MISSING",
                    phase="recovery",
                    message="checkpointed Trial workspace has no entrypoint",
                    suggestion="Review the workspace before resuming.",
                )
            )
        metrics, metric_issues = load_metrics(
            workspace / self.manifest.metric_file,
            spec.metric.name,
            spec.metric.direction,
            parameters,
            expected_dataset_id=(
                dataset_integrity.dataset_id if dataset_integrity else None
            ),
            expected_dataset_sha256=(
                dataset_integrity.dataset_sha256 if dataset_integrity else None
            ),
            expected_seed=spec.seed,
            expected_trial_id=trial_id if self.manifest.evaluator_entrypoint else None,
        )
        issues.extend(metric_issues)
        return TrialOutcome(
            trial_id=trial_id,
            parameters=parameters,
            status="succeeded" if metrics and not issues else "metric_failed",
            workspace=workspace,
            dataset_integrity=dataset_integrity,
            preflight=preflight,
            validation=validation,
            execution=None,
            metrics=metrics if not issues else None,
            issues=issues,
        )
