from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from autoexp.domain import ExperimentRun, ExperimentSpec, TemplateManifest
from autoexp.application.datasets import DatasetCatalog, DatasetCatalogError
from autoexp.execution import Executor, build_executor
from autoexp.persistence import ArtifactStore, SQLiteRepository
from autoexp.planning import (
    ActionPlannerProtocol,
    DeterministicPlanner,
    PlannerError,
    PlannerProtocol,
    build_action_planner,
    build_planner,
)
from autoexp.reporting import ReportBuilder, RunSummaryGenerator
from autoexp.tracking import ExperimentTracker, build_tracker

from autoexp.graph.runner import LangGraphExperimentOrchestrator
from autoexp.application.templates import TemplateCatalog

from .trial_runner import TrialRunner


class AutoExpApplicationService:
    """Framework-independent application service shared by CLI and Streamlit."""

    def __init__(
        self,
        project_root: Path | str,
        template_id: str = "text-classification-v1",
        planner: PlannerProtocol | None = None,
        planner_mode: str | None = None,
        action_planner: ActionPlannerProtocol | None = None,
        action_planner_mode: str | None = None,
        database_path: Path | str | None = None,
        artifact_root: Path | str | None = None,
        summary_mode: str | None = None,
        summary_generator: RunSummaryGenerator | None = None,
        executor: Executor | None = None,
        executor_mode: str | None = None,
        executor_image: str | None = None,
        tracker: ExperimentTracker | None = None,
        tracker_mode: str | None = None,
        checkpoint_path: Path | str | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.template_root = self._find_template_root(template_id)
        self.base_manifest = TemplateManifest.load(self.template_root / "manifest.yaml")
        self.manifest = self.base_manifest
        if self.base_manifest.template_id != template_id:
            raise ValueError(f"unsupported AutoExp template: {template_id}")
        self.planner = planner
        self.planner_mode = planner_mode
        self.action_planner = action_planner
        self.action_planner_mode = action_planner_mode
        selected_database_path = database_path or os.getenv(
            "AUTOEXP_DB_PATH", self.project_root / "workspaces" / "autoexp.sqlite3"
        )
        self.repository = SQLiteRepository(selected_database_path)
        self.dataset_catalog = DatasetCatalog(self.project_root, self.repository)
        self.dataset_catalog.ensure_builtin(self.template_root, self.base_manifest)
        self.artifact_store = ArtifactStore(
            artifact_root
            or os.getenv(
                "AUTOEXP_ARTIFACT_ROOT",
                self.project_root / "workspaces" / "autoexp-artifacts",
            )
        )
        self.report_builder = ReportBuilder()
        selected_summary_mode = summary_mode or os.getenv(
            "AUTOEXP_SUMMARY_MODE", "deterministic"
        )
        self.summary_generator = (
            None
            if selected_summary_mode.strip().lower() == "disabled"
            else summary_generator or RunSummaryGenerator(mode=selected_summary_mode)
        )
        self.executor = executor or build_executor(executor_mode, executor_image)
        self.tracker = tracker or build_tracker(
            tracker_mode, str(self.artifact_store.root)
        )
        default_checkpoint = Path(selected_database_path).with_name(
            f"{Path(selected_database_path).stem}-langgraph.sqlite3"
        )
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("AUTOEXP_CHECKPOINT_DB_PATH", default_checkpoint)
        ).resolve()

    def build_spec(
        self,
        objective: str,
        hypothesis: str,
        max_trials: int,
        seed: int = 42,
        dataset_id: str | None = None,
    ) -> ExperimentSpec:
        manifest = (
            self._resolve_dataset(dataset_id)[1] if dataset_id else self.base_manifest
        )
        return (
            DeterministicPlanner()
            .plan(
                objective=objective,
                hypothesis=hypothesis,
                manifest=manifest,
                max_trials=max_trials,
                seed=seed,
            )
            .spec
        )

    def list_datasets(self) -> list[Any]:
        return self.dataset_catalog.list_for_template(
            self.template_root, self.base_manifest
        )

    def list_templates(self) -> list[Any]:
        return TemplateCatalog(self.project_root).list_templates()

    def register_dataset(
        self,
        uploads: list[tuple[str, bytes]],
        display_name: str | None = None,
    ) -> Any:
        return self.dataset_catalog.register_upload(
            self.template_root,
            self.base_manifest,
            uploads,
            display_name=display_name,
        )

    def run(
        self,
        objective: str,
        hypothesis: str,
        max_trials: int,
        output_root: Path | str | None = None,
        seed: int = 42,
        planner_mode: str | None = None,
        action_planner_mode: str | None = None,
        dataset_id: str | None = None,
        run_id: UUID | None = None,
        progress_path: Path | str | None = None,
    ) -> ExperimentRun:
        if not 1 <= max_trials <= 8:
            raise ValueError("max_trials must be between 1 and 8")
        run_root = Path(
            output_root or self.project_root / "workspaces" / "autoexp-runs"
        ).resolve()
        run_id = run_id or uuid4()
        dataset, manifest = self._resolve_dataset(dataset_id)
        self.repository.initialize_run(run_id, "PLANNING", run_root)

        selected_planner = self._select_planner(planner_mode)
        runner = TrialRunner(
            self.project_root,
            self.template_root,
            manifest,
            executor=self.executor,
            dataset=dataset,
            dataset_catalog=self.dataset_catalog,
        )
        orchestrator = LangGraphExperimentOrchestrator(
            runner,
            self.repository,
            self.artifact_store,
            self._select_action_planner(action_planner_mode or planner_mode),
            tracker=self.tracker,
            checkpoint_path=self.checkpoint_path,
            experiment_planner=selected_planner,
            progress_path=progress_path,
        )
        run = orchestrator.run(
            None,
            run_root,
            run_id=run_id,
            planning_request={
                "objective": objective,
                "hypothesis": hypothesis,
                "max_trials": max_trials,
                "seed": seed,
                "dataset_id": dataset.dataset_id,
            },
        )
        return self._finalize_report(run)

    def resume(
        self,
        run_id: UUID | str,
        command: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """Resume a persisted run without calling its initial Planner again.

        A waiting run stays paused until an explicit human command is supplied.
        """
        existing = self.repository.load_run(run_id)
        if existing is None:
            raise KeyError(f"run not found: {run_id}")
        if existing.spec is None:
            raise PlannerError("cannot resume a run without a persisted ExperimentSpec")
        dataset, manifest = self._resolve_dataset(existing.spec.dataset_id)
        output_root = (
            existing.output_root or self.project_root / "workspaces" / "autoexp-runs"
        )
        runner = TrialRunner(
            self.project_root,
            self.template_root,
            manifest,
            executor=self.executor,
            dataset=dataset,
            dataset_catalog=self.dataset_catalog,
        )
        orchestrator = LangGraphExperimentOrchestrator(
            runner,
            self.repository,
            self.artifact_store,
            self._select_action_planner(),
            tracker=self.tracker,
            checkpoint_path=self.checkpoint_path,
        )
        run = orchestrator.run(
            existing.spec,
            output_root,
            run_id=existing.run_id,
            initial_run=existing,
            planner_metadata=existing.planner_metadata,
            resume_command=command,
        )
        return self._finalize_report(run)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.repository.list_runs(limit)

    def load_run(self, run_id: UUID | str) -> ExperimentRun | None:
        return self.repository.load_run(run_id)

    def delete_all_runs(self) -> int:
        """Delete persisted AutoExp records and generated artifacts, not templates or input data."""
        deleted = self.repository.delete_all_runs()
        self.artifact_store.clear()
        return deleted

    def _select_planner(self, planner_mode: str | None) -> PlannerProtocol:
        if self.planner is not None and planner_mode is None:
            return self.planner
        return build_planner(planner_mode or self.planner_mode)

    def _select_action_planner(
        self, action_planner_mode: str | None = None
    ) -> ActionPlannerProtocol:
        if self.action_planner is not None and action_planner_mode is None:
            return self.action_planner
        return build_action_planner(
            action_planner_mode or self.action_planner_mode or self.planner_mode
        )

    @staticmethod
    def _failed_planner_metadata(
        planner: PlannerProtocol, exc: Exception
    ) -> dict[str, Any]:
        return {
            "source": (
                "llm"
                if planner.__class__.__name__ == "LLMStructuredPlanner"
                else "deterministic"
            ),
            "model": getattr(planner, "model", None),
            "attempts": getattr(planner, "max_attempts", 1),
            "error": str(exc),
        }

    def _finalize_report(self, run: ExperimentRun) -> ExperimentRun:
        if self.summary_generator is not None:
            summary_result = self.summary_generator.generate(run)
            run.ai_summary = summary_result.markdown
            run.ai_summary_metadata = summary_result.metadata.model_dump(mode="json")
            summary_record = self.artifact_store.put_text(
                run.ai_summary,
                run.run_id,
                "ai_summary",
                "run_summary.md",
            )
            self.repository.register_artifact(summary_record)
            try:
                self.tracker.log_artifact(run, summary_record)
            except Exception:
                pass
        records = self.repository.list_artifacts(run.run_id)
        report = self.report_builder.build(run, records)
        record = self.artifact_store.put_text(report, run.run_id, "report", "report.md")
        self.repository.register_artifact(record)
        try:
            self.tracker.log_artifact(run, record)
        except Exception:
            pass
        run.report_path = str(self.artifact_store.root / record.relative_path)
        self.repository.set_report_path(run.run_id, run.report_path)
        self.repository.save_run(run)
        try:
            self.tracker.finish_run(run)
        except Exception:
            pass
        return run

    def summarize(self, run: ExperimentRun) -> dict[str, Any]:
        best = run.best
        dataset_record = (
            self.repository.load_dataset(run.spec.dataset_id) if run.spec else None
        )
        report_markdown = ""
        if run.report_path and Path(run.report_path).is_file():
            report_markdown = Path(run.report_path).read_text(encoding="utf-8")
        return {
            "run_id": str(run.run_id),
            "status": run.status,
            "phase": run.phase,
            "elapsed_seconds": run.elapsed_seconds,
            "trial_count": len(run.outcomes),
            "best_trial_id": str(best.trial_id) if best else None,
            "best_metric": (
                best.metrics.primary.value if best and best.metrics else None
            ),
            "planner": run.planner_metadata,
            "spec": run.spec.model_dump(mode="json") if run.spec else None,
            "dataset": (
                dataset_record.model_dump(mode="json") if dataset_record else None
            ),
            "report_path": run.report_path,
            "report_markdown": report_markdown,
            "ai_summary": run.ai_summary,
            "ai_summary_metadata": run.ai_summary_metadata,
            "tracking": {
                "source": getattr(self.tracker, "source", "none"),
                "error": getattr(self.tracker, "reason", None),
                "errors": getattr(self.tracker, "errors", []),
                "run_url": getattr(self.tracker, "run_urls", {}).get(str(run.run_id)),
                "pending_spool": (
                    self.tracker.replay_spool()
                    if hasattr(self.tracker, "replay_spool")
                    else 0
                ),
            },
            "artifacts": [
                artifact.model_dump(mode="json")
                for artifact in self.repository.list_artifacts(run.run_id)
            ],
            "events": [event.model_dump(mode="json") for event in run.events],
            "issues": [issue.model_dump(mode="json") for issue in run.issues],
            "decisions": [
                {
                    "decision_id": str(record.decision_id),
                    "trial_id": str(record.trial_id) if record.trial_id else None,
                    "action": record.decision.action,
                    "reason": record.decision.reason,
                    "parameters": record.decision.parameters,
                    "search_space": record.decision.search_space,
                    "metadata": record.metadata,
                    "observation": record.observation.model_dump(mode="json"),
                    "created_at": record.created_at.isoformat(),
                }
                for record in run.decisions
            ],
            "trials": [
                {
                    "trial_id": str(outcome.trial_id),
                    "status": outcome.status,
                    "metric": (
                        outcome.metrics.primary.value if outcome.metrics else None
                    ),
                    "secondary_metrics": (
                        outcome.metrics.secondary if outcome.metrics else {}
                    ),
                    "metric_details": (
                        outcome.metrics.details if outcome.metrics else {}
                    ),
                    "dataset_sha256": (
                        outcome.metrics.dataset_sha256 if outcome.metrics else None
                    ),
                    "parameters": outcome.parameters,
                    "workspace": str(outcome.workspace),
                    "preflight_passed": (
                        outcome.preflight.passed if outcome.preflight else False
                    ),
                    "validation": (
                        outcome.validation.model_dump(mode="json")
                        if outcome.validation
                        else None
                    ),
                    "issues": [
                        issue.model_dump(mode="json") for issue in outcome.issues
                    ],
                }
                for outcome in run.outcomes
            ],
        }

    def _find_template_root(self, template_id: str) -> Path:
        return TemplateCatalog(self.project_root).get(template_id).root

    def _resolve_dataset(self, dataset_id: str | None) -> tuple[Any, TemplateManifest]:
        selected_id = dataset_id or self.base_manifest.dataset_id
        try:
            return self.dataset_catalog.get_for_template(
                self.template_root,
                self.base_manifest,
                selected_id,
            )
        except DatasetCatalogError:
            if dataset_id is None:
                builtin = self.dataset_catalog.ensure_builtin(
                    self.template_root, self.base_manifest
                )
                if builtin.status != "ready" or not builtin.files:
                    raise DatasetCatalogError(
                        "No dataset is available for this template. Upload a compatible train.csv first."
                    )
                return builtin, self.dataset_catalog.bind_manifest(
                    self.base_manifest, builtin
                )
            raise
