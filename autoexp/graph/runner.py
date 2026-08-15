from __future__ import annotations

import shutil
import sqlite3
import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

import yaml

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from autoexp.domain import (
    ActionDecision,
    DecisionRecord,
    ExperimentObservation,
    ExperimentSpec,
    ParameterRange,
)
from autoexp.domain.errors import ValidationIssue
from autoexp.domain.policies import validate_experiment_spec, validate_parameters
from autoexp.planning import (
    ActionPlanResult,
    DeterministicActionPlanner,
    PlanResult,
    PlannerError,
    PlannerProtocol,
)
from autoexp.repair import RepairError, apply_unified_patch, validate_repaired_template

from autoexp.application.observation_builder import build_observation
from autoexp.application.orchestrator import ExperimentOrchestrator
from autoexp.domain import ExperimentRun
from autoexp.application.runtime import RunRuntime
from autoexp.application.trial_runner import TrialRunner

from .state import (
    AutoExpGraphState,
    GRAPH_SCHEMA_VERSION,
    run_to_snapshot,
    snapshot_to_run,
)


def _write_progress_file(path: Path, payload: dict[str, Any]) -> None:
    """Best-effort Windows-safe progress publication.

    Streamlit polls the destination while the graph updates it. A locked
    progress file must never turn an otherwise successful experiment into a
    failed Run, so replacement is retried and then skipped if the UI still
    holds the file briefly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class LangGraphExperimentOrchestrator(ExperimentOrchestrator):
    """LangGraph-backed replacement for the legacy orchestration loop.

    Existing domain services remain responsible for execution and persistence;
    this class owns graph state, routing, checkpointing and human interrupts.
    """

    def __init__(
        self,
        runner: TrialRunner,
        repository: Any | None = None,
        artifact_store: Any | None = None,
        action_planner: Any | None = None,
        tracker: Any | None = None,
        *,
        checkpoint_path: Path | str | None = None,
        experiment_planner: PlannerProtocol | None = None,
        progress_path: Path | str | None = None,
    ):
        self.runner = runner
        self.repository = repository
        self.artifact_store = artifact_store
        self.action_planner = action_planner or DeterministicActionPlanner()
        self.tracker = tracker
        self.runtime = RunRuntime(runner, repository, artifact_store, tracker)
        default_path = (
            self.runner.project_root / "workspaces" / "autoexp-langgraph.sqlite3"
        )
        self.checkpoint_path = Path(checkpoint_path or default_path).resolve()
        self.experiment_planner = experiment_planner
        self.progress_path = Path(progress_path).resolve() if progress_path else None
        self._progress_total_trials = 1
        self._invocation_clock_started: float | None = None
        self._invocation_elapsed_before = 0.0

    def run(
        self,
        spec: ExperimentSpec | None,
        output_root: Path,
        run_id: UUID | None = None,
        initial_run: ExperimentRun | None = None,
        planner_metadata: dict[str, Any] | None = None,
        resume_command: dict[str, Any] | None = None,
        planning_request: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        result = initial_run or ExperimentRun(run_id=run_id or uuid4())
        if spec is not None:
            result.spec = spec
        result.output_root = output_root.resolve()
        self._progress_total_trials = int(
            (planning_request or {}).get("max_trials")
            or (spec.budget.max_trials if spec is not None else 1)
        )
        if planner_metadata is not None:
            result.planner_metadata = planner_metadata
        if result.status == "WAITING_FOR_HUMAN" and resume_command is None:
            # A plain resume must not bypass a persisted human decision.
            return result
        self._invocation_clock_started = time.perf_counter()
        self._invocation_elapsed_before = result.elapsed_seconds
        initial_state = self._initial_state(result, spec, planning_request)
        config = {"configurable": {"thread_id": str(result.run_id)}}
        self._fallback_checkpoint_path = (
            result.output_root / ".langgraph-checkpoint.sqlite3"
        )
        try:
            with self._checkpointer() as checkpointer:
                graph = self._build_graph(checkpointer)
                try:
                    if resume_command is not None:
                        final_state = graph.invoke(
                            Command(resume=resume_command), config=config
                        )
                    else:
                        final_state = graph.invoke(initial_state, config=config)
                except GraphInterrupt:
                    persisted = (
                        self.repository.load_run(result.run_id)
                        if self.repository is not None
                        else None
                    )
                    return persisted or result
            return snapshot_to_run(final_state["run"])
        finally:
            self._invocation_clock_started = None

    def stream(
        self,
        spec: ExperimentSpec,
        output_root: Path,
        run_id: UUID | None = None,
        planner_metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield LangGraph node updates for future SSE/Web event adapters."""

        result = ExperimentRun(
            run_id=run_id or uuid4(), spec=spec, output_root=output_root.resolve()
        )
        if planner_metadata is not None:
            result.planner_metadata = planner_metadata
        config = {"configurable": {"thread_id": str(result.run_id)}}
        with self._checkpointer() as checkpointer:
            graph = self._build_graph(checkpointer)
            yield from graph.stream(
                self._initial_state(result, spec, None),
                config=config,
                stream_mode="updates",
            )

    def _refresh_elapsed(self, result: ExperimentRun) -> None:
        """Persist active invocation time without counting human-review pauses."""
        if self._invocation_clock_started is None:
            return
        active = max(0.0, time.perf_counter() - self._invocation_clock_started)
        result.elapsed_seconds = max(
            result.elapsed_seconds,
            self._invocation_elapsed_before + active,
        )

    @contextmanager
    def _checkpointer(self) -> Iterator[SqliteSaver]:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with SqliteSaver.from_conn_string(
                str(self.checkpoint_path)
            ) as checkpointer:
                checkpointer.setup()
                yield checkpointer
        except sqlite3.OperationalError:
            fallback = getattr(self, "_fallback_checkpoint_path", self.checkpoint_path)
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = fallback
            with SqliteSaver.from_conn_string(str(fallback)) as checkpointer:
                checkpointer.setup()
                yield checkpointer

    def _build_graph(self, checkpointer: SqliteSaver):
        graph = StateGraph(AutoExpGraphState)
        graph.add_node("plan_experiment", self._plan_experiment)
        graph.add_node("initialize_run", self._initialize_run)
        graph.add_node("validate_plan", self._validate_plan)
        graph.add_node("budget_guard", self._budget_guard)
        graph.add_node("select_parameters", self._select_parameters)
        graph.add_node("prepare_trial", self._prepare_trial)
        graph.add_node("execute_trial", self._execute_trial)
        graph.add_node("persist_trial", self._persist_trial)
        graph.add_node("build_observation", self._build_observation)
        graph.add_node("plan_action", self._plan_action)
        graph.add_node("persist_decision", self._persist_decision)
        graph.add_node("validate_action", self._validate_action)
        graph.add_node("update_search_space", self._update_search_space)
        graph.add_node("repair_validate", self._repair_validate)
        graph.add_node("repair_apply", self._repair_apply)
        graph.add_node("human_review", self._human_review)
        graph.add_node("apply_human_command", self._apply_human_command)
        graph.add_node("finalize_run", self._finalize_run)

        graph.add_edge(START, "plan_experiment")
        graph.add_conditional_edges(
            "plan_experiment",
            self._route,
            {
                "initialize_run": "initialize_run",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_edge("initialize_run", "validate_plan")
        graph.add_conditional_edges(
            "validate_plan",
            self._route,
            {
                "budget_guard": "budget_guard",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_conditional_edges(
            "budget_guard",
            self._route,
            {
                "select_parameters": "select_parameters",
                "prepare_trial": "prepare_trial",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_conditional_edges(
            "select_parameters",
            self._route,
            {
                "prepare_trial": "prepare_trial",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_edge("prepare_trial", "execute_trial")
        graph.add_edge("execute_trial", "persist_trial")
        graph.add_edge("persist_trial", "build_observation")
        graph.add_edge("build_observation", "plan_action")
        graph.add_edge("plan_action", "persist_decision")
        graph.add_edge("persist_decision", "validate_action")
        graph.add_conditional_edges(
            "validate_action",
            self._route,
            {
                "budget_guard": "budget_guard",
                "update_search_space": "update_search_space",
                "repair_validate": "repair_validate",
                "human_review": "human_review",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_edge("update_search_space", "budget_guard")
        graph.add_conditional_edges(
            "repair_validate",
            self._route,
            {
                "repair_apply": "repair_apply",
                "human_review": "human_review",
            },
        )
        graph.add_conditional_edges(
            "repair_apply",
            self._route,
            {
                "budget_guard": "budget_guard",
                "plan_action": "plan_action",
                "human_review": "human_review",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_edge("human_review", "apply_human_command")
        graph.add_conditional_edges(
            "apply_human_command",
            self._route,
            {
                "repair_apply": "repair_apply",
                "budget_guard": "budget_guard",
                "finalize_run": "finalize_run",
            },
        )
        graph.add_edge("finalize_run", END)
        return graph.compile(checkpointer=checkpointer)

    def _initial_state(
        self,
        result: ExperimentRun,
        spec: ExperimentSpec | None,
        planning_request: dict[str, Any] | None,
    ) -> AutoExpGraphState:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "run": run_to_snapshot(result),
            "spec": spec.model_dump(mode="json") if spec else None,
            "planning_request": planning_request,
            "output_root": str(result.output_root),
            "route": "initialize_run",
            "candidates": [],
            "completed_keys": [],
            "best_value": None,
            "stale_trials": 0,
            "current_observation": None,
            "pending_action": None,
            "current_trial_id": None,
            "current_parameters": None,
            "narrowed_space": {},
            "retry_current": False,
            "human_command": None,
            "failure_fingerprint": None,
            "failure_context_ref": None,
            "code_context": None,
            "code_context_ref": None,
            "repair_attempts": [],
            "pending_human_review": None,
            "operation_keys": [],
            "graph_events": [],
        }

    def _rehydrate(
        self, state: AutoExpGraphState
    ) -> tuple[ExperimentRun, ExperimentSpec]:
        result = snapshot_to_run(state["run"])
        raw_spec = state.get("spec")
        if raw_spec is None and result.spec is not None:
            raw_spec = result.spec.model_dump(mode="json")
        if raw_spec is None:
            raise PlannerError("graph state has no ExperimentSpec after planning")
        spec = ExperimentSpec.model_validate(raw_spec)
        result.spec = spec
        result.output_root = Path(state["output_root"])
        if result.active_template_root and result.active_template_root.is_dir():
            self.runner = TrialRunner(
                self.runner.project_root,
                result.active_template_root,
                self.runner.manifest,
                executor=self.runner.executor,
            )
            self.runtime.runner = self.runner
        return result, spec

    def _plan_experiment(self, state: AutoExpGraphState) -> dict[str, Any]:
        """Generate and validate the ExperimentSpec inside the graph boundary."""
        result = snapshot_to_run(state["run"])
        result.output_root = Path(state["output_root"]).resolve()
        raw_spec = state.get("spec") or (
            result.spec.model_dump(mode="json") if result.spec else None
        )
        if raw_spec is not None:
            return self._update(result, "initialize_run", spec=raw_spec)

        request = state.get("planning_request") or {}
        if self.experiment_planner is None:
            issue = ValidationIssue(
                code="PLAN_GENERATOR_MISSING",
                phase="planning",
                message="the LangGraph plan node has no configured Planner",
                suggestion="Configure a deterministic or LLM structured Planner before starting the run.",
            )
            result.phase = "PLAN_REJECTED"
            result.status = "FAILED"
            result.issues.append(issue)
            self.runtime.save_run(result)
            return self._update(result, "finalize_run")

        result.phase = "PLANNING"
        result.status = "PLANNING"
        self.runtime.save_run(result)
        try:
            plan: PlanResult = self.experiment_planner.plan(
                str(request["objective"]),
                str(request["hypothesis"]),
                self.runner.manifest,
                int(request["max_trials"]),
                int(request.get("seed", 42)),
            )
            result.spec = plan.spec
            result.planner_metadata = plan.metadata.model_dump(mode="json")
            self._save_plan_artifact(result.run_id, plan)
            self.runtime.emit(
                result,
                "plan.generated",
                "Planner generated a policy-valid ExperimentSpec",
                payload={
                    "source": plan.metadata.source,
                    "model": plan.metadata.model,
                },
            )
            self.runtime.save_run(result)
            return self._update(
                result, "initialize_run", spec=plan.spec.model_dump(mode="json")
            )
        except Exception as exc:
            result.phase = "PLAN_REJECTED"
            result.status = "FAILED"
            result.planner_metadata = {
                "source": (
                    "llm"
                    if self.experiment_planner.__class__.__name__
                    == "LLMStructuredPlanner"
                    else "deterministic"
                ),
                "model": getattr(self.experiment_planner, "model", None),
                "attempts": getattr(self.experiment_planner, "max_attempts", 1),
                "error": str(exc),
            }
            result.issues.append(
                ValidationIssue(
                    code="PLAN_GENERATION_FAILED",
                    phase="planning",
                    message=str(exc),
                    suggestion="Use deterministic planning or correct the LLM/API configuration before retrying.",
                )
            )
            self.runtime.emit(
                result,
                "plan.rejected",
                "Planner failed to generate a valid ExperimentSpec",
            )
            self.runtime.save_run(result)
            return self._update(result, "finalize_run")

    def _save_plan_artifact(self, run_id: UUID, plan: PlanResult) -> None:
        if self.artifact_store is None:
            return
        record = self.artifact_store.put_json(
            plan.spec.model_dump(mode="json"), run_id, "plan", "experiment_spec.json"
        )
        if self.repository is not None:
            self.repository.register_artifact(record)

    def _update(
        self, result: ExperimentRun, route: str, **fields: Any
    ) -> dict[str, Any]:
        self._write_progress(result)
        return {"run": run_to_snapshot(result), "route": route, **fields}

    def _write_progress(self, result: ExperimentRun) -> None:
        if self.progress_path is None:
            return
        total_trials = max(
            1,
            (
                int(result.spec.budget.max_trials)
                if result.spec is not None
                else self._progress_total_trials
            ),
        )
        completed_trials = len(result.outcomes)
        active_trial = (
            completed_trials + 1
            if result.phase in {"TRIAL_RUNNING", "PREFLIGHT"}
            else None
        )
        phase_messages = {
            "PLANNING": "Generating ExperimentSpec",
            "PLAN_VALIDATION": "Validating ExperimentSpec",
            "READY": "Selecting next Trial",
            "TRIAL_RUNNING": "Running current Trial gates and experiment",
            "OBSERVATION": "Reading Trial result",
            "ACTION_PLANNING": "Planning next Trial parameters",
            "COMPLETED": "All Trials completed",
            "HUMAN_REVIEW": "Waiting for human review",
        }
        payload = {
            "status": (
                "completed"
                if result.status == "COMPLETED"
                else (
                    "failed"
                    if result.status in {"FAILED", "TIMEOUT", "CANCELLED"}
                    else "running"
                )
            ),
            "phase": result.phase,
            "completed_trials": completed_trials,
            "total_trials": total_trials,
            "trial_index": active_trial,
            "message": phase_messages.get(result.phase, result.status),
        }
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        _write_progress_file(self.progress_path, payload)

    def _initialize_run(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        result.started_at = result.started_at or datetime.now(timezone.utc)
        result.phase = "PLAN_VALIDATION"
        result.status = "PLAN_VALIDATION"
        result.output_root.mkdir(parents=True, exist_ok=True)
        # The Run row must exist before baseline events and Artifacts reference it.
        self.runtime.save_run(result)
        self._snapshot_run_baseline(result, spec)
        self.runtime.track_start(result)
        self.runtime.save_run(result)
        return self._update(result, "validate_plan")

    def _snapshot_run_baseline(
        self, result: ExperimentRun, spec: ExperimentSpec
    ) -> None:
        """Persist an immutable per-Run starting point without touching the template."""
        baseline_root = result.output_root / ".baselines" / str(result.run_id)
        if not baseline_root.exists():
            shutil.copytree(self.runner.template_root, baseline_root)
        result.planner_metadata.setdefault(
            "baseline_parameters", self.runner.manifest.baseline_parameter_values
        )
        result.planner_metadata.setdefault(
            "baseline_template_sha256", _tree_digest(baseline_root)
        )
        already_recorded = any(
            event.name == "baseline.snapshot" for event in result.events
        )
        if not already_recorded:
            self.runtime.emit(
                result,
                "baseline.snapshot",
                "Run initialized from the weak registered baseline.",
                payload={
                    "path": str(baseline_root),
                    "parameters": self.runner.manifest.baseline_parameter_values,
                    "template_sha256": result.planner_metadata[
                        "baseline_template_sha256"
                    ],
                },
            )
            if self.artifact_store is not None:
                record = self.artifact_store.put_json(
                    {
                        "run_id": str(result.run_id),
                        "template_id": spec.template_id,
                        "parameters": self.runner.manifest.baseline_parameter_values,
                        "template_sha256": result.planner_metadata[
                            "baseline_template_sha256"
                        ],
                    },
                    result.run_id,
                    "baseline",
                    "baseline.json",
                )
                if self.repository is not None:
                    self.repository.register_artifact(record)

    def _validate_plan(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        issues = validate_experiment_spec(spec, self.runner.manifest)
        if issues:
            result.issues = issues
            result.phase = "PLAN_REJECTED"
            result.status = "FAILED"
            self.runtime.emit(
                result, "plan.rejected", "experiment plan violates the template policy"
            )
            self.runtime.save_run(result)
            return self._update(result, "finalize_run")

        if result.status == "COMPLETED" and result.outcomes:
            return self._update(result, "finalize_run")

        result.phase = "READY"
        result.status = "READY"
        candidates = self.runtime.build_candidates(spec)
        completed = [_parameter_key(item.parameters) for item in result.outcomes]
        best_value, stale_trials = self.runtime.score_history(result.outcomes, spec)
        self.runtime.save_run(result)
        return self._update(
            result,
            "budget_guard",
            candidates=candidates,
            completed_keys=completed,
            best_value=best_value,
            stale_trials=stale_trials,
            current_observation=None,
            pending_action=None,
        )

    def _budget_guard(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        self._refresh_elapsed(result)
        if result.status in {"FAILED", "CANCELLED"}:
            return self._update(result, "finalize_run")
        if result.elapsed_seconds >= spec.budget.max_total_seconds:
            result.phase = "BUDGET_EXCEEDED"
            result.status = "TIMEOUT"
            result.issues.append(
                ValidationIssue(
                    code="TOTAL_BUDGET_EXCEEDED",
                    phase="orchestration",
                    message="the experiment exceeded its total time budget",
                    suggestion="Increase max_total_seconds or reduce the Trial budget.",
                )
            )
            self.runtime.emit(
                result, "experiment.timeout", "total experiment time budget exceeded"
            )
            return self._update(result, "finalize_run")
        if state.get("current_parameters"):
            return self._update(result, "prepare_trial")
        if len(result.outcomes) >= spec.budget.max_trials:
            return self._update(result, "finalize_run")
        return self._update(result, "select_parameters")

    def _select_parameters(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        current = state.get("current_parameters") or result.next_parameters
        if current:
            return self._update(
                result, "prepare_trial", current_parameters=dict(current)
            )
        completed = set(state.get("completed_keys", []))
        candidate = self.runtime.first_uncompleted(
            state.get("candidates", []), completed
        )
        if candidate is None:
            return self._update(result, "finalize_run")
        return self._update(result, "prepare_trial", current_parameters=candidate)

    def _prepare_trial(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        parameters = dict(
            state.get("current_parameters") or result.next_parameters or {}
        )
        if not parameters:
            return self._update(result, "finalize_run")
        trial_id = result.current_trial_id or uuid4()
        result.current_trial_id = trial_id
        result.next_parameters = parameters
        result.phase = "TRIAL_RUNNING"
        result.status = "PREFLIGHT"
        self.runtime.save_run(result)
        return self._update(
            result,
            "execute_trial",
            current_trial_id=str(trial_id),
            current_parameters=parameters,
        )

    def _execute_trial(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        parameters = dict(
            state.get("current_parameters") or result.next_parameters or {}
        )
        trial_id = UUID(
            state.get("current_trial_id") or str(result.current_trial_id or uuid4())
        )
        workspace = result.output_root / str(trial_id)
        if workspace.is_dir() and not any(
            item.trial_id == trial_id for item in result.outcomes
        ):
            outcome = self.runner.recover_existing(
                spec, parameters, workspace, trial_id
            )
        else:
            timeout = self.runtime.remaining_trial_timeout(result, spec)
            outcome = self.runner.run(
                spec,
                parameters,
                result.output_root,
                result.run_id,
                trial_id=trial_id,
                timeout_seconds=timeout,
                trial_index=len(result.outcomes) + 1,
            )
        result.phase = "OBSERVATION"
        result.outcomes.append(outcome)
        return self._update(
            result,
            "persist_trial",
            current_trial_id=str(trial_id),
            current_parameters=parameters,
        )

    def _persist_trial(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        outcome = result.outcomes[-1]
        self.runtime.save_trial(result, len(result.outcomes), outcome)
        self.runtime.capture_artifacts(result, outcome)
        self.runtime.track_trial(result, outcome)
        if outcome.validation:
            self.runtime.emit(
                result,
                "trial.validation",
                "validation gates completed",
                outcome.trial_id,
                {
                    "gates": outcome.validation.model_dump(mode="json"),
                },
            )
        self.runtime.emit(
            result, f"trial.{outcome.status}", outcome.status, outcome.trial_id
        )
        result.current_trial_id = None
        result.next_parameters = None
        self._refresh_elapsed(result)
        self.runtime.save_run(result)
        completed = list(
            dict.fromkeys(
                state.get("completed_keys", []) + [_parameter_key(outcome.parameters)]
            )
        )
        return self._update(
            result,
            "build_observation",
            completed_keys=completed,
            current_parameters=None,
            current_trial_id=None,
        )

    def _build_observation(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        observation = build_observation(
            result.run_id,
            spec,
            result.outcomes,
            max(0, spec.budget.max_trials - len(result.outcomes)),
            int(state.get("stale_trials", 0)),
            result.decisions,
        )
        best_value, stale_trials = self.runtime.score_history(result.outcomes, spec)
        context_ref = None
        if observation.failure_context:
            context_ref = self._save_code_context(result, observation)
        elif result.outcomes:
            observation.failure_context = self._code_optimization_context()
            context_ref = self._save_code_context(result, observation)
        return self._update(
            result,
            "plan_action",
            current_observation=observation.model_dump(mode="json"),
            failure_fingerprint=observation.failure_fingerprint,
            best_value=best_value,
            stale_trials=stale_trials,
            code_context=observation.failure_context or None,
            code_context_ref=context_ref,
        )

    def _code_optimization_context(self) -> dict[str, Any]:
        root = self.runner.template_root
        source_path = root / self.runner.manifest.entrypoint
        config_path = root / "configs" / "experiment.yaml"
        try:
            source = source_path.read_text(encoding="utf-8")[-12_000:]
        except OSError:
            source = ""
        try:
            config = (
                config_path.read_text(encoding="utf-8")[-3_000:]
                if config_path.is_file()
                else ""
            )
        except OSError:
            config = ""
        return {
            "mode": "code_optimization",
            "sources": ["current_entrypoint", "current_config"],
            "code_context": [
                {
                    "file": self.runner.manifest.entrypoint,
                    "line": None,
                    "source": source,
                }
            ],
            "config": config,
            "mutable_files": sorted(self.runner.manifest.patchable_files),
            "immutable_files": sorted(self.runner.manifest.immutable_files),
        }

    def _save_code_context(
        self, result: ExperimentRun, observation: ExperimentObservation
    ) -> str | None:
        if self.artifact_store is None:
            return None
        trial_id = observation.latest_trial_id
        record = self.artifact_store.put_json(
            observation.failure_context,
            result.run_id,
            "code_context",
            f"{trial_id or result.run_id}.json",
            trial_id,
        )
        if self.repository is not None:
            self.repository.register_artifact(record)
        return record.relative_path

    def _plan_action(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        observation = ExperimentObservation.model_validate(state["current_observation"])
        try:
            plan: ActionPlanResult = self.action_planner.decide(
                observation, self.runner.manifest
            )
            decision = plan.decision
            metadata = plan.metadata.model_dump(mode="json")
        except Exception as exc:
            decision = ActionDecision(
                action="HUMAN_REVIEW",
                reason="The Action Planner failed and requires human review before another trial.",
            )
            metadata = {"source": "deterministic", "error": str(exc)}
        metadata["failure_fingerprint"] = observation.failure_fingerprint
        record = DecisionRecord(
            run_id=result.run_id,
            trial_id=result.outcomes[-1].trial_id if result.outcomes else None,
            observation=observation,
            decision=decision,
            metadata=metadata,
        )
        result.decisions.append(record)
        return self._update(
            result,
            "persist_decision",
            pending_action={
                "decision": decision.model_dump(mode="json"),
                "trial_id": str(record.trial_id) if record.trial_id else None,
                "metadata": metadata,
            },
        )

    def _persist_decision(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        record = result.decisions[-1]
        if self.repository is not None:
            self.repository.save_decision(record)
        self.runtime.track_decision(result, record)
        decision = record.decision
        self.runtime.emit(
            result,
            f"action.{decision.action.lower()}",
            decision.reason,
            record.trial_id,
            {
                "action": decision.action,
                "reason": decision.reason,
                "source": record.metadata.get("source"),
                "parameters": decision.parameters,
                "failure_code": decision.failure_code,
                "repair": (
                    decision.repair.model_dump(mode="json") if decision.repair else None
                ),
            },
        )
        self.runtime.save_run(result)
        return self._update(result, "validate_action")

    def _validate_action(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        decision = ActionDecision.model_validate(state["pending_action"]["decision"])
        observation = ExperimentObservation.model_validate(state["current_observation"])
        completed = set(state.get("completed_keys", []))
        if decision.action == "STOP":
            return self._update(result, "finalize_run")
        if decision.action == "HUMAN_REVIEW":
            self._set_waiting(result, "ACTION_REQUIRES_HUMAN", decision.reason)
            return self._update(
                result,
                "human_review",
                pending_human_review={
                    "reason": decision.reason,
                    "action": decision.action,
                },
            )
        if (
            decision.action in {"CONTINUE", "NARROW_SPACE"}
            and len(result.outcomes) >= spec.budget.max_trials
        ):
            self.runtime.emit(
                result,
                "experiment.budget_reached",
                "The configured Trial budget was reached; the Action Planner cannot schedule another Trial.",
                result.outcomes[-1].trial_id if result.outcomes else None,
            )
            return self._update(result, "finalize_run")
        if decision.action == "REPAIR":
            repeated = sum(
                1
                for item in result.decisions[:-1]
                if item.metadata.get("failure_fingerprint")
                and item.metadata.get("failure_fingerprint")
                == observation.failure_fingerprint
                and item.metadata.get("repair_status") == "accepted"
            )
            if observation.failure_fingerprint and repeated >= 1:
                self._set_waiting(
                    result,
                    "REPEATED_FAILURE",
                    "the same failure fingerprint appeared in an earlier Repair attempt",
                )
                return self._update(
                    result,
                    "human_review",
                    pending_human_review={"reason": "Repeated failure fingerprint"},
                )
            if decision.repair is None:
                self._set_waiting(
                    result,
                    "REPAIR_SPEC_MISSING",
                    "REPAIR action did not include a structured RepairSpec",
                )
                return self._update(
                    result,
                    "human_review",
                    pending_human_review={"reason": "RepairSpec missing"},
                )
            return self._update(result, "repair_validate")
        if decision.action == "CONTINUE":
            parameters = self.runtime.complete_parameters(
                decision.parameters or {}, spec
            )
            issues = validate_parameters(parameters, self.runner.manifest)
            if _parameter_key(parameters) in completed:
                issues.append(
                    ValidationIssue(
                        code="ACTION_DUPLICATE_PARAMETERS",
                        phase="action_validation",
                        message="Action selected parameters that have already been executed.",
                        suggestion="Choose a new policy-valid parameter combination or stop.",
                    )
                )
            if issues:
                return self._action_error(result, issues)
            self.runtime.emit(
                result,
                "action.accepted",
                "Action accepted; the next Trial will use the selected parameters.",
                result.outcomes[-1].trial_id,
            )
            return self._update(result, "budget_guard", current_parameters=parameters)
        if decision.action == "NARROW_SPACE":
            issues = self.runtime.validate_search_space(decision.search_space or {})
            if issues:
                return self._action_error(result, issues)
            narrowed = dict(spec.search_space)
            narrowed.update(decision.search_space or {})
            return self._update(
                result,
                "update_search_space",
                narrowed_space={
                    name: item.model_dump(mode="json")
                    for name, item in narrowed.items()
                },
            )
        return self._action_error(
            result,
            [
                ValidationIssue(
                    code="ACTION_UNKNOWN",
                    phase="action_validation",
                    message=f"Unsupported action: {decision.action}",
                    suggestion="Use one of the registered Action types.",
                )
            ],
        )

    def _update_search_space(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        narrowed = {
            name: ParameterRange.model_validate(value)
            for name, value in state.get("narrowed_space", {}).items()
        }
        spec.search_space = narrowed
        result.spec = spec
        candidates = self.runtime.build_candidates(spec, narrowed)
        next_candidate = self.runtime.first_uncompleted(
            candidates, set(state.get("completed_keys", []))
        )
        if next_candidate is None:
            return self._update(result, "finalize_run")
        self.runtime.emit(
            result,
            "action.accepted",
            "Narrowed search space accepted for the next Trial.",
            result.outcomes[-1].trial_id,
        )
        self.runtime.save_run(result)
        return self._update(
            result,
            "budget_guard",
            spec=spec.model_dump(mode="json"),
            candidates=candidates,
            current_parameters=next_candidate,
        )

    def _repair_validate(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        trial_id = result.outcomes[-1].trial_id if result.outcomes else None
        repair_count = sum(
            1
            for item in result.decisions
            if item.trial_id == trial_id and item.decision.action == "REPAIR"
        )
        if repair_count >= spec.budget.max_repairs_per_trial:
            self._set_waiting(
                result,
                "REPAIR_BUDGET_EXCEEDED",
                "the repair budget for this Trial has been exhausted",
            )
            return self._update(
                result,
                "human_review",
                pending_human_review={"reason": "Repair budget exceeded"},
            )
        return self._update(result, "repair_apply")

    def _repair_apply(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, spec = self._rehydrate(state)
        decision = ActionDecision.model_validate(state["pending_action"]["decision"])
        observation = ExperimentObservation.model_validate(state["current_observation"])
        trial_id = result.outcomes[-1].trial_id if result.outcomes else uuid4()
        repair_root = (
            result.output_root / ".repairs" / str(result.decisions[-1].decision_id)
        )
        try:
            patch_digest = hashlib.sha256(
                (decision.repair.patch if decision.repair else "").encode("utf-8")
            ).hexdigest()[:16]
            if any(
                item.get("patch_sha256") == patch_digest
                for item in state.get("repair_attempts", [])
            ):
                raise RepairError(
                    "REPAIR_DUPLICATE_PATCH",
                    "the proposed Patch is identical to a rejected Patch attempt",
                )
            repair_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.runner.template_root, repair_root)
            config_path = repair_root / "configs" / "experiment.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = dict(observation.current_parameters)
            config["seed"] = spec.seed
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
            )
            applied = apply_unified_patch(
                repair_root, decision.repair, self.runner.manifest
            )
            preflight = validate_repaired_template(
                repair_root,
                self.runner.manifest,
                self.runner.project_root / "requirements.txt",
            )
            applied.result.preflight_passed = preflight.passed
            if not preflight.passed:
                raise RepairError(
                    "REPAIR_PREFLIGHT_FAILED",
                    "repaired code failed AST or dependency preflight",
                )
            validation = self.runner.validation.run(
                repair_root,
                result.run_id,
                trial_id,
                self.runtime.remaining_trial_timeout(result, spec),
            )
            if not validation.passed:
                raise RepairError(
                    "REPAIR_GATE_FAILED",
                    "repaired template failed Pytest or Smoke gate",
                )
            self.runner = TrialRunner(
                self.runner.project_root,
                repair_root,
                self.runner.manifest,
                executor=self.runner.executor,
            )
            self.runtime.runner = self.runner
            result.active_template_root = repair_root
            if self.artifact_store is not None:
                patch_record = self.artifact_store.put_text(
                    decision.repair.patch,
                    result.run_id,
                    "repair_patch",
                    f"{result.decisions[-1].decision_id}.diff",
                    trial_id,
                )
                result_record = self.artifact_store.put_json(
                    applied.result.model_dump(mode="json"),
                    result.run_id,
                    "repair_result",
                    f"{result.decisions[-1].decision_id}.json",
                    trial_id,
                )
                if self.repository is not None:
                    self.repository.register_artifact(patch_record)
                    self.repository.register_artifact(result_record)
            result.decisions[-1].metadata.update(
                {
                    "repair_status": "accepted",
                    "patched_base_sha256": applied.result.base_sha256,
                    "patched_sha256": applied.result.patched_sha256,
                }
            )
            if self.repository is not None:
                self.repository.save_decision(result.decisions[-1])
            result.phase = "REPAIR_ACCEPTED"
            self.runtime.emit(
                result,
                "repair.accepted",
                "Repair passed all gates; retrying the failed parameters.",
                trial_id,
            )
            self.runtime.save_run(result)
            return self._update(
                result,
                "budget_guard",
                current_parameters=dict(observation.current_parameters),
                retry_current=True,
            )
        except RepairError as exc:
            record = result.decisions[-1]
            record.metadata.update(
                {"repair_status": "rejected", "repair_error": str(exc)}
            )
            if self.repository is not None:
                self.repository.save_decision(record)
            self.runtime.emit(result, "repair.rejected", str(exc), trial_id)
            attempts = list(state.get("repair_attempts", []))
            attempts.append(
                {"code": exc.code, "message": str(exc), "patch_sha256": patch_digest}
            )
            observation.failure_context["repair_attempts"] = attempts
            if len(attempts) >= spec.budget.max_repairs_per_trial:
                self._set_waiting(
                    result,
                    "REPAIR_BUDGET_EXCEEDED",
                    "the repair budget was exhausted after failed Patch attempts",
                )
                return self._update(
                    result,
                    "human_review",
                    pending_human_review={
                        "reason": "Repair budget exceeded",
                        "attempts": attempts,
                    },
                    repair_attempts=attempts,
                )
            result.status = "READY"
            result.phase = "REPAIR_RETRY_PLANNING"
            self.runtime.save_run(result)
            return self._update(
                result,
                "plan_action",
                current_observation=observation.model_dump(mode="json"),
                pending_action=None,
                repair_attempts=attempts,
            )

    def _human_review(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, _ = self._rehydrate(state)
        self._refresh_elapsed(result)
        review = state.get("pending_human_review") or {
            "reason": "Human review required"
        }
        self.runtime.save_run(result)
        response = interrupt(review)
        return {"human_command": response, "route": "apply_human_command"}

    def _apply_human_command(self, state: AutoExpGraphState) -> dict[str, Any]:
        result, _ = self._rehydrate(state)
        command = state.get("human_command") or {}
        action = (
            str(command.get("action", "STOP_AND_REPORT")).upper()
            if isinstance(command, dict)
            else str(command).upper()
        )
        if action == "APPROVE_REPAIR":
            result.status = "READY"
            return self._update(result, "repair_apply")
        if action in {"RETRY_TRIAL", "RESUME"}:
            result.status = "READY"
            return self._update(result, "budget_guard")
        result.status = "FAILED" if action != "CANCEL" else "CANCELLED"
        return self._update(result, "finalize_run")

    def _finalize_run(self, state: AutoExpGraphState) -> dict[str, Any]:
        result = snapshot_to_run(state["run"])
        self._refresh_elapsed(result)
        if state.get("spec"):
            result.spec = ExperimentSpec.model_validate(state["spec"])
        result.output_root = Path(state["output_root"]).resolve()
        if result.status not in {"FAILED", "TIMEOUT", "CANCELLED", "WAITING_FOR_HUMAN"}:
            result.status = "COMPLETED"
            result.phase = "COMPLETED"
        if not result.events or result.events[-1].name != "experiment.completed":
            self.runtime.emit(result, "experiment.completed", result.status)
        self.runtime.save_run(result)
        self.runtime.track_finish(result)
        return self._update(result, "END")

    def _action_error(
        self, result: ExperimentRun, issues: list[ValidationIssue]
    ) -> dict[str, Any]:
        result.issues.extend(issues)
        self._set_waiting(result, issues[0].code, issues[0].message)
        self.runtime.emit(result, "action.rejected", issues[0].message)
        return self._update(
            result,
            "human_review",
            pending_human_review={"reason": issues[0].message, "code": issues[0].code},
        )

    def _set_waiting(self, result: ExperimentRun, code: str, message: str) -> None:
        result.phase = "HUMAN_REVIEW"
        result.status = "WAITING_FOR_HUMAN"
        result.issues.append(
            ValidationIssue(
                code=code,
                phase="graph",
                message=message,
                suggestion="Review the persisted graph state and resume with an explicit command.",
            )
        )

    @staticmethod
    def _route(state: AutoExpGraphState) -> str:
        return state.get("route", "finalize_run")


def _parameter_key(parameters: dict[str, Any]) -> str:
    import json

    return json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
