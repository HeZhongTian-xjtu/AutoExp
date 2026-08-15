from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import Any
from uuid import UUID, uuid4

from autoexp.domain import (
    ActionDecision,
    DecisionRecord,
    ExperimentRun,
    ExperimentObservation,
    ExperimentSpec,
    ParameterRange,
    TrialOutcome,
)
from autoexp.domain.errors import ValidationIssue
from autoexp.domain.policies import validate_experiment_spec, validate_parameters
from autoexp.planning import (
    ActionPlanResult,
    ActionPlannerProtocol,
    DeterministicActionPlanner,
)
from autoexp.repair import RepairError, apply_unified_patch, validate_repaired_template
from autoexp.tracking import ExperimentTracker, NullTracker

from .observation_builder import build_observation
from .trial_runner import TrialRunner
from .runtime import RunRuntime


class ExperimentOrchestrator:
    """Framework-independent trial orchestration with persisted checkpoints."""

    def __init__(
        self,
        runner: TrialRunner,
        repository: Any | None = None,
        artifact_store: Any | None = None,
        action_planner: ActionPlannerProtocol | None = None,
        tracker: ExperimentTracker | None = None,
    ):
        self.runner = runner
        self.repository = repository
        self.artifact_store = artifact_store
        self.action_planner = action_planner or DeterministicActionPlanner()
        self.tracker = tracker or NullTracker()

        self.runtime = RunRuntime(
            self.runner, self.repository, self.artifact_store, self.tracker
        )

    def run(
        self,
        spec: ExperimentSpec,
        output_root: Path,
        run_id: UUID | None = None,
        initial_run: ExperimentRun | None = None,
        planner_metadata: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        result = initial_run or ExperimentRun(run_id=run_id or uuid4())
        initial_status = result.status
        initial_phase = result.phase
        clock_started = time.perf_counter()
        elapsed_before = result.elapsed_seconds
        result.started_at = result.started_at or datetime.now(timezone.utc)
        if result.active_template_root and result.active_template_root.is_dir():
            self.runner = TrialRunner(
                self.runner.project_root,
                result.active_template_root,
                self.runner.manifest,
                executor=self.runner.executor,
            )
            self.runtime.runner = self.runner
        result.spec = spec
        result.output_root = output_root.resolve()
        if planner_metadata is not None:
            result.planner_metadata = planner_metadata
        self._track_start(result)
        output_root.mkdir(parents=True, exist_ok=True)
        self._save_run(result)

        result.phase = "PLAN_VALIDATION"
        result.status = "PLAN_VALIDATION"
        self._checkpoint(result, elapsed_before, clock_started)
        self._save_run(result)
        result.issues = validate_experiment_spec(spec, self.runner.manifest)
        if result.issues:
            result.status = "FAILED"
            self._emit(
                result, "plan.rejected", "experiment plan violates the template policy"
            )
            self._save_run(result)
            return result

        if (
            initial_run is not None
            and initial_status == "COMPLETED"
            and result.outcomes
        ):
            result.status = initial_status
            result.phase = initial_phase
            return result

        result.phase = "READY"
        result.status = "READY"
        self._checkpoint(result, elapsed_before, clock_started)
        candidates = self._build_candidates(spec)
        completed = {_parameter_key(outcome.parameters) for outcome in result.outcomes}
        best_value, stale_trials = self._score_history(result.outcomes, spec)
        next_parameters = result.next_parameters or self._resume_next_parameters(result)

        try:
            recovery_blocked = False
            if result.current_trial_id and any(
                item.trial_id == result.current_trial_id for item in result.outcomes
            ):
                result.current_trial_id = None
                result.next_parameters = None
                self._checkpoint(result, elapsed_before, clock_started)
            if result.current_trial_id and next_parameters:
                checkpoint_workspace = output_root / str(result.current_trial_id)
                if checkpoint_workspace.is_dir():
                    recovered = self.runner.recover_existing(
                        spec,
                        next_parameters,
                        checkpoint_workspace,
                        result.current_trial_id,
                    )
                    completed.add(_parameter_key(next_parameters))
                    result.outcomes.append(recovered)
                    result.phase = "OBSERVATION"
                    self._save_trial(result, len(result.outcomes), recovered)
                    self._capture_artifacts(result, recovered)
                    self._track_trial(result, recovered)
                    result.current_trial_id = None
                    result.next_parameters = None
                    self._checkpoint(result, elapsed_before, clock_started)
                    self._emit(
                        result,
                        f"trial.recovered.{recovered.status}",
                        "recovered checkpointed Trial workspace",
                        recovered.trial_id,
                    )
                    if recovered.metrics is None:
                        observation = build_observation(
                            result.run_id,
                            spec,
                            result.outcomes,
                            spec.budget.max_trials - len(result.outcomes),
                            stale_trials,
                            result.decisions,
                        )
                        _, action_issues, _ = self._plan_next_action(
                            result,
                            observation,
                            recovered.trial_id,
                            completed,
                            spec,
                        )
                        result.issues.extend(recovered.issues)
                        result.issues.extend(action_issues)
                        result.status = "WAITING_FOR_HUMAN"
                        recovery_blocked = True
                    else:
                        best_value, stale_trials = self._score_history(
                            result.outcomes, spec
                        )
                        observation = build_observation(
                            result.run_id,
                            spec,
                            result.outcomes,
                            spec.budget.max_trials - len(result.outcomes),
                            stale_trials,
                            result.decisions,
                        )
                        next_parameters, action_issues, should_stop = (
                            self._plan_next_action(
                                result,
                                observation,
                                recovered.trial_id,
                                completed,
                                spec,
                            )
                        )
                        if action_issues:
                            result.issues.extend(action_issues)
                            result.status = "WAITING_FOR_HUMAN"
                            recovery_blocked = True
                        elif should_stop:
                            recovery_blocked = True

            while (
                not recovery_blocked and len(result.outcomes) < spec.budget.max_trials
            ):
                self._checkpoint(result, elapsed_before, clock_started)
                if result.elapsed_seconds >= spec.budget.max_total_seconds:
                    result.phase = "BUDGET_EXCEEDED"
                    result.status = "TIMEOUT"
                    result.issues.append(
                        ValidationIssue(
                            code="TOTAL_BUDGET_EXCEEDED",
                            phase="orchestration",
                            message="the experiment exceeded its total time budget before the next Trial",
                            suggestion="Increase max_total_seconds or reduce the Trial budget.",
                        )
                    )
                    self._emit(
                        result,
                        "experiment.timeout",
                        "total experiment time budget exceeded",
                    )
                    break
                if next_parameters is None:
                    next_parameters = self._first_uncompleted(candidates, completed)
                if next_parameters is None:
                    break

                ordinal = len(result.outcomes) + 1
                parameter_key = _parameter_key(next_parameters)
                if parameter_key in completed:
                    next_parameters = None
                    continue
                result.phase = "TRIAL_RUNNING"
                result.status = "PREFLIGHT"
                result.current_trial_id = result.current_trial_id or uuid4()
                result.next_parameters = dict(next_parameters)
                self._checkpoint(result, elapsed_before, clock_started)
                timeout_seconds = self._remaining_trial_timeout(result, spec)
                if timeout_seconds <= 0:
                    result.phase = "BUDGET_EXCEEDED"
                    result.status = "TIMEOUT"
                    result.issues.append(
                        ValidationIssue(
                            code="TOTAL_BUDGET_EXCEEDED",
                            phase="orchestration",
                            message="no time remained for the next Trial",
                            suggestion="Increase max_total_seconds or reduce the Trial budget.",
                        )
                    )
                    break
                outcome = self.runner.run(
                    spec,
                    next_parameters,
                    output_root,
                    result.run_id,
                    trial_id=result.current_trial_id,
                    timeout_seconds=timeout_seconds,
                )
                result.phase = "OBSERVATION"
                result.outcomes.append(outcome)
                completed.add(parameter_key)
                self._save_trial(result, ordinal, outcome)
                self._capture_artifacts(result, outcome)
                self._track_trial(result, outcome)
                if outcome.validation:
                    self._emit(
                        result,
                        "trial.validation",
                        "validation gates completed",
                        outcome.trial_id,
                        {"gates": outcome.validation.model_dump(mode="json")},
                    )
                result.current_trial_id = None
                result.next_parameters = None
                self._checkpoint(result, elapsed_before, clock_started)
                self._emit(
                    result, f"trial.{outcome.status}", outcome.status, outcome.trial_id
                )

                if outcome.metrics is None:
                    observation = build_observation(
                        result.run_id,
                        spec,
                        result.outcomes,
                        spec.budget.max_trials - len(result.outcomes),
                        stale_trials,
                        result.decisions,
                    )
                    _, action_issues, _ = self._plan_next_action(
                        result,
                        observation,
                        outcome.trial_id,
                        completed,
                        spec,
                    )
                    result.issues.extend(outcome.issues)
                    result.issues.extend(action_issues)
                    result.status = (
                        "FAILED"
                        if outcome.status == "execution_failed"
                        else "WAITING_FOR_HUMAN"
                    )
                    self._save_run(result)
                    break

                current = outcome.metrics.primary.value
                improved = best_value is None or (
                    current > best_value
                    if spec.metric.direction == "maximize"
                    else current < best_value
                )
                if improved:
                    best_value = current
                    stale_trials = 0
                else:
                    stale_trials += 1

                observation = build_observation(
                    result.run_id,
                    spec,
                    result.outcomes,
                    spec.budget.max_trials - len(result.outcomes),
                    stale_trials,
                    result.decisions,
                )
                next_parameters, action_issues, should_stop = self._plan_next_action(
                    result,
                    observation,
                    outcome.trial_id,
                    completed,
                    spec,
                )
                if (
                    result.decisions
                    and result.decisions[-1].decision.action == "REPAIR"
                    and next_parameters
                ):
                    completed.discard(_parameter_key(next_parameters))
                if action_issues:
                    result.issues.extend(action_issues)
                    result.status = "WAITING_FOR_HUMAN"
                    break
                if should_stop:
                    break

            result.phase = (
                "COMPLETED"
                if result.status not in {"FAILED", "WAITING_FOR_HUMAN", "TIMEOUT"}
                else result.phase
            )
            result.status = (
                "COMPLETED"
                if result.status not in {"FAILED", "WAITING_FOR_HUMAN", "TIMEOUT"}
                else result.status
            )
            if not result.events or result.events[-1].name != "experiment.completed":
                self._emit(result, "experiment.completed", result.status)
            self._save_run(result)
        except KeyboardInterrupt:
            result.phase = "PAUSED"
            result.status = "PAUSED"
            self._emit(
                result,
                "experiment.paused",
                "experiment interrupted; resume from persisted state",
            )
            self._save_run(result)
            raise
        return result

    def _checkpoint(
        self, result: ExperimentRun, elapsed_before: float, clock_started: float
    ) -> None:
        result.elapsed_seconds = elapsed_before + max(
            0.0, time.perf_counter() - clock_started
        )
        self._save_run(result)

    def _remaining_trial_timeout(
        self, result: ExperimentRun, spec: ExperimentSpec
    ) -> int:
        return self.runtime.remaining_trial_timeout(result, spec)

    def _plan_next_action(
        self,
        result: ExperimentRun,
        observation: ExperimentObservation,
        trial_id: UUID | None,
        completed: set[str] | None = None,
        spec: ExperimentSpec | None = None,
    ) -> tuple[dict[str, Any] | None, list[ValidationIssue], bool]:
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

        record = DecisionRecord(
            run_id=result.run_id,
            trial_id=trial_id,
            observation=observation,
            decision=decision,
            metadata=metadata,
        )
        record.metadata["failure_fingerprint"] = observation.failure_fingerprint
        result.decisions.append(record)
        if self.repository is not None:
            self.repository.save_decision(record)
        self._track_decision(result, record)
        self._emit(
            result,
            f"action.{decision.action.lower()}",
            decision.reason,
            trial_id,
            {
                "action": decision.action,
                "reason": decision.reason,
                "source": metadata.get("source"),
                "parameters": decision.parameters,
                "failure_code": decision.failure_code,
                "repair": (
                    decision.repair.model_dump(mode="json") if decision.repair else None
                ),
            },
        )

        if decision.action == "STOP":
            return None, [], True
        if decision.action == "HUMAN_REVIEW":
            issue = ValidationIssue(
                code="ACTION_REQUIRES_HUMAN",
                phase="action_validation",
                message=decision.reason,
                suggestion="Inspect the persisted Observation and decide whether the next run needs a code or template change.",
            )
            return None, [issue], True
        if spec is None or completed is None:
            return None, [], False

        if decision.action == "REPAIR":
            repeated = sum(
                1
                for item in result.decisions[:-1]
                if item.metadata.get("failure_fingerprint")
                and item.metadata.get("failure_fingerprint")
                == observation.failure_fingerprint
            )
            if observation.failure_fingerprint and repeated >= 1:
                issue = ValidationIssue(
                    code="REPEATED_FAILURE",
                    phase="action_validation",
                    message="the same failure fingerprint appeared in an earlier Repair attempt",
                    suggestion="Stop automatic repair and review the persisted code context manually.",
                )
                return None, [issue], True
            repair_count = sum(
                1
                for item in result.decisions
                if item.trial_id == trial_id and item.decision.action == "REPAIR"
            )
            if repair_count > spec.budget.max_repairs_per_trial:
                issue = ValidationIssue(
                    code="REPAIR_BUDGET_EXCEEDED",
                    phase="action_validation",
                    message="the repair budget for this Trial has been exhausted",
                    suggestion="Review the failure manually or start a new experiment with a corrected template.",
                )
                return None, [issue], True
            if decision.repair is None:
                issue = ValidationIssue(
                    code="REPAIR_SPEC_MISSING",
                    phase="action_validation",
                    message="REPAIR action did not include a structured RepairSpec",
                    suggestion="Provide a Unified Diff and an allowlisted target file.",
                )
                return None, [issue], True
            try:
                repair_root = result.output_root / ".repairs" / str(record.decision_id)
                repair_root.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(self.runner.template_root, repair_root)
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
                    issue = ValidationIssue(
                        code="REPAIR_PREFLIGHT_FAILED",
                        phase="action_validation",
                        message="repaired code failed AST or dependency preflight",
                        suggestion="Inspect the persisted repair patch and preflight findings.",
                        details={
                            "issues": [
                                item.model_dump(mode="json")
                                for item in preflight.issues
                            ]
                        },
                    )
                    return None, [issue], True
                validation = self.runner.validation.run(
                    repair_root,
                    result.run_id,
                    trial_id or uuid4(),
                    self._remaining_trial_timeout(result, spec),
                )
                if not validation.passed:
                    issue = ValidationIssue(
                        code="REPAIR_GATE_FAILED",
                        phase="action_validation",
                        message="repaired template failed Pytest or Smoke gate",
                        suggestion="Inspect the persisted gate logs and review the Repair proposal.",
                        details={"gates": validation.model_dump(mode="json")},
                    )
                    self._emit(result, "repair.rejected", issue.message, trial_id)
                    return None, [issue], True
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
                        f"{record.decision_id}.diff",
                        trial_id,
                    )
                    result_record = self.artifact_store.put_json(
                        applied.result.model_dump(mode="json"),
                        result.run_id,
                        "repair_result",
                        f"{record.decision_id}.json",
                        trial_id,
                    )
                    if self.repository is not None:
                        self.repository.register_artifact(patch_record)
                        self.repository.register_artifact(result_record)
                result.phase = "REPAIR_ACCEPTED"
                self._emit(
                    result,
                    "repair.accepted",
                    "Repair passed allowlist and preflight; retrying the failed parameters.",
                    trial_id,
                )
                self._save_run(result)
                return dict(observation.current_parameters), [], False
            except RepairError as exc:
                issue = ValidationIssue(
                    code=exc.code,
                    phase="action_validation",
                    message=str(exc),
                    suggestion="Keep the Unified Diff inside the template allowlist and match the current file context.",
                )
                self._emit(result, "repair.rejected", str(exc), trial_id)
                return None, [issue], True

        if decision.action == "CONTINUE":
            parameters = self._complete_parameters(decision.parameters or {}, spec)
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
                self._emit(
                    result,
                    "action.rejected",
                    "Action parameters failed server validation.",
                    trial_id,
                )
                return None, issues, True
            self._emit(
                result,
                "action.accepted",
                "Action accepted; the next Trial will use the selected parameters.",
                trial_id,
            )
            return parameters, [], False

        if decision.action == "NARROW_SPACE":
            issues = self._validate_search_space(decision.search_space or {})
            if issues:
                self._emit(
                    result,
                    "action.rejected",
                    "Action search space failed server validation.",
                    trial_id,
                )
                return None, issues, True
            narrowed = dict(spec.search_space)
            narrowed.update(decision.search_space or {})
            candidates = self._build_candidates(spec, narrowed)
            next_candidate = self._first_uncompleted(candidates, completed)
            if next_candidate is None:
                self._emit(
                    result,
                    "action.accepted",
                    "The narrowed search space has no untried candidate; stopping.",
                    trial_id,
                )
                return None, [], True
            self._emit(
                result,
                "action.accepted",
                "Narrowed search space accepted for the next Trial.",
                trial_id,
            )
            return next_candidate, [], False

        issue = ValidationIssue(
            code="ACTION_UNKNOWN",
            phase="action_validation",
            message=f"Unsupported action: {decision.action}",
            suggestion="Use one of the registered Action types.",
        )
        return None, [issue], True

    def _validate_search_space(
        self, search_space: dict[str, ParameterRange]
    ) -> list[ValidationIssue]:
        return self.runtime.validate_search_space(search_space)

    def _complete_parameters(
        self, parameters: dict[str, Any], spec: ExperimentSpec
    ) -> dict[str, Any]:
        return self.runtime.complete_parameters(parameters, spec)

    def _resume_next_parameters(self, result: ExperimentRun) -> dict[str, Any] | None:
        if not result.decisions:
            return None
        last = result.decisions[-1].decision
        if last.action == "CONTINUE" and last.parameters:
            return dict(last.parameters)
        return None

    def _score_history(
        self, outcomes: list[TrialOutcome], spec: ExperimentSpec
    ) -> tuple[float | None, int]:
        return self.runtime.score_history(outcomes, spec)

    def _first_uncompleted(
        self, candidates: list[dict[str, Any]], completed: set[str]
    ) -> dict[str, Any] | None:
        return self.runtime.first_uncompleted(candidates, completed)

    def _emit(
        self,
        result: ExperimentRun,
        name: str,
        message: str,
        trial_id: UUID | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.runtime.emit(result, name, message, trial_id, payload)

    def _save_run(self, result: ExperimentRun) -> None:
        self.runtime.save_run(result)

    def _save_trial(
        self, result: ExperimentRun, ordinal: int, outcome: TrialOutcome
    ) -> None:
        self.runtime.save_trial(result, ordinal, outcome)

    def _capture_artifacts(self, result: ExperimentRun, outcome: TrialOutcome) -> None:
        self.runtime.capture_artifacts(result, outcome)

    def _track_start(self, result: ExperimentRun) -> None:
        self.runtime.track_start(result)

    def _track_trial(self, result: ExperimentRun, outcome: TrialOutcome) -> None:
        self.runtime.track_trial(result, outcome)

    def _track_decision(self, result: ExperimentRun, decision: DecisionRecord) -> None:
        self.runtime.track_decision(result, decision)

    def _track_artifact(self, result: ExperimentRun, artifact: Any) -> None:
        self.runtime.track_artifact(result, artifact)

    def _track_finish(self, result: ExperimentRun) -> None:
        self.runtime.track_finish(result)

    def _build_candidates(
        self,
        spec: ExperimentSpec,
        search_space: dict[str, ParameterRange] | None = None,
    ) -> list[dict[str, Any]]:
        return self.runtime.build_candidates(spec, search_space)


def _parameter_key(parameters: dict[str, Any]) -> str:
    import json

    return json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
