from __future__ import annotations

import json
import math

from typing import Any
from uuid import UUID

from autoexp.domain import (
    DecisionRecord,
    ExperimentRun,
    ExperimentSpec,
    ParameterRange,
    RunEvent,
    TemplateManifest,
    TrialOutcome,
)
from autoexp.domain.errors import ValidationIssue

from autoexp.planning import deterministic_candidates
from autoexp.tracking import ExperimentTracker, NullTracker


class RunRuntime:
    """Shared side-effect boundary for the legacy and LangGraph workflows.

    The workflow decides *when* an operation happens. This object owns the
    repeatable mechanics for persistence, event emission, tracking, candidate
    generation and policy validation.
    """

    def __init__(
        self,
        runner: Any,
        repository: Any | None = None,
        artifact_store: Any | None = None,
        tracker: ExperimentTracker | None = None,
    ):
        self.runner = runner
        self.repository = repository
        self.artifact_store = artifact_store
        self.tracker = tracker or NullTracker()

    def remaining_trial_timeout(
        self, result: ExperimentRun, spec: ExperimentSpec
    ) -> int:
        remaining = spec.budget.max_total_seconds - result.elapsed_seconds
        if remaining <= 0:
            return 0
        return max(1, min(spec.budget.timeout_seconds, math.ceil(remaining)))

    def validate_search_space(
        self,
        search_space: dict[str, ParameterRange],
        manifest: TemplateManifest | None = None,
    ) -> list[ValidationIssue]:
        active_manifest = manifest or self.runner.manifest
        issues: list[ValidationIssue] = []
        for name, policy in search_space.items():
            manifest_policy = active_manifest.parameter_policy.get(name)
            if manifest_policy is None:
                issues.append(
                    ValidationIssue(
                        code="ACTION_PARAMETER_UNKNOWN",
                        phase="action_validation",
                        message=f"Action search space contains an unknown parameter: {name}",
                        suggestion="Use only parameters registered by the template manifest.",
                    )
                )
                continue
            if policy.type != manifest_policy.type:
                issues.append(
                    ValidationIssue(
                        code="ACTION_PARAMETER_TYPE",
                        phase="action_validation",
                        message=f"Action parameter type mismatch: {name}",
                        suggestion="Keep the manifest parameter type unchanged.",
                    )
                )
                continue
            if (
                policy.min is not None
                and manifest_policy.min is not None
                and policy.min < manifest_policy.min
            ):
                issues.append(
                    ValidationIssue(
                        code="ACTION_PARAMETER_RANGE",
                        phase="action_validation",
                        message=f"Action search range exceeds the lower policy bound: {name}",
                        suggestion="Narrow the range inside the manifest limits.",
                    )
                )
            if (
                policy.max is not None
                and manifest_policy.max is not None
                and policy.max > manifest_policy.max
            ):
                issues.append(
                    ValidationIssue(
                        code="ACTION_PARAMETER_RANGE",
                        phase="action_validation",
                        message=f"Action search range exceeds the upper policy bound: {name}",
                        suggestion="Narrow the range inside the manifest limits.",
                    )
                )
            if policy.choices is not None and manifest_policy.choices is not None:
                if not set(policy.choices).issubset(manifest_policy.choices):
                    issues.append(
                        ValidationIssue(
                            code="ACTION_PARAMETER_CHOICES",
                            phase="action_validation",
                            message=f"Action choices exceed the manifest policy: {name}",
                            suggestion="Use a subset of the registered choices.",
                        )
                    )
        return issues

    def complete_parameters(
        self, parameters: dict[str, Any], spec: ExperimentSpec
    ) -> dict[str, Any]:
        completed = {
            name: _default_value(policy)
            for name, policy in self.runner.manifest.parameter_policy.items()
        }
        completed.update(spec.fixed_parameters)
        completed.update(parameters)
        return completed

    def score_history(
        self, outcomes: list[TrialOutcome], spec: ExperimentSpec
    ) -> tuple[float | None, int]:
        best_value: float | None = None
        stale_trials = 0
        for outcome in outcomes:
            if not outcome.metrics:
                continue
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
        return best_value, stale_trials

    def build_candidates(
        self,
        spec: ExperimentSpec,
        search_space: dict[str, ParameterRange] | None = None,
    ) -> list[dict[str, Any]]:
        active_space = search_space or spec.search_space
        return deterministic_candidates(
            self.runner.manifest,
            fixed_parameters=spec.fixed_parameters,
            search_space=active_space,
            count=max(1, spec.budget.max_trials),
        )

    @staticmethod
    def first_uncompleted(
        candidates: list[dict[str, Any]],
        completed: set[str],
    ) -> dict[str, Any] | None:
        for candidate in candidates:
            if _parameter_key(candidate) not in completed:
                return dict(candidate)
        return None

    def emit(
        self,
        result: ExperimentRun,
        name: str,
        message: str,
        trial_id: UUID | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = RunEvent(
            name=name, message=message, trial_id=trial_id, payload=payload or {}
        )
        result.events.append(event)
        if self.repository is not None:
            self.repository.append_event(result.run_id, event)

    def save_run(self, result: ExperimentRun) -> None:
        if self.repository is not None:
            self.repository.save_run(result)

    def save_trial(
        self, result: ExperimentRun, ordinal: int, outcome: TrialOutcome
    ) -> None:
        if self.repository is not None:
            self.repository.save_trial(result.run_id, ordinal, outcome)

    def capture_artifacts(self, result: ExperimentRun, outcome: TrialOutcome) -> None:
        if self.artifact_store is None:
            return
        records = self.artifact_store.capture_trial(
            result.run_id, outcome, self.runner.manifest
        )
        if self.repository is not None:
            for record in records:
                self.repository.register_artifact(record)
                self.track_artifact(result, record)

    def track_start(self, result: ExperimentRun) -> None:
        self._track(lambda: self.tracker.start_run(result))

    def track_trial(self, result: ExperimentRun, outcome: TrialOutcome) -> None:
        self._track(lambda: self.tracker.log_trial(result, outcome))

    def track_decision(self, result: ExperimentRun, decision: DecisionRecord) -> None:
        self._track(lambda: self.tracker.log_decision(result, decision))

    def track_artifact(self, result: ExperimentRun, artifact: Any) -> None:
        self._track(lambda: self.tracker.log_artifact(result, artifact))

    def track_finish(self, result: ExperimentRun) -> None:
        self._track(lambda: self.tracker.finish_run(result))

    @staticmethod
    def _track(operation: Any) -> None:
        try:
            operation()
        except Exception:
            # Tracking is deliberately best-effort; SQLite remains authoritative.
            pass


def _default_value(policy: ParameterRange) -> Any:
    if policy.default is not None:
        return policy.default
    if policy.choices:
        return policy.choices[0]
    if policy.min is not None:
        return policy.min
    return {"int": 1, "float": 0.0, "str": "", "bool": False}[policy.type]


def _parameter_key(parameters: dict[str, Any]) -> str:
    return json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
