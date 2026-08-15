from __future__ import annotations

from uuid import UUID

from autoexp.domain import ExperimentObservation, ExperimentSpec, TrialOutcome
from autoexp.repair.context import collect_failure_context


def build_observation(
    run_id: UUID,
    spec: ExperimentSpec,
    outcomes: list[TrialOutcome],
    remaining_trials: int,
    stale_trials: int,
    decisions: list | None = None,
) -> ExperimentObservation:
    trial_records = []
    for ordinal, outcome in enumerate(outcomes, start=1):
        failure_codes = [issue.code for issue in outcome.issues]
        failure_messages = [issue.message for issue in outcome.issues]
        if outcome.execution and outcome.execution.error:
            failure_codes.append(outcome.execution.error.code)
            failure_messages.append(outcome.execution.error.message)
        trial_records.append(
            {
                "trial_id": outcome.trial_id,
                "ordinal": ordinal,
                "status": outcome.status,
                "parameters": outcome.parameters,
                "primary_metric": (
                    outcome.metrics.primary.value if outcome.metrics else None
                ),
                "secondary_metrics": (
                    outcome.metrics.secondary if outcome.metrics else {}
                ),
                "metric_details": outcome.metrics.details if outcome.metrics else {},
                "dataset_sha256": (
                    outcome.metrics.dataset_sha256 if outcome.metrics else None
                ),
                "preflight_passed": bool(
                    outcome.preflight and outcome.preflight.passed
                ),
                "failure_codes": list(dict.fromkeys(failure_codes)),
                "failure_messages": list(dict.fromkeys(failure_messages)),
            }
        )

    best = None
    metric_history: list[float] = []
    for outcome in outcomes:
        if not outcome.metrics:
            continue
        value = outcome.metrics.primary.value
        metric_history.append(value)
        if best is None or (
            value > best.metrics.primary.value
            if spec.metric.direction == "maximize"
            else value < best.metrics.primary.value
        ):
            best = outcome

    latest = outcomes[-1] if outcomes else None
    latest_failure_codes = trial_records[-1]["failure_codes"] if trial_records else []
    latest_failure_messages = (
        trial_records[-1]["failure_messages"] if trial_records else []
    )
    failure_context = (
        collect_failure_context(latest, None, decisions or [])
        if latest and latest.metrics is None
        else {}
    )
    dataset_sha256 = next(
        (
            record["dataset_sha256"]
            for record in reversed(trial_records)
            if record["dataset_sha256"]
        ),
        None,
    )
    return ExperimentObservation(
        run_id=run_id,
        template_id=spec.template_id,
        dataset_id=spec.dataset_id,
        model_id=spec.model_id,
        objective=spec.objective,
        hypothesis=spec.hypothesis,
        metric_name=spec.metric.name,
        metric_direction=spec.metric.direction,
        current_parameters=latest.parameters if latest else dict(spec.fixed_parameters),
        fixed_parameters=dict(spec.fixed_parameters),
        search_space=dict(spec.search_space),
        trials=trial_records,
        latest_trial_id=latest.trial_id if latest else None,
        latest_status=latest.status if latest else None,
        latest_failure_codes=latest_failure_codes,
        latest_failure_messages=latest_failure_messages,
        failure_fingerprint=failure_context.get("fingerprint"),
        failure_context=failure_context,
        best_trial_id=best.trial_id if best else None,
        best_metric=best.metrics.primary.value if best and best.metrics else None,
        metric_history=metric_history,
        dataset_sha256=dataset_sha256,
        remaining_trials=max(0, remaining_trials),
        stale_trials=max(0, stale_trials),
        patience=spec.stop_conditions.patience,
        target_metric=spec.stop_conditions.target_metric,
        max_repairs_per_trial=spec.budget.max_repairs_per_trial,
    )


__all__ = ["build_observation"]
