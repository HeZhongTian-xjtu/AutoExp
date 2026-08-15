from __future__ import annotations

import json
from typing import Any, Iterable


class ReportBuilder:
    """Build a deterministic Markdown report from structured AutoExp records."""

    def build(self, run: Any, artifacts: Iterable[Any] = ()) -> str:
        spec = run.spec
        objective = (
            spec.objective
            if spec
            else "Planner did not produce a valid experiment specification."
        )
        hypothesis = spec.hypothesis if spec else "No hypothesis was available."
        phase = getattr(run, "phase", run.status)
        elapsed_seconds = getattr(run, "elapsed_seconds", 0.0)
        lines = [
            "# AutoExp Experiment Report",
            "",
            f"- Run ID: `{run.run_id}`",
            f"- Status: **{run.status}**",
            f"- Phase: `{phase}`",
            f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
            f"- Planner: `{self._planner_label(run)}`",
            "",
            "## Objective and Hypothesis",
            "",
            f"**Objective**: {self._text(objective)}",
            "",
            f"**Hypothesis**: {self._text(hypothesis)}",
            "",
            "## Experiment Plan",
            "",
        ]
        if spec is None:
            lines.append("No valid ExperimentSpec was produced.")
        else:
            lines.extend(
                [
                    f"- Template: `{spec.template_id}`",
                    f"- Dataset: `{spec.dataset_id}`",
                    f"- Model: `{spec.model_id}`",
                    f"- Metric: `{spec.metric.name}` ({spec.metric.direction})",
                    f"- Trial budget: `{spec.budget.max_trials}`",
                    f"- Seed: `{spec.seed}`",
                    f'- Weak baseline parameters: `{json.dumps((run.planner_metadata or {}).get("baseline_parameters", {}), ensure_ascii=False, sort_keys=True)}`',
                    "",
                    "### Fixed Parameters",
                    "",
                    self._code_json(spec.fixed_parameters),
                    "",
                    "### Search Space",
                    "",
                    self._code_json(
                        {
                            name: policy.model_dump(mode="json")
                            for name, policy in spec.search_space.items()
                        }
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "## Trial Results",
                "",
                "| Trial | Status | Parameters | Primary Metric | Preflight |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        if run.outcomes:
            for outcome in run.outcomes:
                metric = outcome.metrics.primary.value if outcome.metrics else "-"
                preflight = (
                    "passed"
                    if outcome.preflight and outcome.preflight.passed
                    else "not passed"
                )
                parameters = json.dumps(
                    outcome.parameters, ensure_ascii=False, sort_keys=True
                )
                lines.append(
                    f"| `{outcome.trial_id}` | `{outcome.status}` | `{parameters}` | {metric} | `{preflight}` |"
                )
                if outcome.validation:
                    gate_state = ", ".join(
                        f"{gate.name}={gate.status}"
                        for gate in outcome.validation.gates
                    )
                    lines.append(f"  Validation gates: `{gate_state}`")
        else:
            lines.append("| - | `no_trial` | - | - | - |")

        lines.extend(["", "### Detailed Metrics", ""])
        detailed = False
        for outcome in run.outcomes:
            if not outcome.metrics:
                continue
            detailed = True
            lines.append(
                f"- Trial `{outcome.trial_id}` secondary metrics: `{json.dumps(outcome.metrics.secondary, ensure_ascii=False, sort_keys=True)}`"
            )
            if outcome.metrics.details:
                lines.append(
                    f"  Details: `{json.dumps(outcome.metrics.details, ensure_ascii=False, sort_keys=True)}`"
                )
            if outcome.metrics.dataset_sha256:
                lines.append(f"  Dataset SHA-256: `{outcome.metrics.dataset_sha256}`")
        if not detailed:
            lines.append("No detailed metrics were recorded.")

        lines.extend(["", "### Aggregate Statistics", ""])
        values = [
            outcome.metrics.primary.value
            for outcome in run.outcomes
            if outcome.metrics is not None
        ]
        successful = sum(outcome.status == "succeeded" for outcome in run.outcomes)
        total = len(run.outcomes)
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            lines.extend(
                [
                    f"- Valid metric count: `{len(values)}` / `{total}`",
                    (
                        f"- Trial success rate: `{successful / total:.2%}`"
                        if total
                        else "- Trial success rate: `0.00%`"
                    ),
                    f"- Primary metric mean: `{mean:.6f}`",
                    f"- Primary metric standard deviation: `{variance ** 0.5:.6f}`",
                    f"- Primary metric range: `[{min(values):.6f}, {max(values):.6f}]`",
                ]
            )
        else:
            lines.append("No valid metric was available for aggregate statistics.")
        resource_samples = []
        for outcome in run.outcomes:
            for execution in (
                outcome.execution,
                getattr(outcome, "evaluation_execution", None),
            ):
                if (
                    execution
                    and execution.resource_usage.get("wall_seconds") is not None
                ):
                    resource_samples.append(
                        float(execution.resource_usage["wall_seconds"])
                    )
        if resource_samples:
            lines.append(
                f"- Recorded execution wall time: `{sum(resource_samples):.3f}s` across `{len(resource_samples)}` process invocations"
            )

        lines.extend(["", "### Validation Gates", ""])
        validation_seen = False
        for outcome in run.outcomes:
            if not outcome.validation:
                continue
            validation_seen = True
            for gate in outcome.validation.gates:
                lines.append(
                    f"- Trial `{outcome.trial_id}` `{gate.name}`: **{gate.status}** (exit `{gate.exit_code}`)"
                )
                for issue in gate.issues:
                    lines.append(f"  - [{issue.code}] {self._text(issue.message)}")
        if not validation_seen:
            lines.append("No validation gates were configured for this template.")

        best = run.best
        lines.extend(["", "## Best Result", ""])
        if best and best.metrics:
            lines.extend(
                [
                    f"The best completed Trial is `{best.trial_id}` with `{best.metrics.primary.name} = {best.metrics.primary.value:.6f}`.",
                    "",
                    f"Parameters: `{json.dumps(best.parameters, ensure_ascii=False, sort_keys=True)}`",
                    "",
                    "This is empirical evidence for the registered comparison, not a causal proof beyond the selected dataset and template.",
                ]
            )
        else:
            lines.append("No completed Trial produced a valid primary metric.")

        lines.extend(["", "## Agent Decisions", ""])
        if run.decisions:
            for record in run.decisions:
                observation = record.observation
                decision = record.decision
                source = record.metadata.get("source", "unknown")
                lines.append(
                    f"- `{record.created_at.isoformat()}` after Trial `{record.trial_id}`: **{decision.action}** "
                    f"({source}) - {self._text(decision.reason)}"
                )
                if decision.strategy:
                    lines.append(f"  Strategy: `{decision.strategy}`")
                lines.append(
                    f"  Observation: latest metric `{observation.best_metric}`, "
                    f"remaining trials `{observation.remaining_trials}`, stale trials `{observation.stale_trials}`."
                )
                if decision.parameters:
                    lines.append(
                        f"  Next parameters: `{json.dumps(decision.parameters, ensure_ascii=False, sort_keys=True)}`"
                    )
                if decision.failure_code:
                    lines.append(f"  Failure code: `{decision.failure_code}`")
        else:
            lines.append("No Action Planner decision was recorded.")

        lines.extend(["", "## Failure Reasons", ""])
        failures = list(self._issues(run.issues))
        for outcome in run.outcomes:
            failures.extend(
                self._issues(outcome.issues, prefix=f"Trial {outcome.trial_id}")
            )
            for label, execution in (
                ("execution", outcome.execution),
                ("evaluation", getattr(outcome, "evaluation_execution", None)),
            ):
                if execution and execution.error:
                    failures.append(
                        f"Trial {outcome.trial_id} {label}: {execution.error.code} - {execution.error.message}"
                    )
            if outcome.dataset_integrity and not outcome.dataset_integrity.passed:
                failures.extend(
                    self._issues(
                        outcome.dataset_integrity.issues,
                        prefix=f"Trial {outcome.trial_id} dataset",
                    )
                )
        if failures:
            lines.extend(f"- {failure}" for failure in failures)
        else:
            lines.append("No structured failure was recorded.")

        lines.extend(["", "## Execution Trace", ""])
        if run.events:
            for event in run.events:
                trial = f" (Trial `{event.trial_id}`)" if event.trial_id else ""
                lines.append(
                    f"- `{event.created_at.isoformat()}` `{event.name}`{trial}: {event.message}"
                )
        else:
            lines.append("No events were recorded.")

        lines.extend(["", "## Artifacts", ""])
        artifact_list = list(artifacts)
        if artifact_list:
            lines.extend(
                f"- `{artifact.kind}`: `{artifact.relative_path}` ({artifact.size_bytes} bytes, `{artifact.sha256[:12]}`)"
                for artifact in artifact_list
            )
        else:
            lines.append("No artifacts were registered.")

        lines.extend(["", "## Conclusion", ""])
        if run.status == "COMPLETED" and best and best.metrics:
            lines.append(
                "The controlled experiment completed successfully. The report records the selected hypothesis, "
                "the exact parameter trials, the validated metric output, the agent decisions, and the reproducibility artifacts."
            )
        elif run.status == "COMPLETED":
            lines.append(
                "The workflow completed, but no valid primary metric was available for a conclusion."
            )
        else:
            lines.append(
                "The workflow did not complete successfully. Review the structured failure reasons before resuming."
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _planner_label(run: Any) -> str:
        metadata = run.planner_metadata or {}
        source = (
            metadata.get("source", "unknown")
            if isinstance(metadata, dict)
            else metadata.source
        )
        model = metadata.get("model") if isinstance(metadata, dict) else metadata.model
        suffix = f"/{model}" if model else ""
        return f"{source}{suffix}"

    @staticmethod
    def _text(value: str) -> str:
        return value.replace("\n", " ").strip()

    @staticmethod
    def _code_json(value: Any) -> str:
        return (
            "```json\n"
            + json.dumps(value, ensure_ascii=False, indent=2, default=str)
            + "\n```"
        )

    @staticmethod
    def _issues(issues: Iterable[Any], prefix: str | None = None) -> list[str]:
        rendered = []
        for issue in issues:
            line = f"[{issue.code}] {issue.message}"
            if issue.suggestion:
                line += f" Suggestion: {issue.suggestion}"
            rendered.append(f"{prefix}: {line}" if prefix else line)
        return rendered
