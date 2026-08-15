from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autoexp.domain import ActionDecision, ExperimentObservation, TemplateManifest
from autoexp.planning import ActionPlanResult, deterministic_candidates
from autoexp.planning.compat import PlannerMetadata


def _parameter_key(parameters: dict[str, Any]) -> str:
    return json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _sample(policy: Any, rng: random.Random) -> Any:
    if policy.choices:
        return rng.choice(policy.choices)
    if policy.type == "int":
        return rng.randint(int(policy.min), int(policy.max))
    if policy.type == "float":
        low, high = float(policy.min), float(policy.max)
        return (
            math.exp(rng.uniform(math.log(low), math.log(high)))
            if low > 0 and high / low >= 100
            else rng.uniform(low, high)
        )
    if policy.type == "bool":
        return bool(rng.getrandbits(1))
    return policy.default


class RandomActionPlanner:
    """Seeded random search using the shared Action Planner contract."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult:
        if observation.remaining_trials <= 0:
            return ActionPlanResult(
                ActionDecision(action="STOP", reason="Trial budget exhausted."),
                PlannerMetadata(source="random"),
            )
        tried = {_parameter_key(item.parameters) for item in observation.trials}
        for _ in range(100):
            parameters = {
                name: _sample(policy, self.rng)
                for name, policy in observation.search_space.items()
            }
            parameters.update(
                {
                    name: value
                    for name, value in observation.fixed_parameters.items()
                    if name not in observation.search_space
                }
            )
            if _parameter_key(parameters) not in tried:
                return ActionPlanResult(
                    ActionDecision(
                        action="CONTINUE",
                        parameters=parameters,
                        reason="Seeded random-search proposal.",
                    ),
                    PlannerMetadata(source="random"),
                )
        return ActionPlanResult(
            ActionDecision(action="STOP", reason="No unique random proposal found."),
            PlannerMetadata(source="random"),
        )


class OptunaActionPlanner:
    """History-conditioned Optuna TPE policy; AutoExp still owns execution."""

    def __init__(self, seed: int = 42):
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna policy requires `pip install optuna`.") from exc
        self.optuna, self.seed, self.study, self.reported = optuna, seed, None, 0

    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult:
        if observation.remaining_trials <= 0:
            return ActionPlanResult(
                ActionDecision(action="STOP", reason="Trial budget exhausted."),
                PlannerMetadata(source="optuna"),
            )
        if self.study is None:
            self.study = self.optuna.create_study(
                direction=observation.metric_direction,
                sampler=self.optuna.samplers.TPESampler(seed=self.seed),
            )
        for item in observation.trials[self.reported :]:
            if item.primary_metric is not None:
                distributions = {
                    name: self._distribution(observation.search_space[name])
                    for name in item.parameters
                    if name in observation.search_space
                }
                self.study.add_trial(
                    self.optuna.trial.create_trial(
                        params={name: item.parameters[name] for name in distributions},
                        distributions=distributions,
                        value=item.primary_metric,
                    )
                )
        self.reported = len(observation.trials)
        trial, parameters = self.study.ask(), {}
        for name, policy in observation.search_space.items():
            if policy.choices:
                parameters[name] = trial.suggest_categorical(name, policy.choices)
            elif policy.type == "int":
                parameters[name] = trial.suggest_int(
                    name, int(policy.min), int(policy.max)
                )
            elif policy.type == "float":
                log = (
                    float(policy.min) > 0
                    and float(policy.max) / float(policy.min) >= 100
                )
                parameters[name] = trial.suggest_float(
                    name, float(policy.min), float(policy.max), log=log
                )
            elif policy.type == "bool":
                parameters[name] = trial.suggest_categorical(name, [False, True])
            else:
                parameters[name] = policy.default
        parameters.update(
            {
                name: value
                for name, value in observation.fixed_parameters.items()
                if name not in observation.search_space
            }
        )
        tried = {_parameter_key(item.parameters) for item in observation.trials}
        if _parameter_key(parameters) in tried:
            fallback = deterministic_candidates(
                manifest,
                fixed_parameters=observation.fixed_parameters,
                search_space=observation.search_space,
                count=max(2, observation.remaining_trials + 1),
            )
            parameters = next(
                (
                    candidate
                    for candidate in fallback
                    if _parameter_key(candidate) not in tried
                ),
                None,
            )
            if parameters is None:
                return ActionPlanResult(
                    ActionDecision(
                        action="STOP", reason="No unique Optuna proposal found."
                    ),
                    PlannerMetadata(source="optuna"),
                )
        return ActionPlanResult(
            ActionDecision(
                action="CONTINUE",
                parameters=parameters,
                reason="Optuna TPE proposal from shared Trial history.",
            ),
            PlannerMetadata(source="optuna"),
        )

    def _distribution(self, policy: Any) -> Any:
        if policy.choices:
            return self.optuna.distributions.CategoricalDistribution(policy.choices)
        if policy.type == "int":
            return self.optuna.distributions.IntDistribution(
                int(policy.min), int(policy.max)
            )
        if policy.type == "float":
            log = float(policy.min) > 0 and float(policy.max) / float(policy.min) >= 100
            return self.optuna.distributions.FloatDistribution(
                float(policy.min), float(policy.max), log=log
            )
        return self.optuna.distributions.CategoricalDistribution(
            [False, True] if policy.type == "bool" else [policy.default]
        )


@dataclass
class PolicyResult:
    policy: str
    seed: int
    run_id: str
    status: str
    best_metric: float | None
    metric_history: list[float]
    elapsed_seconds: float
    successful_trials: int
    baseline_metric: float | None = None
    improvement: float | None = None
    failed_trials: int = 0
    dataset_sha256: str | None = None
    baseline_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    template_id: str
    metric_name: str
    direction: str
    max_trials: int
    results: list[PolicyResult] = field(default_factory=list)
    benchmark_version: str = "phase1"
    task_description: str = ""
    dataset_id: str = ""
    dataset_sha256: str | None = None
    baseline_parameters: dict[str, Any] = field(default_factory=dict)
    parameter_policy: dict[str, Any] = field(default_factory=dict)
    action_scope: str = "parameter_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "template_id": self.template_id,
            "task_description": self.task_description,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "max_trials": self.max_trials,
            "baseline_parameters": self.baseline_parameters,
            "parameter_policy": self.parameter_policy,
            "action_scope": self.action_scope,
            "policy_summary": self.policy_summary(),
            "results": [asdict(item) for item in self.results],
        }

    def policy_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for policy in sorted({item.policy for item in self.results}):
            items = [item for item in self.results if item.policy == policy]
            best_values = [
                item.best_metric for item in items if item.best_metric is not None
            ]
            improvements = [
                item.improvement for item in items if item.improvement is not None
            ]
            summary.append(
                {
                    "policy": policy,
                    "runs": len(items),
                    "completed_runs": sum(item.status == "COMPLETED" for item in items),
                    "mean_best_metric": (
                        statistics.mean(best_values) if best_values else None
                    ),
                    "mean_improvement": (
                        statistics.mean(improvements) if improvements else None
                    ),
                    "mean_elapsed_seconds": statistics.mean(
                        item.elapsed_seconds for item in items
                    ),
                }
            )
        return summary

    def to_markdown(self) -> str:
        rows = [
            "# AutoExp Optimization Benchmark",
            "",
            f"Benchmark version: {self.benchmark_version}",
            f"Template: {self.template_id}",
            f"Dataset: {self.dataset_id}",
            f"Dataset SHA256: {self.dataset_sha256 or 'unavailable'}",
            f"Metric: {self.metric_name} ({self.direction})",
            f"Action scope: {self.action_scope}",
            f"Baseline: {json.dumps(self.baseline_parameters, sort_keys=True)}",
            "",
            "## Policy Summary",
            "",
            "| Policy | Runs | Completed | Mean Best | Mean Improvement | Mean Seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in self.policy_summary():
            best = (
                "-"
                if item["mean_best_metric"] is None
                else f"{item['mean_best_metric']:.6f}"
            )
            improvement = (
                "-"
                if item["mean_improvement"] is None
                else f"{item['mean_improvement']:.6f}"
            )
            rows.append(
                f"| {item['policy']} | {item['runs']} | {item['completed_runs']} | {best} | {improvement} | {item['mean_elapsed_seconds']:.2f} |"
            )
        rows.extend(
            [
                "",
                "## Runs",
                "",
                "| Policy | Seed | Baseline | Best | Improvement | Successful | Failed | Seconds | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in self.results:
            best = "-" if item.best_metric is None else f"{item.best_metric:.6f}"
            baseline = (
                "-" if item.baseline_metric is None else f"{item.baseline_metric:.6f}"
            )
            improvement = "-" if item.improvement is None else f"{item.improvement:.6f}"
            rows.append(
                f"| {item.policy} | {item.seed} | {baseline} | {best} | {improvement} | {item.successful_trials} | {item.failed_trials} | {item.elapsed_seconds:.2f} | {item.status} |"
            )
        return "\n".join(rows) + "\n"


class OptimizationBenchmarkRunner:
    def __init__(self, service_factory: Any):
        self.service_factory = service_factory

    def run(
        self,
        template_id: str,
        objective: str,
        hypothesis: str,
        max_trials: int,
        seeds: list[int],
        policies: list[str],
        output_root: Path,
        allow_code_optimization: bool = False,
    ) -> BenchmarkReport:
        report = None
        for seed in seeds:
            for name in policies:
                scope = output_root / "state" / name / str(seed)
                scope.mkdir(parents=True, exist_ok=True)
                service = self._build_service(
                    template_id, name, seed, scope, allow_code_optimization
                )
                started = time.perf_counter()
                run = service.run(
                    objective,
                    hypothesis,
                    max_trials,
                    output_root=output_root / "runs" / name / str(seed),
                    seed=seed,
                )
                history = [
                    item.metrics.primary.value for item in run.outcomes if item.metrics
                ]
                baseline = history[0] if history else None
                best = (
                    run.best.metrics.primary.value
                    if run.best and run.best.metrics
                    else None
                )
                improvement = None
                if baseline is not None and best is not None:
                    improvement = (
                        best - baseline
                        if run.spec.metric.direction == "maximize"
                        else baseline - best
                    )
                report = report or BenchmarkReport(
                    template_id=template_id,
                    metric_name=run.spec.metric.name,
                    direction=run.spec.metric.direction,
                    max_trials=max_trials,
                    task_description=getattr(service.base_manifest, "description", None)
                    or "",
                    dataset_id=run.spec.dataset_id,
                    dataset_sha256=self._dataset_sha256(service, run.spec.dataset_id),
                    baseline_parameters=dict(run.spec.fixed_parameters),
                    parameter_policy={
                        name: policy.model_dump(mode="json")
                        for name, policy in service.base_manifest.parameter_policy.items()
                    },
                    action_scope=(
                        "parameters_and_code"
                        if allow_code_optimization
                        else "parameter_only"
                    ),
                )
                report.results.append(
                    PolicyResult(
                        policy=name,
                        seed=seed,
                        run_id=str(run.run_id),
                        status=run.status,
                        best_metric=best,
                        metric_history=history,
                        elapsed_seconds=time.perf_counter() - started,
                        successful_trials=len(history),
                        baseline_metric=baseline,
                        improvement=improvement,
                        failed_trials=max(0, len(run.outcomes) - len(history)),
                        dataset_sha256=report.dataset_sha256,
                        baseline_parameters=dict(run.spec.fixed_parameters),
                    )
                )
        if report is None:
            raise ValueError("Benchmark requires at least one seed and policy.")
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "benchmark_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_version": report.benchmark_version,
                    "template_id": report.template_id,
                    "dataset_id": report.dataset_id,
                    "dataset_sha256": report.dataset_sha256,
                    "metric_name": report.metric_name,
                    "direction": report.direction,
                    "max_trials": report.max_trials,
                    "seeds": seeds,
                    "policies": policies,
                    "baseline_parameters": report.baseline_parameters,
                    "action_scope": report.action_scope,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (output_root / "benchmark.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_root / "benchmark.md").write_text(
            report.to_markdown(), encoding="utf-8"
        )
        return report

    def _build_service(
        self,
        template_id: str,
        policy: str,
        seed: int,
        scope: Path,
        allow_code_optimization: bool = False,
    ) -> Any:
        """Create an isolated service while keeping old factory callables usable."""
        try:
            return self.service_factory(
                template_id=template_id,
                action_planner=self._policy(policy, seed, allow_code_optimization),
                benchmark_root=scope,
            )
        except TypeError as exc:
            if "benchmark_root" not in str(exc):
                raise
            return self.service_factory(
                template_id=template_id,
                action_planner=self._policy(policy, seed, allow_code_optimization),
            )

    @staticmethod
    def _dataset_sha256(service: Any, dataset_id: str) -> str | None:
        records = [
            item for item in service.list_datasets() if item.dataset_id == dataset_id
        ]
        if not records:
            return None
        payload = [
            {
                "name": item.name,
                "role": item.role,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in sorted(
                records[0].files, key=lambda value: (value.role, value.name)
            )
        ]
        return (
            __import__("hashlib")
            .sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

    @staticmethod
    def _policy(name: str, seed: int, allow_code_optimization: bool = False) -> Any:
        if name == "random":
            return RandomActionPlanner(seed)
        if name == "optuna":
            return OptunaActionPlanner(seed)
        if name == "llm":
            from autoexp.planning import build_action_planner

            return build_action_planner(
                "llm", seed=seed, allow_code_optimization=allow_code_optimization
            )
        raise ValueError(f"unknown benchmark policy: {name}")
