from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Protocol

from dotenv import load_dotenv

from autoexp.domain import (
    ActionDecision,
    ExperimentObservation,
    ParameterRange,
    TemplateManifest,
)
from autoexp.llm import FunctionSpec

from .candidates import deterministic_candidates
from .compat import PlannerError, PlannerMetadata, _bounded_openai_query
from .planner import _parse_json_response


class ActionPlanResult:
    def __init__(self, decision: ActionDecision, metadata: PlannerMetadata):
        self.decision = decision
        self.metadata = metadata


class ActionPlannerProtocol(Protocol):
    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult: ...


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


def _candidate_pool(
    observation: ExperimentObservation,
    manifest: TemplateManifest,
    search_space: dict[str, ParameterRange] | None = None,
) -> list[dict[str, Any]]:
    active_space = search_space or observation.search_space
    return deterministic_candidates(
        manifest,
        fixed_parameters=observation.fixed_parameters,
        search_space=active_space,
        count=max(1, observation.remaining_trials, len(active_space) * 2),
    )


def _next_untried(
    observation: ExperimentObservation,
    manifest: TemplateManifest,
    search_space: dict[str, ParameterRange] | None = None,
) -> dict[str, Any] | None:
    tried = {_parameter_key(trial.parameters) for trial in observation.trials}
    for candidate in _candidate_pool(observation, manifest, search_space):
        if _parameter_key(candidate) not in tried:
            return candidate
    return None


class DeterministicActionPlanner:
    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult:
        if observation.latest_status and observation.latest_status != "succeeded":
            decision = ActionDecision(
                action="HUMAN_REVIEW",
                failure_code=(
                    observation.latest_failure_codes[0]
                    if observation.latest_failure_codes
                    else "TRIAL_FAILED"
                ),
                reason="The latest trial did not produce a valid metric; human review is required before another execution.",
            )
            return ActionPlanResult(decision, PlannerMetadata(source="deterministic"))
        if observation.remaining_trials <= 0:
            return ActionPlanResult(
                ActionDecision(
                    action="STOP",
                    reason="The configured trial budget has been exhausted.",
                ),
                PlannerMetadata(source="deterministic"),
            )
        if (
            observation.target_metric is not None
            and observation.best_metric is not None
        ):
            reached = (
                observation.best_metric >= observation.target_metric
                if observation.metric_direction == "maximize"
                else observation.best_metric <= observation.target_metric
            )
            if reached:
                return ActionPlanResult(
                    ActionDecision(
                        action="STOP",
                        reason="The configured target metric has been reached.",
                    ),
                    PlannerMetadata(source="deterministic"),
                )
        if observation.stale_trials >= observation.patience:
            return ActionPlanResult(
                ActionDecision(
                    action="STOP",
                    reason="The configured patience limit was reached without improvement.",
                ),
                PlannerMetadata(source="deterministic"),
            )
        next_parameters = _next_untried(observation, manifest, observation.search_space)
        if next_parameters is None:
            return ActionPlanResult(
                ActionDecision(
                    action="STOP",
                    reason="No new policy-valid parameter combination remains.",
                ),
                PlannerMetadata(source="deterministic"),
            )
        return ActionPlanResult(
            ActionDecision(
                action="CONTINUE",
                parameters=next_parameters,
                reason="Continue with the next untried policy-valid parameter combination.",
            ),
            PlannerMetadata(source="deterministic"),
        )


def _action_function_spec(
    manifest: TemplateManifest, allow_repair: bool = True
) -> FunctionSpec:
    json_types = {
        "int": "integer",
        "float": "number",
        "str": "string",
        "bool": "boolean",
    }
    parameter_properties: dict[str, dict[str, Any]] = {}
    for name, policy in manifest.parameter_policy.items():
        schema: dict[str, Any] = {"type": json_types[policy.type]}
        if policy.min is not None:
            schema["minimum"] = policy.min
        if policy.max is not None:
            schema["maximum"] = policy.max
        if policy.choices is not None:
            schema["enum"] = policy.choices
        parameter_properties[name] = schema
    return FunctionSpec(
        name="propose_experiment_action",
        description="Return one bounded next action for the registered experiment.",
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": (
                        ["CONTINUE", "NARROW_SPACE", "REPAIR", "STOP", "HUMAN_REVIEW"]
                        if allow_repair
                        else ["CONTINUE", "NARROW_SPACE", "STOP", "HUMAN_REVIEW"]
                    ),
                },
                "parameters": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": parameter_properties,
                },
                "search_space": {"type": "object", "additionalProperties": True},
                "failure_code": {"type": ["string", "null"]},
                "strategy": {"type": ["string", "null"]},
                "repair": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "target_file": {"type": "string"},
                        "patch": {"type": "string"},
                        "reason": {"type": "string"},
                        "expected_base_sha256": {"type": ["string", "null"]},
                        "verification": {"type": "string"},
                    },
                },
                "reason": {"type": "string"},
                "conclusion": {"type": ["string", "null"]},
            },
            "required": ["action", "reason"],
        },
    )


class LLMActionPlanner:
    def __init__(
        self,
        model: str | None = None,
        query_fn: Callable[..., Any] | None = None,
        max_attempts: int = 2,
        seed: int | None = None,
        allow_code_optimization: bool = True,
    ):
        load_dotenv(override=True)
        self.model = model or os.getenv("AUTOEXP_PLANNER_MODEL", "deepseek-v4-flash")
        self.query_fn = query_fn or _bounded_openai_query
        self.max_attempts = max(1, max_attempts)
        self.seed = seed
        self.allow_code_optimization = allow_code_optimization

    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult:
        function_spec = _action_function_spec(
            manifest, allow_repair=self.allow_code_optimization
        )
        system_message = (
            "You are the action component of a controlled AutoExp agent. "
            "Return only a function call. Choose a bounded action using only registered parameters. "
            "Do not invent files, models, shell commands, dependencies, or metrics. "
            "The server will validate every action before execution."
        )
        observation_payload = observation.model_dump(mode="json")
        if not self.allow_code_optimization:
            observation_payload["failure_context"] = {
                "mode": "parameter_optimization",
                "sources": ["trial_history"],
            }
        prompt = json.dumps(
            {
                "observation": observation_payload,
                "benchmark_seed": self.seed,
                "registered_parameters": {
                    name: policy.model_dump(mode="json")
                    for name, policy in manifest.parameter_policy.items()
                },
                "file_policy": {
                    "mutable_files": sorted(manifest.patchable_files),
                    "immutable_files": sorted(manifest.immutable_files),
                    "evaluator_entrypoint": manifest.evaluator_entrypoint,
                },
                "action_rules": {
                    "CONTINUE": (
                        "provide a complete next parameter object. When a parameter has min/max but no choices, "
                        "choose a new numeric value inside that interval; use the full trial history and keep the "
                        "best observed configuration in mind."
                    ),
                    "NARROW_SPACE": "provide a smaller policy-valid search_space; continuous ranges are valid when choices are absent",
                    "REPAIR": (
                        "provide a minimal unified diff in repair, limited to an allowed file. "
                        "This action is allowed after a successful Trial as code_optimization: "
                        "use the observed metric trend and the supplied code_context to improve the "
                        "registered training code, then keep the same parameters for verification."
                    ),
                    "STOP": "use only when target, patience, budget, or evidence requires stopping; do not stop after the weak baseline if trials remain",
                    "HUMAN_REVIEW": "use when a failure cannot be safely resolved automatically",
                },
                "continuous_optimization_rule": (
                    "This template intentionally starts from an underfit baseline. After a successful Trial, "
                    "prefer CONTINUE with a policy-valid, non-duplicate parameter set until the trial budget is "
                    "exhausted or the metric clearly stops improving. Do not restrict values to old discrete choices."
                ),
                "code_optimization_rule": (
                    "When the current code is a deliberately weak baseline and Trials remain, you may choose "
                    "REPAIR with strategy=code_optimization. Make one small, explainable change; never modify "
                    "datasets, evaluators, resource limits, or files outside mutable_files."
                    if self.allow_code_optimization
                    else "Code optimization is disabled for this parameter-only benchmark. Never return REPAIR."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.query_fn(
                    system_message=system_message,
                    user_message=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1_200,
                    func_spec=function_spec,
                )
                candidate = _parse_json_response(raw)
                if isinstance(candidate.get("action_decision"), dict):
                    candidate = candidate["action_decision"]
                candidate["action"] = str(candidate.get("action", "")).upper()
                decision = ActionDecision.model_validate(candidate)
                return ActionPlanResult(
                    decision,
                    PlannerMetadata(
                        source="llm",
                        model=self.model,
                        attempts=attempt,
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            except Exception as exc:
                last_error = str(exc)
        raise PlannerError(last_error or "action planner failed without a diagnostic")


class FallbackActionPlanner:
    def __init__(
        self,
        primary: ActionPlannerProtocol,
        fallback: ActionPlannerProtocol | None = None,
    ):
        self.primary = primary
        self.fallback = fallback or DeterministicActionPlanner()

    def decide(
        self, observation: ExperimentObservation, manifest: TemplateManifest
    ) -> ActionPlanResult:
        try:
            return self.primary.decide(observation, manifest)
        except Exception as exc:
            fallback = self.fallback.decide(observation, manifest)
            fallback.metadata = PlannerMetadata(
                source="deterministic",
                model=getattr(self.primary, "model", None),
                attempts=getattr(self.primary, "max_attempts", 1),
                error=str(exc),
            )
            return fallback


def build_action_planner(
    mode: str | None = None,
    seed: int | None = None,
    allow_code_optimization: bool = True,
) -> ActionPlannerProtocol:
    selected = (
        (
            mode
            or os.getenv("AUTOEXP_ACTION_PLANNER_MODE")
            or os.getenv("AUTOEXP_PLANNER_MODE", "deterministic")
        )
        .strip()
        .lower()
    )
    if selected == "deterministic":
        return DeterministicActionPlanner()
    if selected == "llm":
        return FallbackActionPlanner(
            LLMActionPlanner(seed=seed, allow_code_optimization=allow_code_optimization)
        )
    if selected == "auto":
        return (
            FallbackActionPlanner(
                LLMActionPlanner(
                    seed=seed, allow_code_optimization=allow_code_optimization
                )
            )
            if os.getenv("OPENAI_API_KEY")
            else DeterministicActionPlanner()
        )
    raise ValueError("action planner mode must be deterministic, llm, or auto")


__all__ = [
    "ActionPlanResult",
    "ActionPlannerProtocol",
    "DeterministicActionPlanner",
    "FallbackActionPlanner",
    "LLMActionPlanner",
    "build_action_planner",
]
