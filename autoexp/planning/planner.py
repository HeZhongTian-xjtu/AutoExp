from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from autoexp.domain import BudgetSpec, ExperimentSpec, MetricSpec, TemplateManifest
from autoexp.llm import FunctionSpec, query as autoexp_query
from autoexp.domain.policies import validate_experiment_spec


class PlannerError(RuntimeError):
    """Raised when a planner cannot produce a policy-valid experiment spec."""


class PlannerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["deterministic", "llm", "random", "optuna"]
    model: str | None = None
    attempts: int = Field(default=1, ge=1)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class PlannerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=2_000)
    hypothesis: str = Field(min_length=1, max_length=2_000)
    max_trials: int = Field(ge=1, le=8)
    seed: int = Field(ge=0, le=2**31 - 1)


class PlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ExperimentSpec
    metadata: PlannerMetadata


class PlannerProtocol(Protocol):
    def plan(
        self,
        objective: str,
        hypothesis: str,
        manifest: TemplateManifest,
        max_trials: int,
        seed: int = 42,
    ) -> PlanResult: ...


def _normalize_search_space(
    candidate: dict[str, Any],
    fixed_parameters: dict[str, Any],
    manifest: TemplateManifest,
) -> dict[str, Any]:
    """Accept either range objects or model-selected scalar parameter values."""
    raw = candidate.get("search_space")
    if raw is None:
        return {
            name: policy.model_dump(mode="json")
            for name, policy in manifest.parameter_policy.items()
        }
    if not isinstance(raw, dict):
        raise PlannerError("planner search_space must be a JSON object")
    normalized: dict[str, Any] = {}
    for name, value in raw.items():
        policy = manifest.parameter_policy.get(name)
        if policy is None:
            normalized[name] = value
            continue
        if isinstance(value, dict):
            normalized[name] = value
        else:
            if not policy.accepts(value):
                raise PlannerError(
                    f"planner selected an invalid value for {name}: {value!r}"
                )
            fixed_parameters.setdefault(name, value)
            normalized[name] = policy.model_dump(mode="json")
    for name, policy in manifest.parameter_policy.items():
        normalized.setdefault(name, policy.model_dump(mode="json"))
    return normalized


def _server_owned_spec(
    candidate: dict[str, Any],
    request: PlannerRequest,
    manifest: TemplateManifest,
) -> ExperimentSpec:
    metric_data = candidate.get("metric") or {
        "name": manifest.metric_name,
        "direction": manifest.metric_direction,
    }
    metric = MetricSpec.model_validate(metric_data)
    if (
        metric.name != manifest.metric_name
        or metric.direction != manifest.metric_direction
    ):
        raise PlannerError(
            f"planner must use the registered {manifest.metric_name} {manifest.metric_direction} metric"
        )

    # The first Trial is a controlled baseline, even when the LLM returns a
    # different scalar in fixed_parameters. Optimization remains available
    # through the validated search space and later Action Planner decisions.
    fixed_parameters = dict(manifest.baseline_parameter_values)
    search_space = _normalize_search_space(candidate, fixed_parameters, manifest)
    spec = ExperimentSpec(
        objective=str(candidate.get("objective") or request.objective),
        hypothesis=str(candidate.get("hypothesis") or request.hypothesis),
        template_id=manifest.template_id,
        dataset_id=manifest.dataset_id,
        model_id=manifest.allowed_models[0],
        metric=metric,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
        allowed_files=["configs/experiment.yaml"],
        budget=BudgetSpec(
            max_trials=request.max_trials,
            max_repairs_per_trial=int(
                manifest.resource_policy.get("max_repairs_per_trial", 2)
            ),
            timeout_seconds=int(manifest.resource_policy.get("timeout_seconds", 120)),
            max_total_seconds=int(
                manifest.resource_policy.get("max_total_seconds", 3_600)
            ),
        ),
        seed=request.seed,
    )
    issues = validate_experiment_spec(spec, manifest)
    if issues:
        rendered = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise PlannerError(f"planner output violates template policy: {rendered}")
    return spec


def _planner_function_spec(manifest: TemplateManifest) -> FunctionSpec:
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
        if policy.default is not None:
            schema["default"] = policy.default
        parameter_properties[name] = schema
    return FunctionSpec(
        name="propose_experiment_spec",
        description="Return a structured experiment proposal using only registered template parameters.",
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objective": {"type": "string"},
                "hypothesis": {"type": "string"},
                "metric": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "enum": [manifest.metric_name]},
                        "direction": {
                            "type": "string",
                            "enum": [manifest.metric_direction],
                        },
                    },
                    "required": ["name", "direction"],
                },
                "fixed_parameters": {"type": "object", "additionalProperties": True},
                "search_space": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": parameter_properties,
                },
            },
            "required": [
                "objective",
                "hypothesis",
                "metric",
                "fixed_parameters",
                "search_space",
            ],
        },
    )


def _parse_json_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        candidate = value
    elif isinstance(value, str):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            value.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlannerError(
                f"planner response is not valid JSON: {exc.msg}"
            ) from exc
    else:
        raise PlannerError(
            f"planner returned unsupported output type: {type(value).__name__}"
        )
    if not isinstance(candidate, dict):
        raise PlannerError("planner response must be a JSON object")
    return (
        candidate.get("experiment_spec", candidate)
        if isinstance(candidate.get("experiment_spec"), dict)
        else candidate
    )


class DeterministicPlanner:
    """No-cost planner used for local tests and offline development."""

    def plan(
        self,
        objective: str,
        hypothesis: str,
        manifest: TemplateManifest,
        max_trials: int,
        seed: int = 42,
    ) -> PlanResult:
        request = PlannerRequest(
            objective=objective.strip()
            or manifest.default_objective
            or f"Optimize the registered {manifest.metric_name} metric.",
            hypothesis=hypothesis.strip()
            or manifest.default_hypothesis
            or f"Registered parameter changes improve {manifest.metric_name}.",
            max_trials=max_trials,
            seed=seed,
        )
        spec = _server_owned_spec(
            {
                "objective": request.objective,
                "hypothesis": request.hypothesis,
                "metric": {
                    "name": manifest.metric_name,
                    "direction": manifest.metric_direction,
                },
                "fixed_parameters": dict(manifest.default_parameters),
                "search_space": {
                    name: policy.model_dump(mode="json")
                    for name, policy in manifest.parameter_policy.items()
                },
            },
            request,
            manifest,
        )
        return PlanResult(spec=spec, metadata=PlannerMetadata(source="deterministic"))


class LLMStructuredPlanner:
    """Generate a structured plan through AutoExp's OpenAI-compatible gateway."""

    def __init__(
        self,
        model: str | None = None,
        query_fn: Callable[..., Any] | None = None,
        max_attempts: int = 2,
    ):
        load_dotenv(override=True)
        self.model = model or os.getenv("AUTOEXP_PLANNER_MODEL", "deepseek-v4-flash")
        self.query_fn = query_fn or autoexp_query
        self.max_attempts = max(1, max_attempts)

    def plan(
        self,
        objective: str,
        hypothesis: str,
        manifest: TemplateManifest,
        max_trials: int,
        seed: int = 42,
    ) -> PlanResult:
        request = PlannerRequest(
            objective=objective.strip()
            or manifest.default_objective
            or f"Optimize the registered {manifest.metric_name} metric.",
            hypothesis=hypothesis.strip()
            or manifest.default_hypothesis
            or f"Registered parameter changes improve {manifest.metric_name}.",
            max_trials=max_trials,
            seed=seed,
        )
        function_spec = _planner_function_spec(manifest)
        system_message = (
            "You are the planning component of a controlled AutoML experiment agent. "
            "Return only a function call containing a valid experiment proposal. "
            "Use only registered parameters. Do not invent datasets, models, metrics, files, shell commands, or dependencies."
        )
        base_prompt = self._user_prompt(request, manifest)
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            prompt = base_prompt
            if last_error:
                prompt += f"\nThe previous proposal was rejected: {last_error}\nReturn a corrected proposal."
            try:
                raw = self.query_fn(
                    system_message=system_message,
                    user_message=prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=2_000,
                    func_spec=function_spec,
                )
                spec = _server_owned_spec(_parse_json_response(raw), request, manifest)
                return PlanResult(
                    spec=spec,
                    metadata=PlannerMetadata(
                        source="llm",
                        model=self.model,
                        attempts=attempt,
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            except Exception as exc:
                last_error = str(exc)
        raise PlannerError(last_error or "planner failed without a diagnostic")

    @staticmethod
    def _user_prompt(request: PlannerRequest, manifest: TemplateManifest) -> str:
        return json.dumps(
            {
                "objective": request.objective,
                "hypothesis": request.hypothesis,
                "template_id": manifest.template_id,
                "dataset_id": manifest.dataset_id,
                "registered_metric": {
                    "name": manifest.metric_name,
                    "direction": manifest.metric_direction,
                },
                "max_trials": request.max_trials,
                "seed": request.seed,
                "allowed_models": manifest.allowed_models,
                "initial_parameters": manifest.baseline_parameter_values,
                "baseline_parameters": manifest.baseline_parameter_values,
                "parameter_policy": {
                    name: policy.model_dump(mode="json")
                    for name, policy in manifest.parameter_policy.items()
                },
                "optimization_rule": (
                    "Keep fixed_parameters empty unless a parameter must remain fixed. "
                    "The first Trial uses the registered initial_parameters as a deliberately weak baseline; "
                    "later continuous optimization is handled by the Action Planner."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
