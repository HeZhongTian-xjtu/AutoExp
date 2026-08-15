from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonValue = Any


class ParameterRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["int", "float", "str", "bool"]
    min: float | int | None = None
    max: float | int | None = None
    choices: list[JsonValue] | None = None
    default: JsonValue | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ParameterRange":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("parameter min must not be greater than max")
        if self.choices is not None and not self.choices:
            raise ValueError("parameter choices must not be empty")
        return self

    def accepts(self, value: JsonValue) -> bool:
        if self.type == "bool" and not isinstance(value, bool):
            return False
        if self.type == "int" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return False
        if self.type == "float" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return False
        if self.type == "str" and not isinstance(value, str):
            return False
        if self.choices is not None and value not in self.choices:
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.min is not None and value < self.min:
                return False
            if self.max is not None and value > self.max:
                return False
        return True


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    direction: Literal["minimize", "maximize"]


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trials: int = Field(default=4, ge=1, le=32)
    max_repairs_per_trial: int = Field(default=2, ge=0, le=5)
    timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    max_total_seconds: int = Field(default=3_600, ge=1, le=172_800)


class StopConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_metric: float | None = None
    patience: int = Field(default=2, ge=1, le=32)


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"path must be a safe relative POSIX path: {value}")
    return value


class TemplateManifest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    default_objective: str | None = None
    default_hypothesis: str | None = None
    dataset_id: str = "embedded-text-v1"
    metric_name: str = "macro_f1"
    metric_direction: Literal["minimize", "maximize"] = "maximize"
    default_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    # Deliberately weak starting point for every new Run. When omitted, the
    # legacy default_parameters remain the baseline for backward compatibility.
    baseline_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    dataset_manifest: str | None = None
    builtin_dataset_path: str | None = None
    dataset_contract: dict[str, JsonValue] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    template_id: str = Field(min_length=1, max_length=100)
    entrypoint: str
    evaluator_entrypoint: str | None = None
    metric_file: str
    allowed_files: list[str] = Field(min_length=1)
    # ``allowed_files`` is retained for manifest compatibility. New manifests
    # must use ``mutable_files`` as the Patch boundary and ``immutable_files``
    # for data, evaluators, and other trusted template assets.
    mutable_files: list[str] = Field(default_factory=list)
    immutable_files: list[str] = Field(default_factory=list)
    allowed_imports: list[str] = Field(default_factory=list)
    allowed_models: list[str] = Field(default_factory=list)
    parameter_policy: dict[str, ParameterRange] = Field(default_factory=dict)
    resource_policy: dict[str, JsonValue] = Field(default_factory=dict)
    validation: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "entrypoint",
        "evaluator_entrypoint",
        "metric_file",
        "builtin_dataset_path",
        mode="before",
    )
    @classmethod
    def validate_single_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_relative_path(value)

    @field_validator("allowed_files")
    @classmethod
    def validate_allowed_files(cls, values: list[str]) -> list[str]:
        return [_safe_relative_path(value) for value in values]

    @field_validator("mutable_files", "immutable_files")
    @classmethod
    def validate_file_lists(cls, values: list[str]) -> list[str]:
        return [_safe_relative_path(value) for value in values]

    @model_validator(mode="after")
    def validate_file_boundary(self) -> "TemplateManifest":
        mutable = set(self.mutable_files)
        immutable = set(self.immutable_files)
        if mutable & immutable:
            raise ValueError("mutable_files and immutable_files must be disjoint")
        declared = set(self.allowed_files)
        if mutable and not mutable.issubset(declared):
            raise ValueError("mutable_files must be included in allowed_files")
        if immutable and not immutable.issubset(declared):
            raise ValueError("immutable_files must be included in allowed_files")
        if self.evaluator_entrypoint and self.evaluator_entrypoint in mutable:
            raise ValueError("evaluator_entrypoint must not be mutable")
        return self

    @property
    def patchable_files(self) -> set[str]:
        """Return the explicit Patch boundary, with legacy fallback."""
        return set(self.mutable_files or self.allowed_files)

    @property
    def baseline_parameter_values(self) -> dict[str, JsonValue]:
        return dict(self.baseline_parameters or self.default_parameters)

    @classmethod
    def load(cls, path: Path | str) -> "TemplateManifest":
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})

    def environment_fingerprint(self, package_versions: dict[str, str | None]) -> str:
        payload = {
            "manifest": self.model_dump(mode="json"),
            "packages": dict(sorted(package_versions.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: UUID = Field(default_factory=uuid4)
    objective: str = Field(min_length=1, max_length=2_000)
    hypothesis: str = Field(min_length=1, max_length=2_000)
    template_id: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1, max_length=100)
    model_id: str = Field(min_length=1, max_length=100)
    metric: MetricSpec
    fixed_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    search_space: dict[str, ParameterRange] = Field(default_factory=dict)
    allowed_files: list[str] = Field(default_factory=list)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("allowed_files")
    @classmethod
    def validate_spec_files(cls, values: list[str]) -> list[str]:
        return [_safe_relative_path(value) for value in values]

    @model_validator(mode="after")
    def validate_parameter_names(self) -> "ExperimentSpec":
        unknown = (
            set(self.fixed_parameters)
            - set(self.search_space)
            - set(self.fixed_parameters)
        )
        if unknown:
            raise ValueError(f"unknown fixed parameters: {sorted(unknown)}")
        return self

    def validate_against_manifest(
        self, manifest: TemplateManifest
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if self.template_id != manifest.template_id:
            issues.append(
                {
                    "code": "PLAN_TEMPLATE_MISMATCH",
                    "message": "template_id is not registered",
                }
            )
        generated_files = {"configs/experiment.yaml"}
        if self.allowed_files and not set(self.allowed_files).issubset(
            manifest.patchable_files | generated_files
        ):
            issues.append(
                {
                    "code": "PLAN_FILE_SCOPE",
                    "message": "allowed_files exceed the mutable template scope",
                }
            )
        for name, policy in self.search_space.items():
            manifest_policy = manifest.parameter_policy.get(name)
            if manifest_policy is None:
                issues.append(
                    {
                        "code": "PLAN_PARAMETER_UNKNOWN",
                        "message": f"parameter is not allowed: {name}",
                    }
                )
            elif policy.type != manifest_policy.type:
                issues.append(
                    {
                        "code": "PLAN_PARAMETER_TYPE",
                        "message": f"parameter type mismatch: {name}",
                    }
                )
            else:
                if (
                    policy.min is not None
                    and manifest_policy.min is not None
                    and policy.min < manifest_policy.min
                ):
                    issues.append(
                        {
                            "code": "PLAN_PARAMETER_RANGE",
                            "message": f"parameter min exceeds policy: {name}",
                        }
                    )
                if (
                    policy.max is not None
                    and manifest_policy.max is not None
                    and policy.max > manifest_policy.max
                ):
                    issues.append(
                        {
                            "code": "PLAN_PARAMETER_RANGE",
                            "message": f"parameter max exceeds policy: {name}",
                        }
                    )
                if policy.choices is not None and manifest_policy.choices is not None:
                    if not set(policy.choices).issubset(manifest_policy.choices):
                        issues.append(
                            {
                                "code": "PLAN_PARAMETER_CHOICES",
                                "message": f"parameter choices exceed policy: {name}",
                            }
                        )
        for name, value in self.fixed_parameters.items():
            policy = manifest.parameter_policy.get(name)
            if policy is None or not policy.accepts(value):
                issues.append(
                    {
                        "code": "PLAN_FIXED_PARAMETER",
                        "message": f"invalid fixed parameter: {name}",
                    }
                )
        return issues
