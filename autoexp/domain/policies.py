from __future__ import annotations

from typing import Any

from .errors import ValidationIssue
from .models import ExperimentSpec, TemplateManifest


def validate_experiment_spec(
    spec: ExperimentSpec, manifest: TemplateManifest
) -> list[ValidationIssue]:
    """Validate an experiment against server-owned template rules."""
    issues = []
    if (
        spec.metric.name != manifest.metric_name
        or spec.metric.direction != manifest.metric_direction
    ):
        issues.append(
            ValidationIssue(
                code="PLAN_METRIC_NOT_ALLOWED",
                phase="plan_validation",
                message=f"metric must be {manifest.metric_name} ({manifest.metric_direction}) for this template",
                suggestion="Use the metric registered by the template manifest.",
            )
        )
    for item in spec.validate_against_manifest(manifest):
        issues.append(
            ValidationIssue(
                code=item["code"],
                phase="plan_validation",
                message=item["message"],
                suggestion="Regenerate the plan within the registered template policy.",
            )
        )
    if spec.model_id not in manifest.allowed_models:
        issues.append(
            ValidationIssue(
                code="PLAN_MODEL_NOT_ALLOWED",
                phase="plan_validation",
                message=f"model is not allowed by template: {spec.model_id}",
                suggestion="Use a model registered by the template manifest.",
            )
        )
    return issues


def validate_parameters(
    parameters: dict[str, Any], manifest: TemplateManifest
) -> list[ValidationIssue]:
    issues = []
    for name, value in parameters.items():
        policy = manifest.parameter_policy.get(name)
        if policy is None:
            issues.append(
                ValidationIssue(
                    code="PARAMETER_NOT_ALLOWED",
                    phase="parameter_validation",
                    message=f"parameter is not allowed: {name}",
                    suggestion="Choose a parameter from the template manifest.",
                )
            )
        elif not policy.accepts(value):
            issues.append(
                ValidationIssue(
                    code="PARAMETER_OUT_OF_RANGE",
                    phase="parameter_validation",
                    message=f"parameter value violates policy: {name}={value!r}",
                    suggestion="Use a value inside the registered parameter range.",
                )
            )
    return issues
