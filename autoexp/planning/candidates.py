from __future__ import annotations

from typing import Any

from autoexp.domain import ParameterRange, TemplateManifest


_LEVELS = (0.25, 0.75, 0.5, 0.1, 0.9)


def _baseline_parameters(
    manifest: TemplateManifest,
    fixed_parameters: dict[str, Any],
    active_space: dict[str, ParameterRange],
) -> dict[str, Any]:
    parameters = dict(manifest.baseline_parameter_values)
    parameters.update(
        {
            name: value
            for name, value in fixed_parameters.items()
            if name not in active_space
        }
    )
    return parameters


def _continuous_value(policy: ParameterRange, fraction: float, baseline: Any) -> Any:
    if policy.choices:
        return policy.choices[0]
    if policy.type == "bool":
        return bool(round(fraction))
    if policy.min is None or policy.max is None:
        return baseline
    value = float(policy.min) + (float(policy.max) - float(policy.min)) * fraction
    if policy.type == "int":
        return int(round(value))
    if policy.type == "float":
        return float(value)
    return baseline


def deterministic_candidates(
    manifest: TemplateManifest,
    fixed_parameters: dict[str, Any] | None = None,
    search_space: dict[str, ParameterRange] | None = None,
    count: int = 1,
) -> list[dict[str, Any]]:
    """Build a bounded, reproducible candidate sequence for offline search.

    The first candidate is always the server-owned weak baseline. Later
    candidates cover registered continuous ranges as well as categorical
    choices, so an API outage does not collapse a multi-Trial run to one Trial.
    """

    fixed = fixed_parameters or {}
    active_space = search_space or manifest.parameter_policy
    baseline = _baseline_parameters(manifest, fixed, active_space)
    candidates = [dict(baseline)]
    names = [name for name in manifest.parameter_policy if name in active_space]
    requested = max(0, int(count))
    for ordinal in range(requested):
        candidate = dict(baseline)
        for dimension, name in enumerate(names):
            policy = active_space[name]
            baseline_value = baseline.get(name)
            if policy.choices:
                values = list(policy.choices)
                value = values[(ordinal + dimension + 1) % len(values)]
                if value == baseline_value and len(values) > 1:
                    value = values[(ordinal + dimension + 2) % len(values)]
            else:
                fraction = _LEVELS[(ordinal + dimension) % len(_LEVELS)]
                value = _continuous_value(policy, fraction, baseline_value)
                if value == baseline_value:
                    for fallback_fraction in _LEVELS:
                        value = _continuous_value(
                            policy, fallback_fraction, baseline_value
                        )
                        if value != baseline_value:
                            break
            candidate[name] = value
        candidates.append(candidate)
    return candidates


__all__ = ["deterministic_candidates"]
