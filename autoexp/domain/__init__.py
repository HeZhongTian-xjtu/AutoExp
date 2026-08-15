"""Domain contracts that are independent from UI, LLM and executor details."""

from importlib import import_module

from .actions import ActionDecision, DecisionRecord
from .datasets import DatasetFile, DatasetRecord
from .errors import ValidationIssue
from .models import (
    BudgetSpec,
    ExperimentSpec,
    MetricSpec,
    ParameterRange,
    StopConditions,
    TemplateManifest,
)
from .observations import ExperimentObservation, TrialObservation
from .repairs import RepairResult, RepairSpec


_LAZY_EXPORTS = {
    "ExperimentRun": (".runs", "ExperimentRun"),
    "RunEvent": (".runs", "RunEvent"),
    "TrialOutcome": (".trials", "TrialOutcome"),
}


def __getattr__(name: str):
    lazy_target = _LAZY_EXPORTS.get(name)
    if lazy_target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = lazy_target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ActionDecision",
    "BudgetSpec",
    "DecisionRecord",
    "DatasetFile",
    "DatasetRecord",
    "ExperimentObservation",
    "ExperimentRun",
    "ExperimentSpec",
    "MetricSpec",
    "ParameterRange",
    "RepairResult",
    "RepairSpec",
    "RunEvent",
    "StopConditions",
    "TemplateManifest",
    "TrialOutcome",
    "TrialObservation",
    "ValidationIssue",
]
