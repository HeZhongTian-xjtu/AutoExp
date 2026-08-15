from __future__ import annotations

import os

from .base import ExperimentTracker, NullTracker
from .mlflow_tracker import MLflowTracker


def build_tracker(
    mode: str | None = None, artifact_root: str | None = None
) -> ExperimentTracker:
    selected = (mode or os.getenv("AUTOEXP_TRACKER", "none")).strip().lower()
    if selected in {"none", "sqlite", "disabled"}:
        return NullTracker()
    if selected == "mlflow":
        try:
            return MLflowTracker(artifact_root=artifact_root)
        except Exception as exc:
            return NullTracker(f"MLflow tracker unavailable: {exc}")
    raise ValueError("tracker mode must be none or mlflow")


__all__ = ["ExperimentTracker", "MLflowTracker", "NullTracker", "build_tracker"]
