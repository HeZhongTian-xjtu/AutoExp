from __future__ import annotations

from typing import Any, Protocol


class ExperimentTracker(Protocol):
    def start_run(self, run: Any) -> None: ...
    def log_trial(self, run: Any, outcome: Any) -> None: ...
    def log_decision(self, run: Any, decision: Any) -> None: ...
    def log_artifact(self, run: Any, artifact: Any) -> None: ...
    def finish_run(self, run: Any) -> None: ...


class NullTracker:
    source = "none"

    def __init__(self, reason: str | None = None):
        self.reason = reason

    def start_run(self, run: Any) -> None:
        pass

    def log_trial(self, run: Any, outcome: Any) -> None:
        pass

    def log_decision(self, run: Any, decision: Any) -> None:
        pass

    def log_artifact(self, run: Any, artifact: Any) -> None:
        pass

    def finish_run(self, run: Any) -> None:
        pass
