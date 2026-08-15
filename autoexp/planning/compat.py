from __future__ import annotations

import os
from typing import Any, Callable

from autoexp.llm import query as _llm_query

from .planner import (
    DeterministicPlanner,
    LLMStructuredPlanner as BaseLLMStructuredPlanner,
    PlanResult,
    PlannerError,
    PlannerMetadata,
    PlannerProtocol,
)


def _bounded_openai_query(**kwargs: Any) -> Any:
    """Backward-compatible name for AutoExp's shared LLM gateway."""
    return _llm_query(**kwargs)


class LLMStructuredPlanner(BaseLLMStructuredPlanner):
    def __init__(
        self,
        model: str | None = None,
        query_fn: Callable[..., Any] | None = None,
        max_attempts: int = 2,
    ):
        super().__init__(
            model=model,
            query_fn=query_fn or _bounded_openai_query,
            max_attempts=max_attempts,
        )


def build_planner(mode: str | None = None) -> PlannerProtocol:
    selected = (
        (mode or os.getenv("AUTOEXP_PLANNER_MODE", "deterministic")).strip().lower()
    )
    if selected == "deterministic":
        return DeterministicPlanner()
    if selected == "llm":
        return LLMStructuredPlanner()
    if selected == "auto":
        return (
            LLMStructuredPlanner()
            if os.getenv("OPENAI_API_KEY")
            else DeterministicPlanner()
        )
    raise ValueError("planner mode must be deterministic, llm, or auto")


__all__ = [
    "DeterministicPlanner",
    "LLMStructuredPlanner",
    "PlanResult",
    "PlannerError",
    "PlannerMetadata",
    "PlannerProtocol",
    "build_planner",
]
