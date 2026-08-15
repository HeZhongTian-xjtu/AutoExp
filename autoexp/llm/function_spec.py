from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionSpec:
    """Small provider-neutral tool schema used by AutoExp's structured calls."""

    name: str
    json_schema: dict[str, Any]
    description: str

    @property
    def as_openai_tool_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name}}
