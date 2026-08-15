from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .function_spec import FunctionSpec


def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs: Any,
) -> Any:
    """Call an OpenAI-compatible endpoint used by AutoExp.

    DeepSeek, OpenAI, and compatible relay services can all be selected with
    OPENAI_BASE_URL and OPENAI_API_KEY. The gateway intentionally has no retry
    loop so the graph can record a single, bounded failure in its run state.
    """

    load_dotenv(override=True)
    timeout = float(os.getenv("AUTOEXP_PLANNER_TIMEOUT_SECONDS", "30"))
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        timeout=timeout,
        max_retries=0,
    )
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or 2_000,
        **model_kwargs,
    }
    request = {key: value for key, value in request.items() if value is not None}
    if func_spec is not None:
        request["tools"] = [func_spec.as_openai_tool_dict]
        request["tool_choice"] = func_spec.openai_tool_choice_dict
        if model.startswith("deepseek-"):
            request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    if message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)
    return message.content or ""
