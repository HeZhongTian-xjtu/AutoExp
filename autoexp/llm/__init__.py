"""Provider-neutral LLM gateway used by the AutoExp agents."""

from .function_spec import FunctionSpec
from .gateway import query

__all__ = ["FunctionSpec", "query"]
