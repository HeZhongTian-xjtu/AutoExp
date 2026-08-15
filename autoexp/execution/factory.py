from __future__ import annotations

import os

from .base import Executor
from .docker import DockerExecutor
from .local import LocalExecutor


def build_executor(mode: str | None = None, image: str | None = None) -> Executor:
    selected = (mode or os.getenv("AUTOEXP_EXECUTOR", "local")).strip().lower()
    if selected == "local":
        return LocalExecutor()
    if selected == "docker":
        return DockerExecutor(image=image)
    raise ValueError("executor mode must be local or docker")
