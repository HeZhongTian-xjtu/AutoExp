from __future__ import annotations

import os
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from uuid import UUID

from .base import ClassifiedError, ExecutionRequest, ExecutionResult


SAFE_EXECUTION_ENVIRONMENT = {
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "AUTOEXP_RUN_ID",
    "AUTOEXP_TRIAL_ID",
    "AUTOEXP_PROGRESS_PATH",
}


class LocalExecutor:
    """Development executor; it is not a security sandbox."""

    is_local = True

    def __init__(self, max_output_chars: int = 200_000):
        self.max_output_chars = max_output_chars
        self._processes: dict[UUID, subprocess.Popen[str]] = {}

    @staticmethod
    def _write_output(path: Path, content: str, limit: int) -> bool:
        truncated = len(content) > limit
        path.write_text(content[:limit], encoding="utf-8", errors="replace")
        return truncated

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        workspace = request.workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        output_dir = workspace / request.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        started_at = datetime.now(timezone.utc)
        immutable_before = _immutable_fingerprints(workspace, request.immutable_paths)
        env = _safe_child_environment()
        env.update(
            {
                key: value
                for key, value in request.environment.items()
                if key in SAFE_EXECUTION_ENVIRONMENT
            }
        )

        process: subprocess.Popen[str] | None = None
        stdout = ""
        stderr = ""
        try:
            process = subprocess.Popen(
                request.command,
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            self._processes[request.trial_id] = process
            stdout, stderr = process.communicate(timeout=request.timeout_seconds)
            exit_code = process.returncode
            status = "succeeded" if exit_code == 0 else "failed"
            error = (
                None
                if status == "succeeded"
                else ClassifiedError(
                    code="PROCESS_FAILED",
                    message=f"process exited with code {exit_code}",
                )
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            if process is not None:
                process.kill()
                tail_stdout, tail_stderr = process.communicate()
                stdout += _coerce_output(tail_stdout)
                stderr += _coerce_output(tail_stderr)
            exit_code = None
            status = "timeout"
            error = ClassifiedError(
                code="TIMEOUT", message=f"execution exceeded {request.timeout_seconds}s"
            )
        except OSError as exc:
            exit_code = None
            status = "failed"
            error = ClassifiedError(code="PROCESS_FAILED", message=str(exc))
        finally:
            self._processes.pop(request.trial_id, None)

        immutable_changes = _immutable_changes(workspace, immutable_before)
        if immutable_changes:
            status = "failed"
            error = ClassifiedError(
                code="IMMUTABLE_ASSET_MODIFIED",
                message="the experiment process modified a registered immutable asset",
                details={"paths": ", ".join(immutable_changes)},
            )
            stderr += "\n".join(
                f"IMMUTABLE_ASSET_MODIFIED: {path}" for path in immutable_changes
            )

        finished_at = datetime.now(timezone.utc)
        stdout_truncated = self._write_output(
            stdout_path, stdout, self.max_output_chars
        )
        stderr_truncated = self._write_output(
            stderr_path, stderr, self.max_output_chars
        )
        return ExecutionResult(
            status=status,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_truncated=stdout_truncated or stderr_truncated,
            resource_usage={
                "wall_seconds": max(0.0, (finished_at - started_at).total_seconds()),
                "executor": "local",
            },
            error=error,
        )

    def cancel(self, trial_id: UUID) -> bool:
        process = self._processes.get(trial_id)
        if process is None or process.poll() is not None:
            return False
        process.kill()
        return True


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(sys.getdefaultencoding(), errors="replace")
    return value


def _safe_child_environment() -> dict[str, str]:
    """Keep platform runtime variables while removing provider credentials."""
    secret_names = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in secret_names and not key.upper().endswith("_API_KEY")
    }
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _immutable_fingerprints(workspace: Path, paths: list[str]) -> dict[str, str]:
    return {
        relative: _sha256(workspace / relative)
        for relative in paths
        if (workspace / relative).is_file()
    }


def _immutable_changes(workspace: Path, before: dict[str, str]) -> list[str]:
    return [
        relative
        for relative, digest in before.items()
        if _sha256(workspace / relative) != digest
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
