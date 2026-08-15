from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from .base import ClassifiedError, ExecutionRequest, ExecutionResult
from .local import (
    SAFE_EXECUTION_ENVIRONMENT,
    _coerce_output,
    _immutable_changes,
    _immutable_fingerprints,
    _safe_child_environment,
)


class DockerExecutor:
    """Run a Trial in a disposable, resource-limited Docker container.

    Docker is deliberately invoked through an argument list, never through a
    shell. The worker owning this executor should be the only service allowed
    to access the Docker Engine socket.
    """

    def __init__(self, image: str | None = None, max_output_chars: int = 200_000):
        self.image = image or os.getenv("AUTOEXP_DOCKER_IMAGE", "autoexp-runner:latest")
        self.max_output_chars = max_output_chars
        self._processes: dict[UUID, tuple[subprocess.Popen[str], str]] = {}

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        workspace = request.workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        output_dir = workspace / request.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        started_at = datetime.now(timezone.utc)
        immutable_before = _immutable_fingerprints(workspace, request.immutable_paths)

        if shutil.which("docker") is None:
            return self._failed(
                stdout_path,
                stderr_path,
                started_at,
                "DOCKER_UNAVAILABLE",
                "Docker CLI was not found on PATH.",
            )

        container_name = f'autoexp-{str(request.trial_id).replace("-", "")[:20]}'
        command = self._build_command(request, container_name)
        process: subprocess.Popen[str] | None = None
        stdout = ""
        stderr = ""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=_safe_child_environment(),
            )
            self._processes[request.trial_id] = (process, container_name)
            stdout, stderr = process.communicate(timeout=request.timeout_seconds)
            exit_code = process.returncode
            status = "succeeded" if exit_code == 0 else "failed"
            error = (
                None
                if status == "succeeded"
                else ClassifiedError(
                    code="CONTAINER_FAILED",
                    message=f"Docker container exited with code {exit_code}.",
                )
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            self._kill_container(container_name)
            if process is not None:
                tail_stdout, tail_stderr = process.communicate()
                stdout += _coerce_output(tail_stdout)
                stderr += _coerce_output(tail_stderr)
            exit_code = None
            status = "timeout"
            error = ClassifiedError(
                code="TIMEOUT",
                message=f"container execution exceeded {request.timeout_seconds}s",
            )
        except OSError as exc:
            exit_code = None
            status = "failed"
            error = ClassifiedError(code="DOCKER_START_FAILED", message=str(exc))
        finally:
            self._processes.pop(request.trial_id, None)
            self._remove_container(container_name)

        immutable_changes = _immutable_changes(workspace, immutable_before)
        if immutable_changes:
            status = "failed"
            error = ClassifiedError(
                code="IMMUTABLE_ASSET_MODIFIED",
                message="the container modified a registered immutable asset",
                details={"paths": ", ".join(immutable_changes)},
            )
            stderr += "\n".join(
                f"IMMUTABLE_ASSET_MODIFIED: {path}" for path in immutable_changes
            )

        finished_at = datetime.now(timezone.utc)
        stdout_truncated = self._write_output(stdout_path, stdout)
        stderr_truncated = self._write_output(stderr_path, stderr)
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
                "executor": "docker",
                "image": request.image or self.image,
                "cpu_count": request.resource_limits.cpu_count,
                "memory_mb": request.resource_limits.memory_mb,
                "pids_limit": request.resource_limits.pids_limit,
            },
            error=error,
        )

    def cancel(self, trial_id: UUID) -> bool:
        item = self._processes.get(trial_id)
        if item is None:
            return False
        process, container_name = item
        self._kill_container(container_name)
        if process.poll() is None:
            process.kill()
        return True

    def _build_command(
        self, request: ExecutionRequest, container_name: str
    ) -> list[str]:
        limits = request.resource_limits
        image = request.image or self.image
        command = list(request.command)
        if command and Path(command[0]).name.lower() in {"python", "python.exe"}:
            command[0] = "python"
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none" if request.network_policy == "none" else "bridge",
            "--cpus",
            str(limits.cpu_count),
            "--memory",
            f"{limits.memory_mb}m",
            "--pids-limit",
            str(limits.pids_limit),
            "--read-only",
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--user",
            "1000:1000",
            "--mount",
            f"type=bind,source={request.workspace.resolve()},target=/workspace",
            *self._immutable_mounts(request),
            "--workdir",
            "/workspace",
            *self._environment_flags(request),
            image,
            *command,
        ]

    @staticmethod
    def _environment_flags(request: ExecutionRequest) -> list[str]:
        allowed = SAFE_EXECUTION_ENVIRONMENT
        flags: list[str] = []
        for key, value in request.environment.items():
            if key not in allowed:
                continue
            flags.extend(["--env", f"{key}={value}"])
        return flags

    @staticmethod
    def _immutable_mounts(request: ExecutionRequest) -> list[str]:
        mounts: list[str] = []
        workspace = request.workspace.resolve()
        for relative in request.immutable_paths:
            source = (workspace / relative).resolve()
            if not source.is_file():
                continue
            target = "/workspace/" + relative.replace("\\", "/")
            mounts.extend(
                ["--mount", f"type=bind,source={source},target={target},readonly"]
            )
        return mounts

    def _kill_container(self, name: str) -> None:
        if shutil.which("docker"):
            subprocess.run(
                ["docker", "kill", name], capture_output=True, text=True, check=False
            )

    def _remove_container(self, name: str) -> None:
        if shutil.which("docker"):
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                check=False,
            )

    def _write_output(self, path: Path, content: str) -> bool:
        truncated = len(content) > self.max_output_chars
        path.write_text(
            content[: self.max_output_chars], encoding="utf-8", errors="replace"
        )
        return truncated

    @staticmethod
    def _failed(
        path_out: Path, path_err: Path, started_at: datetime, code: str, message: str
    ) -> ExecutionResult:
        path_out.write_text("", encoding="utf-8")
        path_err.write_text(message, encoding="utf-8")
        return ExecutionResult(
            status="failed",
            exit_code=None,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stdout_path=path_out,
            stderr_path=path_err,
            resource_usage={"executor": "docker"},
            error=ClassifiedError(code=code, message=message),
        )
