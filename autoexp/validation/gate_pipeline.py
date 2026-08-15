from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import UUID

from autoexp.domain.errors import ValidationIssue
from autoexp.domain.models import TemplateManifest
from autoexp.execution import ExecutionRequest, Executor, ResourceLimits

from .models import GateResult, ValidationReport


class ValidationPipeline:
    """Run registered Pytest and Smoke commands before a formal Trial."""

    def __init__(self, executor: Executor, manifest: TemplateManifest):
        self.executor = executor
        self.manifest = manifest

    def run(
        self,
        workspace: Path,
        run_id: UUID,
        trial_id: UUID,
        timeout_seconds: int,
    ) -> ValidationReport:
        configured = self.manifest.validation or {}
        gates: list[GateResult] = []
        for name in ("pytest", "smoke"):
            raw_command = configured.get(f"{name}_command")
            command = (
                [str(item) for item in raw_command]
                if isinstance(raw_command, list)
                else []
            )
            if not command:
                now = datetime.now(timezone.utc)
                gates.append(
                    GateResult(
                        name=name,
                        status="skipped",
                        command=[],
                        started_at=now,
                        finished_at=now,
                    )
                )
                continue
            gate_timeout = int(
                configured.get(f"{name}_timeout_seconds", timeout_seconds)
            )
            gate = self._run_gate(
                name, command, workspace, run_id, trial_id, gate_timeout
            )
            gates.append(gate)
            if gate.status == "failed":
                break
        return ValidationReport(gates=gates)

    def _run_gate(
        self,
        name: str,
        command: list[str],
        workspace: Path,
        run_id: UUID,
        trial_id: UUID,
        timeout_seconds: int,
    ) -> GateResult:
        started_at = datetime.now(timezone.utc)
        if (
            getattr(self.executor, "is_local", False)
            and command
            and command[0].lower() in {"python", "python.exe"}
        ):
            command = [sys.executable, *command[1:]]
        request = ExecutionRequest(
            run_id=run_id,
            trial_id=trial_id,
            workspace=workspace,
            command=command,
            timeout_seconds=max(1, timeout_seconds),
            environment={
                "PYTHONHASHSEED": "0",
                "AUTOEXP_PROGRESS_PATH": (
                    str(workspace / "working" / "progress.json")
                    if getattr(self.executor, "is_local", False)
                    else "/workspace/working/progress.json"
                ),
            },
            resource_limits=ResourceLimits(
                cpu_count=int(self.manifest.resource_policy.get("cpu_count", 2)),
                memory_mb=int(self.manifest.resource_policy.get("memory_mb", 2048)),
                pids_limit=int(self.manifest.resource_policy.get("pids_limit", 128)),
            ),
            network_policy="none",
            output_subdir=f"working/gates/{name}",
            immutable_paths=list(self.manifest.immutable_files),
        )
        execution = self.executor.execute(request)
        issues: list[ValidationIssue] = []
        if execution.status != "succeeded":
            error = execution.error
            issues.append(
                ValidationIssue(
                    code=f"{name.upper()}_FAILED",
                    phase=f"{name}_gate",
                    message=error.message if error else f"{name} gate failed",
                    suggestion=f"Inspect the {name} stdout and stderr before attempting a repair.",
                    details={"error_code": error.code if error else "UNKNOWN"},
                )
            )
        required_outputs = self.manifest.validation.get(f"{name}_required_outputs", [])
        for relative_path in (
            required_outputs if isinstance(required_outputs, list) else []
        ):
            output = workspace / str(relative_path)
            if not output.is_file():
                issues.append(
                    ValidationIssue(
                        code=f"{name.upper()}_OUTPUT_MISSING",
                        phase=f"{name}_gate",
                        file=str(relative_path),
                        message=f"{name} did not produce required output: {relative_path}",
                        suggestion="Make the template produce the registered output contract.",
                    )
                )
        return GateResult(
            name=name,
            status="passed" if not issues else "failed",
            command=command,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            exit_code=execution.exit_code,
            stdout_path=execution.stdout_path,
            stderr_path=execution.stderr_path,
            execution=execution,
            issues=issues,
        )
