from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


_TRACEBACK_FILE = re.compile(r'File ["\'](.+?)["\'], line (\d+)')


def collect_failure_context(
    outcome: Any,
    manifest: Any,
    recent_decisions: Iterable[Any] = (),
    max_chars: int = 6_000,
) -> dict[str, Any]:
    """Build a bounded, structured context for a repair-capable Agent."""
    workspace = Path(outcome.workspace)
    chunks: list[str] = []
    sources: list[str] = []
    executions = []
    if outcome.execution:
        executions.extend(
            (
                ("stdout", outcome.execution.stdout_path),
                ("stderr", outcome.execution.stderr_path),
            )
        )
    if getattr(outcome, "evaluation_execution", None):
        executions.extend(
            (
                ("evaluator_stdout", outcome.evaluation_execution.stdout_path),
                ("evaluator_stderr", outcome.evaluation_execution.stderr_path),
            )
        )
    if executions:
        for label, path in executions:
            text = _tail(Path(path), 2_000)
            if text:
                chunks.append(f"[{label}]\n{text}")
                sources.append(label)
    if outcome.validation:
        for gate in outcome.validation.gates:
            for label, path in (
                (f"{gate.name}_stdout", gate.stdout_path),
                (f"{gate.name}_stderr", gate.stderr_path),
            ):
                text = _tail(Path(path), 1_500) if path else ""
                if text:
                    chunks.append(f"[{label}]\n{text}")
                    sources.append(label)

    error_text = "\n".join(issue.message for issue in outcome.issues)
    traceback_locations = _TRACEBACK_FILE.findall("\n".join(chunks))
    code_context: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path, line_text in traceback_locations:
        path = Path(raw_path)
        candidate = path if path.is_file() else workspace / path.name
        if not candidate.is_file() or str(candidate) in seen:
            continue
        seen.add(str(candidate))
        line_number = int(line_text)
        code_context.append(
            {
                "file": (
                    candidate.relative_to(workspace).as_posix()
                    if workspace in candidate.parents
                    else candidate.name
                ),
                "line": line_number,
                "source": _around(candidate, line_number, 8),
            }
        )

    config_path = workspace / "configs" / "experiment.yaml"
    config = _tail(config_path, 2_000)
    diffs = []
    for record in list(recent_decisions)[-3:]:
        if record.decision.repair:
            diffs.append(
                {
                    "target_file": record.decision.repair.target_file,
                    "patch": record.decision.repair.patch[:2_000],
                    "reason": record.decision.repair.reason,
                }
            )

    fingerprint_basis = "|".join(
        [
            getattr(issue, "code", "") + ":" + getattr(issue, "message", "")
            for issue in outcome.issues
        ]
        + [chunks[-1] if chunks else "", json_safe(config)]
    )
    fingerprint = hashlib.sha256(
        _normalize(fingerprint_basis).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "failure_codes": list(dict.fromkeys(issue.code for issue in outcome.issues)),
        "error_summary": error_text[:1_500],
        "logs": "\n\n".join(chunks)[:max_chars],
        "code_context": code_context[:3],
        "config": config,
        "recent_diffs": diffs,
        "sources": sources,
        "fingerprint": fingerprint,
    }


def _tail(path: Path, limit: int) -> str:
    try:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _around(path: Path, line_number: int, radius: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line_number - radius - 1)
        end = min(len(lines), line_number + radius)
        return "\n".join(f"{index + 1}: {lines[index]}" for index in range(start, end))
    except OSError:
        return ""


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def json_safe(value: str) -> str:
    return value.replace("\r", "").replace("\n", "\n")[:2_000]
