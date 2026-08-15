from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from autoexp.domain import RepairResult, RepairSpec, TemplateManifest


class RepairError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AppliedPatch:
    result: RepairResult
    path: Path


def apply_unified_patch(
    root: Path, repair: RepairSpec, manifest: TemplateManifest
) -> AppliedPatch:
    """Apply one text-only unified diff inside a manifest allowlist."""
    lines = repair.patch.splitlines(keepends=True)
    if len(repair.patch) > 100_000:
        raise RepairError("PATCH_TOO_LARGE", "repair patch exceeds the maximum size")
    if (
        len(lines) < 3
        or not lines[0].startswith("--- ")
        or not lines[1].startswith("+++ ")
    ):
        raise RepairError(
            "PATCH_FORMAT", "repair must contain one unified diff file header"
        )

    old_path = _header_path(lines[0][4:])
    new_path = _header_path(lines[1][4:])
    target_file = _safe_relative_path(repair.target_file)
    if old_path != target_file or new_path != target_file:
        raise RepairError(
            "PATCH_TARGET_MISMATCH",
            "repair target_file must match both unified diff headers",
        )
    if target_file not in manifest.patchable_files:
        raise RepairError(
            "PATCH_FILE_SCOPE",
            f"repair file is outside the mutable manifest boundary: {target_file}",
        )

    path = (root / target_file).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents or not path.is_file():
        raise RepairError(
            "PATCH_FILE_MISSING", f"repair target does not exist: {target_file}"
        )
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    base_sha256 = _digest(path)
    if repair.expected_base_sha256 and repair.expected_base_sha256 != base_sha256:
        raise RepairError(
            "PATCH_BASE_MISMATCH", "repair base hash does not match the current file"
        )

    output: list[str] = []
    cursor = 0
    index = 2
    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    while index < len(lines):
        match = hunk_pattern.match(lines[index])
        if not match:
            raise RepairError(
                "PATCH_HUNK_FORMAT",
                f"invalid unified diff hunk header: {lines[index].strip()}",
            )
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        source_lines: list[str] = []
        replacement_lines: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith("\\ No newline at end of file"):
                index += 1
                continue
            if not line or line[0] not in " +-":
                raise RepairError(
                    "PATCH_HUNK_LINE", "unified diff contains an invalid hunk line"
                )
            content = line[1:]
            if line[0] in " -":
                source_lines.append(content)
            if line[0] in " +":
                replacement_lines.append(content)
            index += 1
        target_index = max(0, old_start - 1)
        if target_index < cursor or target_index + old_count > len(original):
            raise RepairError(
                "PATCH_CONTEXT", "repair hunk points outside the target file"
            )
        if original[target_index : target_index + old_count] != source_lines:
            raise RepairError(
                "PATCH_CONTEXT", "repair hunk context does not match the target file"
            )
        output.extend(original[cursor:target_index])
        output.extend(replacement_lines)
        cursor = target_index + old_count

    output.extend(original[cursor:])
    temporary = path.with_name(f".{path.name}.repair.tmp")
    temporary.write_text("".join(output), encoding="utf-8")
    temporary.replace(path)
    patched_sha256 = _digest(path)
    return AppliedPatch(
        result=RepairResult(
            accepted=True,
            target_file=target_file,
            base_sha256=base_sha256,
            patched_sha256=patched_sha256,
            preflight_passed=False,
        ),
        path=path,
    )


def _header_path(value: str) -> str:
    return _safe_relative_path(
        value.split("\t", 1)[0].strip().removeprefix("a/").removeprefix("b/")
    )


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or "\\" in value or ".." in path.parts:
        raise RepairError("PATCH_PATH", f"unsafe repair path: {value}")
    return path.as_posix()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
