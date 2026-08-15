from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def profile_dataset(
    root: Path | str,
    max_rows: int = 8,
    include_preview: bool = False,
) -> dict[str, Any]:
    """Return a bounded, JSON-serializable preview for planner/code context.

    The profile contains metadata only. It never uploads data or copies the
    dataset into the LLM request automatically; callers decide whether the
    bounded profile is appropriate for a provider.
    """

    path = Path(root).resolve()
    if not path.is_dir():
        return {"root": str(path), "exists": False, "files": []}
    files: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        item: dict[str, Any] = {
            "path": relative,
            "size_bytes": file_path.stat().st_size,
        }
        if file_path.suffix.lower() == ".csv":
            item.update(
                _profile_csv(
                    file_path, max_rows=max_rows, include_preview=include_preview
                )
            )
        elif file_path.suffix.lower() == ".json":
            item.update(_profile_json(file_path))
        files.append(item)
    return {"root": str(path), "exists": True, "files": files}


def _profile_csv(path: Path, max_rows: int, include_preview: bool) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            preview = []
            row_count = 0
            null_counts = {column: 0 for column in columns}
            for row in reader:
                row_count += 1
                for column in columns:
                    if not str(row.get(column, "")).strip():
                        null_counts[column] += 1
                if include_preview and len(preview) < max_rows:
                    preview.append({column: row.get(column) for column in columns})
        return {
            "format": "csv",
            "rows": row_count,
            "columns": columns,
            "null_counts": null_counts,
            "preview": preview,
        }
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"format": "csv", "error": str(exc)}


def _profile_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            shape: Any = {"kind": "list", "length": len(payload)}
        elif isinstance(payload, dict):
            shape = {"kind": "object", "keys": sorted(payload)[:50]}
        else:
            shape = {"kind": type(payload).__name__}
        return {"format": "json", "shape": shape}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"format": "json", "error": str(exc)}
