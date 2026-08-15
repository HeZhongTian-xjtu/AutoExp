from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


from autoexp.domain.errors import ValidationIssue
from autoexp.domain.models import TemplateManifest


@dataclass(frozen=True)
class DatasetIntegrityReport:
    passed: bool
    dataset_id: str | None = None
    dataset_sha256: str | None = None
    issues: list[ValidationIssue] = field(default_factory=list)


def verify_dataset_manifest(
    workspace: Path, manifest: TemplateManifest
) -> DatasetIntegrityReport:
    """Verify the immutable dataset assets before any user-controlled code runs."""
    if not manifest.dataset_manifest:
        return DatasetIntegrityReport(passed=True)

    manifest_path = workspace / manifest.dataset_manifest
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _failed(
            "DATASET_MANIFEST_MISSING",
            f"dataset manifest does not exist: {manifest.dataset_manifest}",
        )
    except (OSError, ValueError) as exc:
        return _failed(
            "DATASET_MANIFEST_INVALID", f"dataset manifest is invalid: {exc}"
        )

    issues: list[ValidationIssue] = []
    dataset_id = payload.get("dataset_id")
    if dataset_id != manifest.dataset_id:
        issues.append(
            _issue(
                "DATASET_ID_MISMATCH",
                f"dataset id {dataset_id!r} does not match {manifest.dataset_id!r}",
            )
        )

    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        issues.append(
            _issue(
                "DATASET_FILES_MISSING",
                "dataset manifest must declare at least one file",
            )
        )
        return DatasetIntegrityReport(False, dataset_id, None, issues)

    digests: dict[str, str] = {}
    for relative_name, declaration in files.items():
        if not isinstance(declaration, dict) or not isinstance(
            declaration.get("sha256"), str
        ):
            issues.append(
                _issue(
                    "DATASET_FILE_DECLARATION_INVALID",
                    f"invalid fingerprint declaration for {relative_name}",
                )
            )
            continue
        path = (manifest_path.parent / str(relative_name)).resolve()
        if manifest_path.parent.resolve() not in path.parents:
            issues.append(
                _issue(
                    "DATASET_FILE_PATH_UNSAFE",
                    f"dataset file escapes the dataset directory: {relative_name}",
                )
            )
            continue
        if not path.is_file():
            issues.append(
                _issue(
                    "DATASET_FILE_MISSING",
                    f"dataset file does not exist: {relative_name}",
                )
            )
            continue
        digest = _sha256(path)
        digests[str(relative_name)] = digest
        if digest != declaration["sha256"]:
            issues.append(
                _issue(
                    "DATASET_FILE_FINGERPRINT_MISMATCH",
                    f"fingerprint mismatch for {relative_name}",
                )
            )
        expected_rows = declaration.get("rows_including_header")
        if expected_rows is not None and path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                actual_rows = sum(1 for _ in csv.reader(handle))
            if actual_rows != expected_rows:
                issues.append(
                    _issue(
                        "DATASET_ROW_COUNT_MISMATCH",
                        f"row count mismatch for {relative_name}",
                    )
                )

    train_declaration = files.get("train.csv")
    dataset_sha256 = payload.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or not dataset_sha256:
        issues.append(
            _issue(
                "DATASET_FINGERPRINT_MISSING",
                "dataset manifest must declare dataset_sha256",
            )
        )
    elif not isinstance(
        train_declaration, dict
    ) or dataset_sha256 != train_declaration.get("sha256"):
        issues.append(
            _issue(
                "DATASET_FINGERPRINT_CONTRACT",
                "dataset_sha256 must equal the declared train.csv fingerprint",
            )
        )
    elif digests.get("train.csv") != dataset_sha256:
        issues.append(
            _issue(
                "DATASET_FINGERPRINT_MISMATCH",
                "dataset_sha256 does not match the verified train.csv asset",
            )
        )

    return DatasetIntegrityReport(not issues, dataset_id, dataset_sha256, issues)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        phase="dataset_integrity",
        message=message,
        suggestion="Restore the registered immutable dataset assets before retrying.",
    )


def _failed(code: str, message: str) -> DatasetIntegrityReport:
    return DatasetIntegrityReport(False, issues=[_issue(code, message)])
