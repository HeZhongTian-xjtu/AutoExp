from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from autoexp.domain import DatasetFile, DatasetRecord, TemplateManifest
from autoexp.persistence import SQLiteRepository


class DatasetCatalogError(ValueError):
    pass


class DatasetCatalog:
    """Persistent dataset registry and safe staging boundary for AutoExp runs."""

    def __init__(
        self,
        project_root: Path | str,
        repository: SQLiteRepository | None = None,
        storage_root: Path | str | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.repository = repository or SQLiteRepository(
            self.project_root / "workspaces" / "autoexp.sqlite3"
        )
        self.storage_root = Path(
            storage_root or self.repository.path.parent / "datasets"
        ).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def ensure_builtin(
        self, template_root: Path, manifest: TemplateManifest
    ) -> DatasetRecord:
        data_root = self._builtin_data_root(template_root, manifest)
        files = self._inspect_files(data_root) if data_root.is_dir() else []
        existing = self.repository.load_dataset(manifest.dataset_id)
        if existing is not None:
            if (
                existing.source == "builtin"
                and files
                and (
                    existing.storage_path != str(data_root)
                    or not existing.files
                    or {item.sha256 for item in existing.files}
                    != {item.sha256 for item in files}
                )
            ):
                refreshed = existing.model_copy(
                    update={
                        "storage_path": str(data_root),
                        "files": files,
                        "status": "ready",
                        "validation_message": None,
                        "updated_at": _now(),
                    }
                )
                if not (data_root / "dataset_manifest.json").is_file():
                    self._write_builtin_manifest(data_root, manifest.dataset_id, files)
                self.repository.register_dataset(refreshed)
                return refreshed
            if existing.source == "builtin" and not files:
                unavailable = existing.model_copy(
                    update={
                        "storage_path": str(data_root),
                        "files": [],
                        "status": "invalid",
                        "validation_message": "No built-in dataset is bundled. Upload a compatible dataset before running.",
                        "updated_at": _now(),
                    }
                )
                self.repository.register_dataset(unavailable)
                return unavailable
            return existing
        if (
            data_root.is_dir()
            and files
            and not (data_root / "dataset_manifest.json").is_file()
        ):
            self._write_builtin_manifest(data_root, manifest.dataset_id, files)
        now = _now()
        record = DatasetRecord(
            dataset_id=manifest.dataset_id,
            display_name=self._display_name(manifest.dataset_id),
            source="builtin",
            task_type=self._task_type(manifest),
            compatible_template_ids=[manifest.template_id],
            storage_path=str(data_root) if data_root.is_dir() else None,
            files=files,
            target_column=str(manifest.dataset_contract.get("target_column") or "")
            or None,
            status="ready" if files else "invalid",
            validation_message=(
                None
                if files
                else "No built-in dataset is bundled. Upload a compatible dataset before running."
            ),
            created_at=now,
            updated_at=now,
        )
        self.repository.register_dataset(record)
        return record

    def list_for_template(
        self,
        template_root: Path,
        manifest: TemplateManifest,
    ) -> list[DatasetRecord]:
        self.ensure_builtin(template_root, manifest)
        return [
            record
            for record in self.repository.list_datasets()
            if record.is_compatible(manifest.template_id, self._task_type(manifest))
        ]

    def get_for_template(
        self,
        template_root: Path,
        manifest: TemplateManifest,
        dataset_id: str,
    ) -> tuple[DatasetRecord, TemplateManifest]:
        self.ensure_builtin(template_root, manifest)
        record = self.repository.load_dataset(dataset_id)
        if record is None:
            raise DatasetCatalogError(f"dataset is not registered: {dataset_id}")
        if not record.is_compatible(manifest.template_id, self._task_type(manifest)):
            raise DatasetCatalogError(
                f"dataset {dataset_id} is incompatible with template {manifest.template_id}"
            )
        return record, self.bind_manifest(manifest, record)

    def register_upload(
        self,
        template_root: Path,
        manifest: TemplateManifest,
        uploads: Iterable[tuple[str, bytes]],
        display_name: str | None = None,
    ) -> DatasetRecord:
        del template_root
        files_by_name = self._normalise_uploads(uploads)
        contract = manifest.dataset_contract
        required = [str(item) for item in contract.get("required_files", ["train.csv"])]
        if "train.csv" not in files_by_name:
            raise DatasetCatalogError("upload must contain a training CSV file")
        missing = [name for name in required if name not in files_by_name]
        if missing:
            raise DatasetCatalogError(
                f'missing required dataset files: {", ".join(missing)}'
            )

        train_columns, _ = _csv_shape(files_by_name["train.csv"])
        required_columns = [str(item) for item in contract.get("required_columns", [])]
        missing_columns = [
            column for column in required_columns if column not in train_columns
        ]
        if missing_columns:
            raise DatasetCatalogError(
                f'train.csv is missing required columns: {", ".join(missing_columns)}'
            )
        target_column = str(contract.get("target_column") or "") or None
        if target_column and target_column not in train_columns:
            raise DatasetCatalogError(
                f"train.csv must contain target column {target_column!r}"
            )
        text_column = str(contract.get("text_column") or "") or None
        if text_column and text_column not in train_columns:
            raise DatasetCatalogError(
                f"train.csv must contain text column {text_column!r}"
            )
        dataset_id = self._new_dataset_id(display_name or manifest.template_id)
        dataset_dir = self.storage_root / dataset_id / "data"
        dataset_dir.mkdir(parents=True, exist_ok=False)
        records: list[DatasetFile] = []
        for name, content in files_by_name.items():
            path = dataset_dir / name
            path.write_bytes(content)
            role = _file_role(name)
            columns, rows = (
                _csv_shape(content) if path.suffix.lower() == ".csv" else ([], None)
            )
            records.append(
                DatasetFile(
                    role=role,
                    name=name,
                    relative_path=name,
                    sha256=_sha256_bytes(content),
                    size_bytes=len(content),
                    rows_including_header=rows,
                    columns=columns,
                )
            )
        manifest_payload = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "dataset_sha256": next(
                item.sha256 for item in records if item.role == "train"
            ),
            "files": {
                item.relative_path: {
                    "sha256": item.sha256,
                    "rows_including_header": item.rows_including_header,
                }
                for item in records
            },
        }
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        now = _now()
        record = DatasetRecord(
            dataset_id=dataset_id,
            display_name=display_name or dataset_id,
            source="uploaded",
            task_type=self._task_type(manifest),
            compatible_template_ids=[manifest.template_id],
            storage_path=str(dataset_dir),
            files=records,
            target_column=target_column,
            created_at=now,
            updated_at=now,
        )
        self.repository.register_dataset(record)
        return record

    def stage(self, record: DatasetRecord, workspace: Path) -> None:
        data_root = workspace / "data"
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir(parents=True, exist_ok=True)
        if record.storage_path:
            source = Path(record.storage_path).resolve()
            allowed_roots = (self.project_root, self.storage_root)
            if not any(_is_relative_to(source, root) for root in allowed_roots):
                raise DatasetCatalogError(
                    "dataset storage must stay inside a configured AutoExp root"
                )
            if not source.is_dir():
                raise DatasetCatalogError(f"dataset storage does not exist: {source}")
            for item in source.iterdir():
                destination = data_root / item.name
                if item.is_dir():
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)

    @staticmethod
    def bind_manifest(
        manifest: TemplateManifest, record: DatasetRecord
    ) -> TemplateManifest:
        dataset_paths = {f"data/{item.relative_path}" for item in record.files}
        dataset_manifest = (
            "data/dataset_manifest.json"
            if record.file_for_role("train") is not None
            else manifest.dataset_manifest
        )
        if dataset_manifest:
            dataset_paths.add(dataset_manifest)
        return manifest.model_copy(
            update={
                "dataset_id": record.dataset_id,
                "dataset_manifest": dataset_manifest,
                "allowed_files": sorted(set(manifest.allowed_files) | dataset_paths),
                "immutable_files": sorted(
                    set(manifest.immutable_files) | dataset_paths
                ),
            }
        )

    def _normalise_uploads(
        self, uploads: Iterable[tuple[str, bytes]]
    ) -> dict[str, bytes]:
        raw_files = [
            (Path(original_name).name.strip(), bytes(content))
            for original_name, content in uploads
        ]
        csv_names = [
            name for name, _ in raw_files if Path(name).suffix.lower() == ".csv"
        ]
        train_source = next(
            (name for name in csv_names if _file_role(name) == "train"), None
        )
        train_source = train_source or (csv_names[0] if csv_names else None)
        test_source = next(
            (name for name in csv_names if _file_role(name) == "test"), None
        )
        if test_source is None and len(csv_names) == 2:
            test_source = next(
                (name for name in csv_names if name != train_source), None
            )
        result: dict[str, bytes] = {}
        for name, content in raw_files:
            suffix = Path(name).suffix.lower()
            if not name or suffix not in {".csv", ".txt", ".json", ".md"}:
                raise DatasetCatalogError(f"unsupported dataset file: {name}")
            if suffix == ".csv":
                target_name = (
                    "train.csv"
                    if name == train_source
                    else "test.csv" if name == test_source else name
                )
            elif suffix in {".txt", ".md"}:
                target_name = "data_description.txt"
            else:
                target_name = "metadata.json"
            if target_name in result:
                raise DatasetCatalogError(f"duplicate dataset file role: {target_name}")
            result[target_name] = bytes(content)
        return result

    def _new_dataset_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "uploaded-dataset"
        return f"{slug}-{uuid4().hex[:10]}"

    @staticmethod
    def _task_type(manifest: TemplateManifest) -> str:
        return str(manifest.dataset_contract.get("task_type") or "generic")

    @staticmethod
    def _display_name(dataset_id: str) -> str:
        return dataset_id.replace("-", " ").replace("v1", "").strip().title()

    @staticmethod
    def _inspect_files(data_root: Path) -> list[DatasetFile]:
        records: list[DatasetFile] = []
        for path in sorted(data_root.iterdir()):
            if not path.is_file() or path.name == "dataset_manifest.json":
                continue
            content = path.read_bytes()
            columns, rows = (
                _csv_shape(content) if path.suffix.lower() == ".csv" else ([], None)
            )
            records.append(
                DatasetFile(
                    role=_file_role(path.name),
                    name=path.name,
                    relative_path=path.name,
                    sha256=_sha256_bytes(content),
                    size_bytes=len(content),
                    rows_including_header=rows,
                    columns=columns,
                )
            )
        return records

    def _builtin_data_root(
        self, template_root: Path, manifest: TemplateManifest
    ) -> Path:
        configured = manifest.builtin_dataset_path
        if not configured:
            return template_root / "data"
        data_root = (self.project_root / configured).resolve()
        if not _is_relative_to(data_root, self.project_root):
            raise DatasetCatalogError(
                "builtin dataset path must stay inside the project root"
            )
        return data_root

    @staticmethod
    def _write_builtin_manifest(
        data_root: Path, dataset_id: str, files: list[DatasetFile]
    ) -> None:
        train = next((item for item in files if item.role == "train"), None)
        if train is None:
            return
        payload = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "dataset_sha256": train.sha256,
            "files": {
                item.relative_path: {
                    "sha256": item.sha256,
                    "rows_including_header": item.rows_including_header,
                }
                for item in files
            },
        }
        (data_root / "dataset_manifest.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _file_role(name: str) -> str:
    lowered = name.lower()
    if lowered == "train.csv" or "train" in lowered:
        return "train"
    if lowered == "test.csv" or "test" in lowered:
        return "test"
    if lowered.endswith((".txt", ".md")):
        return "description"
    return "metadata"


def _csv_shape(content: bytes) -> tuple[list[str], int]:
    try:
        text = content.decode("utf-8-sig")
        rows = list(csv.reader(text.splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DatasetCatalogError(f"invalid CSV file: {exc}") from exc
    if not rows or not rows[0] or any(not column.strip() for column in rows[0]):
        raise DatasetCatalogError("CSV must contain a non-empty header")
    return [column.strip() for column in rows[0]], len(rows)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
