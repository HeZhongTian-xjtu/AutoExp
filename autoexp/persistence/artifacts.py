from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID
    run_id: UUID
    trial_id: UUID | None = None
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int


class ArtifactStore:
    """Content-addressed metadata for reproducible run files."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(
        self,
        source: Path | str,
        run_id: UUID,
        kind: str,
        trial_id: UUID | None = None,
    ) -> ArtifactRecord:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        bucket = (
            self.root
            / str(run_id)
            / (str(trial_id) if trial_id else "run")
            / _safe_kind(kind)
        )
        bucket.mkdir(parents=True, exist_ok=True)
        destination = bucket / source_path.name
        shutil.copy2(source_path, destination)
        return self._record(destination, run_id, trial_id, kind)

    def put_text(
        self,
        content: str,
        run_id: UUID,
        kind: str,
        filename: str,
        trial_id: UUID | None = None,
    ) -> ArtifactRecord:
        bucket = (
            self.root
            / str(run_id)
            / (str(trial_id) if trial_id else "run")
            / _safe_kind(kind)
        )
        bucket.mkdir(parents=True, exist_ok=True)
        destination = bucket / Path(filename).name
        destination.write_text(content, encoding="utf-8")
        return self._record(destination, run_id, trial_id, kind)

    def put_json(
        self,
        value: Any,
        run_id: UUID,
        kind: str,
        filename: str,
        trial_id: UUID | None = None,
    ) -> ArtifactRecord:
        return self.put_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            run_id,
            kind,
            filename,
            trial_id,
        )

    def clear(self) -> None:
        if not self.root.exists():
            return
        for child in self.root.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    def capture_trial(
        self, run_id: UUID, outcome: Any, manifest: Any
    ) -> list[ArtifactRecord]:
        workspace = Path(outcome.workspace)
        candidates = {
            "dataset_manifest": workspace
            / (manifest.dataset_manifest or "data/dataset_manifest.json"),
            "entrypoint": workspace / manifest.entrypoint,
            "metric": workspace / manifest.metric_file,
            "config": workspace / "configs" / "experiment.yaml",
            "stdout": workspace / "working" / "stdout.log",
            "stderr": workspace / "working" / "stderr.log",
            "evaluator_stdout": workspace / "working" / "evaluator" / "stdout.log",
            "evaluator_stderr": workspace / "working" / "evaluator" / "stderr.log",
            "predictions": workspace / "working" / "predictions.npz",
            "run_metadata": workspace / "working" / "run_metadata.json",
        }
        records: list[ArtifactRecord] = []
        for kind, path in candidates.items():
            if path.is_file():
                records.append(self.put_file(path, run_id, kind, outcome.trial_id))
        gate_root = workspace / "working" / "gates"
        if gate_root.is_dir():
            for gate_file in sorted(gate_root.rglob("*")):
                if gate_file.is_file():
                    records.append(
                        self.put_file(
                            gate_file,
                            run_id,
                            f"gate_{gate_file.parent.name}",
                            outcome.trial_id,
                        )
                    )
        return records

    def _record(
        self, path: Path, run_id: UUID, trial_id: UUID | None, kind: str
    ) -> ArtifactRecord:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ArtifactRecord(
            artifact_id=uuid4(),
            run_id=run_id,
            trial_id=trial_id,
            kind=kind,
            relative_path=path.relative_to(self.root).as_posix(),
            sha256=digest,
            size_bytes=path.stat().st_size,
        )


def _safe_kind(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe or "artifact"
