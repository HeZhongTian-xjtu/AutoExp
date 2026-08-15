from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    rows_including_header: int | None = Field(default=None, ge=0)
    columns: list[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=200)
    source: Literal["builtin", "uploaded"]
    task_type: str = Field(min_length=1, max_length=80)
    compatible_template_ids: list[str] = Field(default_factory=list)
    storage_path: str | None = None
    files: list[DatasetFile] = Field(default_factory=list)
    target_column: str | None = None
    status: Literal["ready", "invalid"] = "ready"
    validation_message: str | None = None
    created_at: datetime
    updated_at: datetime

    def is_compatible(self, template_id: str, task_type: str | None = None) -> bool:
        if self.status != "ready":
            return False
        if not self.files:
            return False
        if (
            self.compatible_template_ids
            and template_id not in self.compatible_template_ids
        ):
            return False
        return task_type is None or self.task_type == task_type

    def file_for_role(self, role: str) -> DatasetFile | None:
        return next((item for item in self.files if item.role == role), None)


def dataset_contract_task_type(contract: dict[str, Any]) -> str:
    return str(contract.get("task_type") or "generic")
