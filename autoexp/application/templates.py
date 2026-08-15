from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from autoexp.domain import TemplateManifest


@dataclass(frozen=True)
class TemplateDescriptor:
    template_id: str
    root: Path
    manifest: TemplateManifest

    @property
    def display_name(self) -> str:
        return self.manifest.display_name or self.template_id.replace("-", " ").title()


class TemplateCatalog:
    """Discover registered experiment templates from their manifests."""

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "experiment_templates"
        if not self.root.is_dir():
            packaged_root = files("experiment_templates")
            if packaged_root.is_dir():
                self.root = Path(str(packaged_root))

    def list_templates(self) -> list[TemplateDescriptor]:
        if not self.root.is_dir():
            return []
        descriptors: list[TemplateDescriptor] = []
        for candidate in sorted(self.root.iterdir()):
            manifest_path = candidate / "manifest.yaml"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            manifest = TemplateManifest.load(manifest_path)
            descriptors.append(
                TemplateDescriptor(manifest.template_id, candidate, manifest)
            )
        return descriptors

    def get(self, template_id: str) -> TemplateDescriptor:
        for descriptor in self.list_templates():
            if descriptor.template_id == template_id:
                return descriptor
        raise ValueError(f"unsupported AutoExp template: {template_id}")
