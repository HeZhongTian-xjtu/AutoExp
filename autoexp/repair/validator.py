from __future__ import annotations

from pathlib import Path

from autoexp.domain import TemplateManifest
from autoexp.preflight import PreflightPipeline


def validate_repaired_template(
    template_root: Path, manifest: TemplateManifest, requirements_path: Path
) -> object:
    """Run AST, dependency and API compatibility gates before a repaired trial."""
    code_path = template_root / manifest.entrypoint
    if not code_path.is_file():
        raise FileNotFoundError(code_path)
    return PreflightPipeline(requirements_path).run(
        code_path.read_text(encoding="utf-8"), manifest
    )
