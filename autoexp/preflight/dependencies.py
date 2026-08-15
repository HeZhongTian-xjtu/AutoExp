from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from autoexp.domain.errors import ValidationIssue


IMPORT_TO_DISTRIBUTION = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "fitz": "PyMuPDF",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "torch_geometric": "torch-geometric",
    "yaml": "PyYAML",
    "xgboost": "xgboost",
}


@dataclass
class EnvironmentInfo:
    package_versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.package_versions, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


def top_level_module(name: str) -> str:
    return name.split(".", 1)[0]


def collect_imports(tree: ast.AST) -> list[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return sorted(modules)


def load_requirements(path: Path | None) -> dict[str, Requirement]:
    if path is None or not path.exists():
        return {}
    requirements: dict[str, Requirement] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http:")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        requirements[requirement.name.lower().replace("-", "_")] = requirement
    return requirements


def distribution_for(module: str) -> str | None:
    top_level = top_level_module(module)
    if top_level in IMPORT_TO_DISTRIBUTION:
        return IMPORT_TO_DISTRIBUTION[top_level]
    distributions = metadata.packages_distributions().get(top_level, [])
    return distributions[0] if distributions else None


def inspect_environment(
    imports: list[str], requirements_path: Path | None
) -> tuple[EnvironmentInfo, list[ValidationIssue]]:
    requirements = load_requirements(requirements_path)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    package_versions: dict[str, str | None] = {}
    issues: list[ValidationIssue] = []

    for module in imports:
        top_level = top_level_module(module)
        if top_level in stdlib or top_level.startswith("_"):
            continue
        try:
            available = importlib.util.find_spec(top_level) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        distribution = distribution_for(module) or top_level.replace("_", "-")
        installed_version: str | None = None
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
        package_versions[distribution] = installed_version

        if not available or installed_version is None:
            requirement = requirements.get(distribution.lower().replace("-", "_"))
            requested = str(requirement) if requirement else distribution
            issues.append(
                ValidationIssue(
                    code="MISSING_DEPENDENCY",
                    phase="dependency_check",
                    message=f"module '{module}' is not importable; expected {requested}",
                    suggestion="Add a reviewed dependency and rebuild the environment; do not install during a run.",
                    details={"module": module, "distribution": distribution},
                )
            )
            continue

        requirement = requirements.get(distribution.lower().replace("-", "_"))
        if requirement and not requirement.specifier.contains(
            installed_version, prereleases=True
        ):
            issues.append(
                ValidationIssue(
                    code="VERSION_CONFLICT",
                    phase="dependency_check",
                    message=f"{distribution}=={installed_version} does not satisfy {requirement.specifier}",
                    suggestion="Recreate the environment from the reviewed dependency manifest.",
                    details={
                        "distribution": distribution,
                        "installed_version": installed_version,
                    },
                )
            )
    return EnvironmentInfo(package_versions), issues
