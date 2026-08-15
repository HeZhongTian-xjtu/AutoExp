from __future__ import annotations

import ast
from pathlib import Path

from autoexp.domain.errors import PreflightReport, ValidationIssue
from autoexp.domain.models import TemplateManifest

from .ast_rules import DEFAULT_RULES, RuleContext
from .compatibility import check_xgboost_api
from .dependencies import collect_imports, inspect_environment


class PreflightPipeline:
    """Run all deterministic checks before handing code to an executor."""

    def __init__(self, requirements_path: Path | str | None = None):
        self.requirements_path = Path(requirements_path) if requirements_path else None

    def run(
        self, code: str, manifest: TemplateManifest | None = None
    ) -> PreflightReport:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return PreflightReport(
                passed=False,
                issues=[
                    ValidationIssue(
                        code="SYNTAX_ERROR",
                        phase="syntax_check",
                        line=exc.lineno,
                        message=f"generated code cannot be parsed: {exc.msg}",
                        suggestion="Regenerate the candidate or repair the syntax before execution.",
                    )
                ],
            )

        imports = collect_imports(tree)
        issues: list[ValidationIssue] = []
        context = RuleContext(manifest=manifest)
        for rule in DEFAULT_RULES:
            issues.extend(rule.check(tree, context))
        environment, dependency_issues = inspect_environment(
            imports, self.requirements_path
        )
        issues.extend(dependency_issues)
        issues.extend(
            check_xgboost_api(tree, environment.package_versions.get("xgboost"))
        )
        return PreflightReport(
            passed=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            discovered_imports=imports,
            environment_fingerprint=environment.fingerprint,
        )
