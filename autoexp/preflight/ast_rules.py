from __future__ import annotations

import ast
import sys
from dataclasses import dataclass

from autoexp.domain.errors import ValidationIssue
from autoexp.domain.models import TemplateManifest
from .dependencies import top_level_module


@dataclass(frozen=True)
class RuleContext:
    manifest: TemplateManifest | None = None


class AstRule:
    rule_id = "AST000"

    def check(self, tree: ast.AST, context: RuleContext) -> list[ValidationIssue]:
        raise NotImplementedError


def issue(
    rule_id: str, node: ast.AST, message: str, suggestion: str
) -> ValidationIssue:
    return ValidationIssue(
        code=rule_id,
        phase="ast_check",
        line=getattr(node, "lineno", None),
        message=message,
        suggestion=suggestion,
    )


class ForbiddenImportRule(AstRule):
    rule_id = "AST001_FORBIDDEN_IMPORT"

    def check(self, tree: ast.AST, context: RuleContext) -> list[ValidationIssue]:
        if context.manifest is None:
            return []
        allowed = {top_level_module(name) for name in context.manifest.allowed_imports}
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        results = []
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name if node.names else None
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module
            if module is None:
                continue
            top_level = top_level_module(module)
            if top_level not in stdlib and top_level not in allowed:
                results.append(
                    issue(
                        self.rule_id,
                        node,
                        f"import is outside the template allowlist: {module}",
                        "Use a registered dependency or update the reviewed template manifest.",
                    )
                )
        return results


class DangerousCallRule(AstRule):
    rule_id = "AST002_DANGEROUS_CALL"

    _blocked_names = {"eval", "exec", "compile", "__import__"}
    _blocked_attributes = {
        "system",
        "popen",
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
    }

    def check(self, tree: ast.AST, context: RuleContext) -> list[ValidationIssue]:
        results = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in self._blocked_names:
                results.append(
                    issue(
                        self.rule_id,
                        node,
                        f"dynamic execution is forbidden: {node.func.id}",
                        "Use the fixed template entrypoint instead.",
                    )
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in self._blocked_attributes
            ):
                results.append(
                    issue(
                        self.rule_id,
                        node,
                        f"process or shell execution is forbidden: {node.func.attr}",
                        "Do not invoke Shell or child processes from an experiment.",
                    )
                )
        return results


class NetworkImportRule(AstRule):
    rule_id = "AST003_NETWORK_IMPORT"
    blocked = {"socket", "requests", "httpx", "urllib", "ftplib"}

    def check(self, tree: ast.AST, context: RuleContext) -> list[ValidationIssue]:
        results = []
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name if node.names else None
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module
            if module and top_level_module(module) in self.blocked:
                results.append(
                    issue(
                        self.rule_id,
                        node,
                        f"network access is forbidden: {module}",
                        "Use the preloaded dataset and offline experiment template.",
                    )
                )
        return results


class ParameterLiteralRule(AstRule):
    rule_id = "AST004_PARAMETER_POLICY"

    def check(self, tree: ast.AST, context: RuleContext) -> list[ValidationIssue]:
        if context.manifest is None:
            return []
        results = []
        policy = context.manifest.parameter_policy
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in policy or not isinstance(
                    keyword.value, ast.Constant
                ):
                    continue
                value = keyword.value.value
                if not isinstance(value, (int, float, bool)):
                    continue
                if not policy[keyword.arg].accepts(value):
                    results.append(
                        issue(
                            self.rule_id,
                            keyword,
                            f"literal parameter is outside policy: {keyword.arg}={value!r}",
                            "Use a value from ExperimentSpec and the template parameter policy.",
                        )
                    )
        return results


DEFAULT_RULES = (
    DangerousCallRule(),
    NetworkImportRule(),
    ForbiddenImportRule(),
    ParameterLiteralRule(),
)
