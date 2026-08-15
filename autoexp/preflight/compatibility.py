from __future__ import annotations

import ast
import inspect

from autoexp.domain.errors import ValidationIssue


def check_xgboost_api(
    tree: ast.AST, installed_version: str | None
) -> list[ValidationIssue]:
    uses_xgboost = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.split(".", 1)[0] == "xgboost" for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".", 1)[0] == "xgboost"
            )
        )
        for node in ast.walk(tree)
    )
    if not uses_xgboost:
        return []
    uses_old_fit_argument = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
        and any(keyword.arg == "early_stopping_rounds" for keyword in node.keywords)
        for node in ast.walk(tree)
    )
    if not uses_old_fit_argument:
        return []
    try:
        from xgboost import XGBModel

        accepts_argument = (
            "early_stopping_rounds" in inspect.signature(XGBModel.fit).parameters
        )
    except (ImportError, TypeError, ValueError):
        return []
    if accepts_argument:
        return []
    return [
        ValidationIssue(
            code="API_INCOMPATIBLE",
            phase="compatibility_check",
            message="XGBModel.fit() does not accept early_stopping_rounds in the installed XGBoost API.",
            suggestion="Move the option to the model constructor or regenerate without early stopping.",
            details={"package": "xgboost", "installed_version": installed_version},
        )
    ]
