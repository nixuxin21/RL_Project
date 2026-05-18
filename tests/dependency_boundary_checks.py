"""
Lightweight dependency-boundary checks for the refactored experiment layer.

The reusable helpers should live under ``ms_aircomp``. Top-level orchestration
scripts may call the execution-mismatch evaluator as a runner, but new code
should not import helper symbols from it or reuse the limited-CSI evaluator as
a helper module.
"""

import ast
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_MODULE = "evaluate_execution_channel_mismatch"
LIMITED_EVALUATOR_MODULE = "evaluate_limited_csi_ms_aircomp"
BLOCKED_PACKAGE_IMPORTS = {
    LIMITED_EVALUATOR_MODULE: (
        "ms_aircomp package code must import limited-CSI helpers from "
        "ms_aircomp.limited_csi, not from the top-level evaluator"
    ),
}
EVALUATOR_PATH = Path("evaluate_execution_channel_mismatch.py")
MAX_EVALUATOR_LINES = 1300
ALLOWED_EVALUATOR_FUNCTIONS = {
    "parse_csv_items",
    "parse_args",
    "validate_args",
    "evaluate_policy",
    "main",
}
ALLOWED_WHOLE_IMPORTS = {
    Path("train_temporal_deviation_selector.py"),
    Path("tests/mainline_regression_checks.py"),
}
ALLOWED_LIMITED_EVALUATOR_IMPORTS = {
    Path("tests/smoke_checks.py"),
}
FORBIDDEN_EVALUATOR_MODULE_IMPORTS = {
    "csv",
    "matplotlib",
    "matplotlib.pyplot",
}
FORBIDDEN_EVALUATOR_REUSABLE_IMPORTS = {
    "ms_aircomp.adaptive_sparse_policies",
    "ms_aircomp.confirmation",
    "ms_aircomp.execution_policies",
    "ms_aircomp.execution_risk_policies",
    "ms_aircomp.feedback",
    "ms_aircomp.probe_sets",
    "ms_aircomp.temporal_policies",
}
SKIPPED_DIRS = {
    ".git",
    ".matplotlib",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "results",
    "rl_logs",
}


def project_python_files():
    """Yield Python files that belong to the checked project surface."""
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        yield path


def import_violations(path):
    """Return dependency-boundary violations found in one Python file."""
    relative = path.relative_to(PROJECT_ROOT)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == BLOCKED_MODULE:
            violations.append(
                (
                    node.lineno,
                    "direct symbol import from evaluator is not allowed; "
                    "import reusable helpers from ms_aircomp instead",
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != BLOCKED_MODULE:
                    continue
                if relative not in ALLOWED_WHOLE_IMPORTS:
                    violations.append(
                        (
                            node.lineno,
                            "whole evaluator import is only allowed for approved orchestration/test runners",
                        )
                    )
        if isinstance(node, ast.ImportFrom) and node.module == LIMITED_EVALUATOR_MODULE:
            if relative not in ALLOWED_LIMITED_EVALUATOR_IMPORTS:
                violations.append(
                    (
                        node.lineno,
                        "direct limited-CSI evaluator import is not allowed; "
                        "import reusable helpers from ms_aircomp.limited_csi instead",
                    )
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != LIMITED_EVALUATOR_MODULE:
                    continue
                if relative not in ALLOWED_LIMITED_EVALUATOR_IMPORTS:
                    violations.append(
                        (
                            node.lineno,
                            "whole limited-CSI evaluator import is not allowed for helper reuse; "
                            "use ms_aircomp.limited_csi",
                        )
                    )
        if relative.parts and relative.parts[0] == "ms_aircomp":
            if isinstance(node, ast.ImportFrom):
                reason = BLOCKED_PACKAGE_IMPORTS.get(node.module or "")
                if reason is not None:
                    violations.append((node.lineno, reason))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    reason = BLOCKED_PACKAGE_IMPORTS.get(alias.name)
                    if reason is not None:
                        violations.append((node.lineno, reason))
    return violations


def evaluator_import_reason(module_name):
    """Return the evaluator-boundary reason for a forbidden import."""
    if module_name in FORBIDDEN_EVALUATOR_MODULE_IMPORTS:
        return (
            "output serialization/plotting belongs in "
            "ms_aircomp.execution_output or ms_aircomp.execution_result_summary"
        )
    if module_name in FORBIDDEN_EVALUATOR_REUSABLE_IMPORTS:
        return (
            "policy/feedback helper modules should stay behind "
            "ms_aircomp.execution_decision_dispatch"
        )
    return None


def evaluator_surface_violations():
    """Return violations of the execution-mismatch evaluator's thin surface."""
    path = PROJECT_ROOT / EVALUATOR_PATH
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EVALUATOR_PATH))
    violations = []

    line_count = len(source.splitlines())
    if line_count > MAX_EVALUATOR_LINES:
        violations.append(
            (
                1,
                f"evaluator has {line_count} lines; keep it below "
                f"{MAX_EVALUATOR_LINES} by promoting reusable logic into ms_aircomp",
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in ALLOWED_EVALUATOR_FUNCTIONS:
                violations.append(
                    (
                        node.lineno,
                        f"unexpected top-level function {node.name!r}; "
                        "new reusable helpers belong in ms_aircomp",
                    )
                )
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                reason = evaluator_import_reason(alias.name)
                if reason is not None:
                    violations.append((node.lineno, reason))
            continue
        if isinstance(node, ast.ImportFrom):
            reason = evaluator_import_reason(node.module or "")
            if reason is not None:
                violations.append((node.lineno, reason))
                continue
            if node.module == "ms_aircomp":
                for alias in node.names:
                    reason = evaluator_import_reason(f"ms_aircomp.{alias.name}")
                    if reason is not None:
                        violations.append((node.lineno, reason))

    return violations


def main():
    """Check evaluator dependency boundaries."""
    violations = []
    for path in project_python_files():
        for line, reason in import_violations(path):
            relative = path.relative_to(PROJECT_ROOT)
            violations.append(f"{relative}:{line}: {reason}")
    for line, reason in evaluator_surface_violations():
        violations.append(f"{EVALUATOR_PATH}:{line}: {reason}")

    if violations:
        print("dependency boundary checks failed", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1

    print("dependency boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
