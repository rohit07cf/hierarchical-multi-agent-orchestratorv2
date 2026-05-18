"""Test-coverage gap analysis primitive."""

from __future__ import annotations

import re

from src.models.responses import ReviewFinding, ReviewSeverity

_DEF_RE = re.compile(r"^\s*def\s+(\w+)", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)


def scan_test_gaps(code: str) -> dict:
    """Return findings describing missing tests for top-level defs/classes."""
    function_names = _DEF_RE.findall(code)
    class_names = _CLASS_RE.findall(code)
    has_tests = bool(re.search(r"def\s+test_", code))

    findings: list[ReviewFinding] = []
    if not has_tests:
        if function_names or class_names:
            findings.append(
                ReviewFinding(
                    category="tests",
                    severity=ReviewSeverity.INFO,
                    message=(
                        f"No tests detected; consider covering "
                        f"{', '.join((function_names + class_names)[:3])}."
                    ),
                )
            )

    return {
        "tool": "test_gap_tool",
        "functions": function_names,
        "classes": class_names,
        "has_tests": has_tests,
        "findings": [f.model_dump(mode="json") for f in findings],
    }
