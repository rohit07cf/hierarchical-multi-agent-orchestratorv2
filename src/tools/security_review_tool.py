"""Security-only review primitives.

Returns structured findings (not a verdict) — the ReviewAgent's LLM
decides how to phrase the overall review around them.
"""

from __future__ import annotations

import re

from src.models.responses import ReviewFinding, ReviewSeverity

_SECRET_RE = re.compile(
    r"(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{4,}['\"]",
    re.IGNORECASE,
)
_SHELL_INJECTION_RE = re.compile(r"shell\s*=\s*True", re.IGNORECASE)
_EVAL_RE = re.compile(r"\b(eval|exec)\s*\(", re.IGNORECASE)


def scan_security(code: str) -> dict:
    """Return security findings for the given snippet."""
    findings: list[ReviewFinding] = []

    if _SECRET_RE.search(code):
        findings.append(
            ReviewFinding(
                category="security",
                severity=ReviewSeverity.BLOCKER,
                message="Hard-coded credential detected — load from environment instead.",
            )
        )
    if _SHELL_INJECTION_RE.search(code):
        findings.append(
            ReviewFinding(
                category="security",
                severity=ReviewSeverity.BLOCKER,
                message="subprocess called with shell=True — risk of command injection.",
            )
        )
    if _EVAL_RE.search(code):
        findings.append(
            ReviewFinding(
                category="security",
                severity=ReviewSeverity.WARNING,
                message="eval/exec usage detected — avoid evaluating untrusted input.",
            )
        )

    return {
        "tool": "security_review_tool",
        "finding_count": len(findings),
        "findings": [f.model_dump(mode="json") for f in findings],
    }
