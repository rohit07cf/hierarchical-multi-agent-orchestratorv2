"""Deterministic heuristic-based code review.

Real-world review tooling would use static analysis or an LLM; this
implementation uses a small ruleset so reviews are reproducible and
testable offline. The intent is to demonstrate the *shape* of a review
agent's output (categorized findings, severity, summary), not to replace
a real linter.
"""

from __future__ import annotations

import re

from src.models.responses import ReviewFinding, ReviewResult, ReviewSeverity


def review_code(code: str) -> ReviewResult:
    """Inspect a code snippet for common production-readiness issues.

    Args:
        code: Source code to review.

    Returns:
        A `ReviewResult` with categorized findings and an overall verdict.
    """
    findings: list[ReviewFinding] = []
    code_lower = code.lower()

    if not code.strip():
        return ReviewResult(
            approved=False,
            summary="No code provided for review.",
            findings=[
                ReviewFinding(
                    category="clarity",
                    severity=ReviewSeverity.BLOCKER,
                    message="Empty input — nothing to review.",
                )
            ],
        )

    if "try" not in code_lower or "except" not in code_lower:
        findings.append(
            ReviewFinding(
                category="bugs",
                severity=ReviewSeverity.WARNING,
                message="No try/except block found — consider explicit error handling.",
            )
        )

    if re.search(r"(api[_-]?key|secret|password)\s*=\s*['\"]", code, re.IGNORECASE):
        findings.append(
            ReviewFinding(
                category="security",
                severity=ReviewSeverity.BLOCKER,
                message="Hard-coded credential detected — load from environment instead.",
            )
        )

    if "test_" not in code_lower and "def test" not in code_lower:
        findings.append(
            ReviewFinding(
                category="tests",
                severity=ReviewSeverity.INFO,
                message="No tests included; add unit tests covering happy and error paths.",
            )
        )

    if "fastapi" in code_lower and "httpexception" not in code_lower:
        findings.append(
            ReviewFinding(
                category="production",
                severity=ReviewSeverity.WARNING,
                message="FastAPI endpoint without HTTPException — error responses may leak internals.",
            )
        )

    if len(code) > 1500 and code.count("def ") < 2:
        findings.append(
            ReviewFinding(
                category="clarity",
                severity=ReviewSeverity.INFO,
                message="Long function — consider breaking into smaller helpers.",
            )
        )

    has_blocker = any(f.severity == ReviewSeverity.BLOCKER for f in findings)
    summary = (
        f"{len(findings)} finding(s); "
        f"{'blockers present' if has_blocker else 'no blockers'}."
    )
    return ReviewResult(approved=not has_blocker, summary=summary, findings=findings)
