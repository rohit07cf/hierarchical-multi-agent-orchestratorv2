"""General-purpose code review primitive.

Focuses on clarity, error handling, and production-readiness signals
that aren't security-specific (those live in `security_review_tool.py`)
and aren't test-coverage-specific (those live in `test_gap_tool.py`).
The ReviewAgent's LLM combines all three into a single review.
"""

from __future__ import annotations

from src.models.responses import ReviewFinding, ReviewResult, ReviewSeverity


def review_code(code: str) -> ReviewResult:
    """Inspect `code` for general bugs and clarity issues.

    Returns a `ReviewResult` rather than a plain dict so callers that
    want a verdict (BuildManager's quality gate) can read `approved`
    directly.
    """
    findings: list[ReviewFinding] = []

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

    code_lower = code.lower()

    if "try" not in code_lower or "except" not in code_lower:
        findings.append(
            ReviewFinding(
                category="bugs",
                severity=ReviewSeverity.WARNING,
                message="No try/except block found — consider explicit error handling.",
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
        f"{len(findings)} general finding(s); "
        f"{'blockers present' if has_blocker else 'no blockers'}."
    )
    return ReviewResult(approved=not has_blocker, summary=summary, findings=findings)
