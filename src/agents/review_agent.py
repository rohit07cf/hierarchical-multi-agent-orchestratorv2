"""ReviewAgent — LLM-driven code review across three review tools."""

from __future__ import annotations

from typing import Any

from src.agents.base import ReasoningAgent
from src.models.requests import AgentRequest
from src.models.responses import ReviewFinding, ReviewResult, ReviewSeverity
from src.models.trace import (
    AgentReasoning,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)
from src.tools.code_review_tool import review_code
from src.tools.security_review_tool import scan_security
from src.tools.test_gap_tool import scan_test_gaps

_SYSTEM_PROMPT = (
    "You are ReviewAgent. Decide whether to call code_review_tool "
    "(general bugs/clarity), security_review_tool (secrets, shell "
    "injection, eval), test_gap_tool (test coverage), or any combination. "
    "Then summarize the review with a verdict and bullet-list of findings."
)


class ReviewAgent(ReasoningAgent):
    name = "ReviewAgent"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(model=model)
        self.tools = [
            ToolSpec(
                name="code_review_tool",
                description=(
                    "General-purpose review: error handling, FastAPI "
                    "exception use, long functions. Args: code (str)."
                ),
                handler=lambda code: review_code(code).model_dump(mode="json"),
            ),
            ToolSpec(
                name="security_review_tool",
                description=(
                    "Security-focused scan for hard-coded secrets, "
                    "shell=True, and eval/exec. Args: code (str)."
                ),
                handler=scan_security,
            ),
            ToolSpec(
                name="test_gap_tool",
                description=(
                    "Detect missing test coverage for top-level functions "
                    "and classes. Args: code (str)."
                ),
                handler=scan_test_gaps,
            ),
        ]

    def user_prompt(self, request: AgentRequest) -> str:
        code = request.context.get("code", "")
        return (
            f"Original request: {request.query}\n\n"
            f"Code to review:\n```\n{code}\n```"
        )

    def _mock_reason_policy(self, request: AgentRequest):
        code = request.context.get("code", "")

        def policy() -> AgentReasoning:
            if not code.strip():
                return AgentReasoning(
                    reasoning="No code in context; nothing to review.",
                    selected_tools=[],
                    skipped_tools=[
                        "code_review_tool",
                        "security_review_tool",
                        "test_gap_tool",
                    ],
                )
            return AgentReasoning(
                reasoning=(
                    "Code present; running all three review tools as "
                    "the default quality gate."
                ),
                selected_tools=[
                    ToolDecision(
                        tool_name="code_review_tool",
                        rationale="Check for general bugs and clarity issues.",
                        arguments={"code": code},
                    ),
                    ToolDecision(
                        tool_name="security_review_tool",
                        rationale="Check for hard-coded secrets and unsafe calls.",
                        arguments={"code": code},
                    ),
                    ToolDecision(
                        tool_name="test_gap_tool",
                        rationale="Identify untested functions/classes.",
                        arguments={"code": code},
                    ),
                ],
                skipped_tools=[],
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        review = self._build_aggregate_review(invocations)
        if not review.findings and not invocations:
            return "[mock-llm] No code to review."
        bullets = (
            "\n".join(
                f"- [{f.severity.value}] ({f.category}) {f.message}"
                for f in review.findings
            )
            or "- No issues detected."
        )
        return (
            f"[mock-llm] {review.summary}\n{bullets}\n"
            "Set OPENAI_API_KEY for a synthesized review narrative."
        )

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        review = self._build_aggregate_review(invocations)
        return {"review": review.model_dump(mode="json")}

    @staticmethod
    def _build_aggregate_review(invocations: list[ToolInvocation]) -> ReviewResult:
        """Merge findings from every review tool into one ReviewResult."""
        findings: list[ReviewFinding] = []
        for inv in invocations:
            if not inv.success or not isinstance(inv.result, dict):
                continue
            for raw in inv.result.get("findings", []):
                try:
                    findings.append(ReviewFinding.model_validate(raw))
                except Exception:
                    continue
        has_blocker = any(f.severity == ReviewSeverity.BLOCKER for f in findings)
        summary = (
            f"{len(findings)} finding(s) across "
            f"{sum(1 for i in invocations if i.success)} tool(s); "
            f"{'blockers present' if has_blocker else 'no blockers'}."
        )
        return ReviewResult(
            approved=not has_blocker, summary=summary, findings=findings
        )
