"""BuildManagerAgent — LLM-driven coordinator for the implementation workflow."""

from __future__ import annotations

from typing import Any

from src.agents.base import ReasoningAgent
from src.agents.coding_agent import CodingAgent
from src.agents.review_agent import ReviewAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.trace import (
    AgentReasoning,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)

_SYSTEM_PROMPT = (
    "You are BuildManagerAgent. Your two worker agents are exposed as "
    "tools: call_coding_agent (generates code) and call_review_agent "
    "(audits the generated code). ReviewAgent is configured as the "
    "default quality gate, so call both unless the user explicitly says "
    "'no review' or asks ONLY for review of pre-existing code."
)


class BuildManagerAgent(ReasoningAgent):
    name = "BuildManagerAgent"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        super().__init__(model=model)
        self.coder = CodingAgent(model=model)
        self.reviewer = ReviewAgent(model=model)
        self.tools = [
            ToolSpec(
                name="call_coding_agent",
                description=(
                    "Delegate to CodingAgent to produce a code snippet. "
                    "Args: query (str)."
                ),
                handler=self._tool_call_coding,
            ),
            ToolSpec(
                name="call_review_agent",
                description=(
                    "Delegate to ReviewAgent to audit the most-recently "
                    "generated code. Args: query (str)."
                ),
                handler=self._tool_call_review,
            ),
        ]
        self._generated_code: str = ""

    def _reset_run_state(self) -> None:
        self._generated_code = ""

    def user_prompt(self, request: AgentRequest) -> str:
        return f"Build request:\n{request.query}"

    def _mock_reason_policy(self, request: AgentRequest):
        skip_review = "no review" in request.query.lower()

        def policy() -> AgentReasoning:
            selected: list[ToolDecision] = [
                ToolDecision(
                    tool_name="call_coding_agent",
                    rationale="Generate the requested code.",
                    arguments={"query": request.query},
                )
            ]
            skipped: list[str] = []
            if skip_review:
                skipped.append("call_review_agent")
                reasoning_text = (
                    "User explicitly opted out of review; calling coder only."
                )
            else:
                selected.append(
                    ToolDecision(
                        tool_name="call_review_agent",
                        rationale=(
                            "Default quality gate — review the generated code."
                        ),
                        arguments={"query": request.query},
                    )
                )
                reasoning_text = (
                    "Default quality gate: generate then review."
                )
            return AgentReasoning(
                reasoning=reasoning_text,
                selected_tools=selected,
                skipped_tools=skipped,
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        coder = self._invocation(invocations, "call_coding_agent")
        reviewer = self._invocation(invocations, "call_review_agent")
        parts: list[str] = []
        if coder and coder.success and isinstance(coder.result, AgentResponse):
            parts.append(coder.result.content)
        if reviewer and reviewer.success and isinstance(reviewer.result, AgentResponse):
            parts.append(f"**Review:**\n{reviewer.result.content}")
        if not parts:
            return "[mock-llm] BuildManagerAgent invoked no workers."
        return "\n\n".join(parts)

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        coder = self._invocation(invocations, "call_coding_agent")
        reviewer = self._invocation(invocations, "call_review_agent")
        worker_traces = [
            inv.result.trace
            for inv in invocations
            if inv.success and isinstance(inv.result, AgentResponse) and inv.result.trace
        ]
        return {
            "code": self._generated_code,
            "language": "python",
            "review": (
                reviewer.result.data.get("review", {})
                if reviewer and reviewer.success and isinstance(reviewer.result, AgentResponse)
                else {}
            ),
            "worker_traces": worker_traces,
        }

    # ----------------- Tool handlers -----------------

    async def _tool_call_coding(self, query: str) -> AgentResponse:
        response = await self.coder.handle(
            AgentRequest(query=query, parent_agent=self.name)
        )
        self._generated_code = response.data.get("code", "")
        return response

    async def _tool_call_review(self, query: str) -> AgentResponse:
        return await self.reviewer.handle(
            AgentRequest(
                query=query,
                context={"code": self._generated_code},
                parent_agent=self.name,
            )
        )

    # ----------------- Helpers -----------------

    @staticmethod
    def _invocation(invocations: list[ToolInvocation], name: str) -> ToolInvocation | None:
        for inv in invocations:
            if inv.tool_name == name:
                return inv
        return None
