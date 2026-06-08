"""SummarizerAgent — LLM-driven summary of retrieved documents or pasted text.

Has no external tools by default; the LLM synthesizes the summary
directly from the user's text and any documents in `request.context`.
The agent still produces a full reasoning trace so the user can see why
it chose to summarize without calling any tools.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import ReasoningAgent
from src.models.requests import AgentRequest
from src.models.trace import AgentReasoning, ToolInvocation


_SYSTEM_PROMPT = (
    "You are SummarizerAgent. Read the query plus any retrieved "
    "documents and produce a concise 2-3 sentence summary that "
    "directly answers the query using only the provided text."
)


class SummarizerAgent(ReasoningAgent):
    name = "SummarizerAgent"
    system_prompt = _SYSTEM_PROMPT
    tools = []  # no external tools; LLM synthesizes directly

    def user_prompt(self, request: AgentRequest) -> str:
        documents = request.context.get("documents", [])
        if documents:
            joined = "\n\n".join(
                f"### {d['name']}\n{d['text']}" for d in documents
            )
            return f"Query: {request.query}\n\nDocuments:\n{joined}"
        return f"Query: {request.query}"

    def _mock_reason_policy(self, request: AgentRequest):
        docs = request.context.get("documents", [])

        def policy() -> AgentReasoning:
            if docs:
                rationale = (
                    f"Found {len(docs)} retrieved document(s); no external "
                    "tools needed — will summarize their content."
                )
            else:
                rationale = (
                    "No retrieved documents; will summarize the pasted text "
                    "directly. No external tools needed."
                )
            return AgentReasoning(
                reasoning=rationale,
                selected_tools=[],
                skipped_tools=[],
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        docs = request.context.get("documents", [])
        if docs:
            names = ", ".join(d["name"] for d in docs[:3])
            return (
                f"[mock-llm] Would summarize {len(docs)} document(s) "
                f"({names}). Set ANTHROPIC_API_KEY for real natural-language summary."
            )
        snippet = request.query.strip().splitlines()[0][:160]
        return (
            f"[mock-llm] Would summarize the input ({len(request.query)} "
            f"chars, starting with {snippet!r}). Set ANTHROPIC_API_KEY for "
            "real natural-language summary."
        )

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        return {}
