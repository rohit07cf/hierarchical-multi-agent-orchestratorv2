"""ResearchManagerAgent — LLM-driven coordinator for the research workflow."""

from __future__ import annotations

import re
from typing import Any

from src.agents.base import ReasoningAgent
from src.agents.rag_agent import RAGAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.trace import (
    AgentReasoning,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)

_SYSTEM_PROMPT = (
    "You are ResearchManagerAgent. Your two worker agents are exposed as "
    "tools: call_rag_agent (retrieves knowledge-base documents) and "
    "call_summarizer_agent (produces a summary). Decide which to call. "
    "Skip the RAG step if the user pasted text inline that does not need "
    "external context; in that case just call call_summarizer_agent."
)

# RAG is called only on explicit lookup intent — never on bare
# summarization/rewriting/philosophical text. Domain-specific words
# (architecture, agent patterns, knowledge base) count as lookup intent
# because they map directly to documents we know we have.
_SEARCH_HINT_RE = re.compile(
    r"\b(search|lookup|knowledge base|architecture|agent pattern|agent patterns|"
    r"orchestration pattern|orchestration patterns|documentation)\b",
    re.IGNORECASE,
)


class ResearchManagerAgent(ReasoningAgent):
    name = "ResearchManagerAgent"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        super().__init__(model=model)
        self.rag = RAGAgent(model=model)
        self.summarizer = SummarizerAgent(model=model)
        self.tools = [
            ToolSpec(
                name="call_rag_agent",
                description=(
                    "Delegate to RAGAgent to retrieve documents from the "
                    "local knowledge base. Args: query (str)."
                ),
                handler=self._tool_call_rag,
            ),
            ToolSpec(
                name="call_summarizer_agent",
                description=(
                    "Delegate to SummarizerAgent to summarize the query "
                    "and any retrieved documents. Args: query (str)."
                ),
                handler=self._tool_call_summarizer,
            ),
        ]
        # Documents retrieved during this run, forwarded to the summarizer
        # and out to downstream managers.
        self._retrieved_docs: list[dict] = []

    def _reset_run_state(self) -> None:
        self._retrieved_docs = []

    def user_prompt(self, request: AgentRequest) -> str:
        return (
            f"User request:\n{request.query}\n\n"
            f"Length: {len(request.query)} chars. "
            "Decide whether to retrieve docs first or skip straight to "
            "summarization."
        )

    def _mock_reason_policy(self, request: AgentRequest):
        needs_rag = self._should_retrieve(request.query)

        def policy() -> AgentReasoning:
            if needs_rag:
                return AgentReasoning(
                    reasoning=(
                        "Query has lookup intent or is short enough that we "
                        "need external context; calling RAG first, then "
                        "the summarizer."
                    ),
                    selected_tools=[
                        ToolDecision(
                            tool_name="call_rag_agent",
                            rationale="Retrieve relevant local docs.",
                            arguments={"query": request.query},
                        ),
                        ToolDecision(
                            tool_name="call_summarizer_agent",
                            rationale="Summarize the query using retrieved docs.",
                            arguments={"query": request.query},
                        ),
                    ],
                    skipped_tools=[],
                )
            return AgentReasoning(
                reasoning=(
                    "Long pasted text without lookup intent — summarizing "
                    "directly, no retrieval needed."
                ),
                selected_tools=[
                    ToolDecision(
                        tool_name="call_summarizer_agent",
                        rationale="Summarize the pasted text in place.",
                        arguments={"query": request.query},
                    ),
                ],
                skipped_tools=["call_rag_agent"],
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        summary_response = self._summarizer_response(invocations)
        if summary_response is not None:
            return summary_response.content
        rag_response = self._rag_response(invocations)
        if rag_response is not None:
            return rag_response.content
        return "[mock-llm] ResearchManagerAgent invoked no workers."

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        worker_traces = [
            inv.result.trace
            for inv in invocations
            if inv.success and isinstance(inv.result, AgentResponse) and inv.result.trace
        ]
        summary_response = self._summarizer_response(invocations)
        return {
            "documents": self._retrieved_docs,
            "summary": summary_response.content if summary_response else "",
            "worker_traces": worker_traces,
        }

    # ----------------- Tool handlers -----------------

    @staticmethod
    def _should_retrieve(query: str) -> bool:
        """RAG fires only on explicit lookup intent.

        Reflective, philosophical, or "rewrite/summarize this text"
        requests get summarized directly without touching the knowledge
        base, even when they're short.
        """
        return bool(_SEARCH_HINT_RE.search(query))

    async def _tool_call_rag(self, query: str) -> AgentResponse:
        response = await self.rag.handle(
            AgentRequest(query=query, parent_agent=self.name)
        )
        self._retrieved_docs = response.data.get("documents", [])
        return response

    async def _tool_call_summarizer(self, query: str) -> AgentResponse:
        return await self.summarizer.handle(
            AgentRequest(
                query=query,
                context={"documents": self._retrieved_docs},
                parent_agent=self.name,
            )
        )

    # ----------------- Helpers -----------------

    @staticmethod
    def _summarizer_response(invocations: list[ToolInvocation]) -> AgentResponse | None:
        for inv in invocations:
            if inv.tool_name == "call_summarizer_agent" and inv.success and isinstance(inv.result, AgentResponse):
                return inv.result
        return None

    @staticmethod
    def _rag_response(invocations: list[ToolInvocation]) -> AgentResponse | None:
        for inv in invocations:
            if inv.tool_name == "call_rag_agent" and inv.success and isinstance(inv.result, AgentResponse):
                return inv.result
        return None
