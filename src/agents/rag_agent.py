"""RAGAgent — LLM-driven retrieval of local knowledge-base context."""

from __future__ import annotations

from typing import Any

from src.agents.base import ReasoningAgent
from src.models.requests import AgentRequest
from src.models.trace import (
    AgentReasoning,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)
from src.observability.middleware import observe_retrieval
from src.tools.document_loader import load_knowledge_base
from src.tools.simple_retriever import SimpleRetriever

_SYSTEM_PROMPT = (
    "You are RAGAgent. Decide whether to (a) call simple_retriever to find "
    "relevant local knowledge-base documents for the query, or (b) call "
    "load_knowledge_base to enumerate what exists, or both. Prefer "
    "simple_retriever when the query has clear topical keywords."
)


class RAGAgent(ReasoningAgent):
    name = "RAGAgent"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, model: str = "gpt-4.1-nano", top_k: int = 2) -> None:
        super().__init__(model=model)
        self._documents = load_knowledge_base()
        self._retriever = SimpleRetriever(self._documents)
        self._top_k = top_k
        self.tools = [
            ToolSpec(
                name="simple_retriever",
                description=(
                    "Return the top-k local knowledge-base documents whose "
                    "tokens overlap the query. Args: query (str), top_k (int)."
                ),
                handler=self._tool_retrieve,
            ),
            ToolSpec(
                name="load_knowledge_base",
                description=(
                    "List every knowledge-base document name; useful when "
                    "the query is exploratory. No arguments."
                ),
                handler=self._tool_list,
            ),
        ]

    def user_prompt(self, request: AgentRequest) -> str:
        return f"Query: {request.query}"

    def _mock_reason_policy(self, request: AgentRequest):
        def policy() -> AgentReasoning:
            return AgentReasoning(
                reasoning=(
                    "Query has topical keywords and the local knowledge base "
                    "exists; using simple_retriever to fetch the top documents."
                ),
                selected_tools=[
                    ToolDecision(
                        tool_name="simple_retriever",
                        rationale="Find the most relevant docs by token overlap.",
                        arguments={"query": request.query, "top_k": self._top_k},
                    )
                ],
                skipped_tools=["load_knowledge_base"],
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        docs = self._docs_from_invocations(invocations)
        if not docs:
            return "[mock-llm] No knowledge-base documents matched the query."
        names = ", ".join(d["name"] for d in docs)
        return f"[mock-llm] Retrieved {len(docs)} document(s): {names}."

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        return {"documents": self._docs_from_invocations(invocations)}

    # ----------------- Tool handlers -----------------

    async def _tool_retrieve(self, query: str, top_k: int | None = None) -> dict:
        k = top_k or self._top_k
        # RAG needs its own span + quality signals: latency alone hides the
        # failures that matter (relevant-but-slow vs fast-but-empty). We
        # record docs returned, the best relevance score, and context size.
        async with observe_retrieval(query, k) as obs:
            docs = self._retriever.retrieve(query, top_k=k)
            top_score = max((d.score for d in docs), default=0.0)
            context_chars = sum(len(d.text) for d in docs)
            obs.record(
                docs_returned=len(docs),
                top_score=top_score,
                context_chars=context_chars,
            )
            return {
                "count": len(docs),
                "documents": [
                    {"name": d.name, "score": d.score, "text": d.text}
                    for d in docs
                ],
            }

    def _tool_list(self) -> dict:
        return {
            "count": len(self._documents),
            "names": sorted(self._documents.keys()),
        }

    @staticmethod
    def _docs_from_invocations(invocations: list[ToolInvocation]) -> list[dict]:
        """Pull the document list out of the retriever invocation, if any."""
        for inv in invocations:
            if inv.tool_name == "simple_retriever" and inv.success and isinstance(inv.result, dict):
                return inv.result.get("documents", [])
        return []
