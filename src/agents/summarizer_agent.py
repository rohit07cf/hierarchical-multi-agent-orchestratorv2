"""SummarizerAgent — condenses retrieved documents into a short context summary."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse

_SUMMARIZER_SYSTEM = (
    "You are a concise summarizer. Given retrieved documents and a query, "
    "produce a 2-3 sentence summary that directly answers the query using "
    "only the documents."
)


class SummarizerAgent(BaseAgent):
    """Summarize retrieved documents into a concise context blurb.

    Consumes `request.context['documents']` (as produced by `RAGAgent`)
    and returns a short summary. Falls back to a deterministic heuristic
    when no LLM is available.
    """

    name = "SummarizerAgent"
    tools = ["llm_summarize"]

    async def handle(self, request: AgentRequest) -> AgentResponse:
        documents = request.context.get("documents", [])
        if not documents:
            self._log("No documents to summarize")
            return AgentResponse(
                agent_name=self.name,
                content="No documents were retrieved; nothing to summarize.",
                data={"summary": ""},
            )

        joined = "\n\n".join(f"### {d['name']}\n{d['text']}" for d in documents)
        prompt = (
            f"Query: {request.query}\n\n"
            f"Documents:\n{joined}\n\n"
            "Write a 2-3 sentence summary answering the query."
        )

        if self.llm.enabled:
            summary = await self.llm.complete(prompt, system=_SUMMARIZER_SYSTEM)
        else:
            summary = _heuristic_summary(request.query, documents)

        self._log("Summary completed")
        return AgentResponse(
            agent_name=self.name,
            content=summary,
            data={"summary": summary, "source_documents": [d["name"] for d in documents]},
        )


def _heuristic_summary(query: str, documents: list[dict]) -> str:
    """Build a deterministic summary by stitching the leading lines of each doc."""
    parts: list[str] = []
    for doc in documents[:2]:
        text = doc.get("text", "")
        first_paragraph = next(
            (p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("#")),
            "",
        )
        if first_paragraph:
            parts.append(f"From {doc['name']}: {first_paragraph[:240]}")
    if not parts:
        return f"No substantive content retrieved for query: {query!r}."
    return " ".join(parts)
