"""ResearchManagerAgent — coordinates retrieval + summarization."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.rag_agent import RAGAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse


class ResearchManagerAgent(BaseAgent):
    """Run the RAG → Summarize chain for a research request.

    The manager is intentionally thin: it owns the workflow shape and the
    contract with the supervisor, while the workers own the actual logic.
    """

    name = "ResearchManagerAgent"
    tools: list[str] = []

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(model=model)
        self.rag = RAGAgent(model=model)
        self.summarizer = SummarizerAgent(model=model)

    async def handle(self, request: AgentRequest) -> AgentResponse:
        self._log("Delegating to RAGAgent")
        rag_response = await self.rag.handle(
            AgentRequest(query=request.query, parent_agent=self.name)
        )

        self._log("Delegating to SummarizerAgent")
        summarizer_response = await self.summarizer.handle(
            AgentRequest(
                query=request.query,
                context={"documents": rag_response.data.get("documents", [])},
                parent_agent=self.name,
            )
        )

        return AgentResponse(
            agent_name=self.name,
            content=summarizer_response.content,
            data={
                "documents": rag_response.data.get("documents", []),
                "summary": summarizer_response.data.get("summary", ""),
                "source_documents": summarizer_response.data.get("source_documents", []),
            },
        )
