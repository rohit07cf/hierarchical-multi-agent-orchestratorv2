"""RAGAgent — retrieves relevant context from the local knowledge base."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.tools.document_loader import load_knowledge_base
from src.tools.simple_retriever import SimpleRetriever


class RAGAgent(BaseAgent):
    """Retrieves the top knowledge-base documents for a query.

    Uses `SimpleRetriever` over local markdown files. The retrieved
    documents are returned in `response.data['documents']` so the
    SummarizerAgent can consume them downstream.
    """

    name = "RAGAgent"
    tools = ["load_knowledge_base", "simple_retriever"]

    def __init__(self, model: str = "gpt-4.1-nano", top_k: int = 2) -> None:
        super().__init__(model=model)
        self._retriever = SimpleRetriever(load_knowledge_base())
        self._top_k = top_k

    async def handle(self, request: AgentRequest) -> AgentResponse:
        if self._retriever.is_empty:
            self._log("Knowledge base empty")
            return AgentResponse(
                agent_name=self.name,
                content="No knowledge base documents available.",
                data={"documents": []},
            )

        docs = self._retriever.retrieve(request.query, top_k=self._top_k)
        self._log(f"Retrieved {len(docs)} document(s)")
        return AgentResponse(
            agent_name=self.name,
            content=f"Retrieved {len(docs)} document(s): "
            + ", ".join(d.name for d in docs),
            data={
                "documents": [
                    {"name": d.name, "score": d.score, "text": d.text}
                    for d in docs
                ],
            },
        )
