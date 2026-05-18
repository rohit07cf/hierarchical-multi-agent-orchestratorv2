"""BuildManagerAgent — coordinates code generation + review."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.coding_agent import CodingAgent
from src.agents.review_agent import ReviewAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse


class BuildManagerAgent(BaseAgent):
    """Run the Code → Review chain for an implementation request.

    Receives any prior context (e.g. retrieved docs from
    ResearchManagerAgent) so the CodingAgent can ground its snippet.
    """

    name = "BuildManagerAgent"
    tools: list[str] = []

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(model=model)
        self.coder = CodingAgent(model=model)
        self.reviewer = ReviewAgent(model=model)

    async def handle(self, request: AgentRequest) -> AgentResponse:
        self._log("Delegating to CodingAgent")
        code_response = await self.coder.handle(
            AgentRequest(
                query=request.query,
                context=request.context,
                parent_agent=self.name,
            )
        )

        self._log("Delegating to ReviewAgent")
        review_response = await self.reviewer.handle(
            AgentRequest(
                query=request.query,
                context={"code": code_response.data.get("code", "")},
                parent_agent=self.name,
            )
        )

        combined = (
            f"{code_response.content}\n\n"
            f"**Review:**\n{review_response.content}"
        )
        return AgentResponse(
            agent_name=self.name,
            content=combined,
            data={
                "code": code_response.data.get("code", ""),
                "language": code_response.data.get("language", "python"),
                "review": review_response.data.get("review", {}),
            },
        )
