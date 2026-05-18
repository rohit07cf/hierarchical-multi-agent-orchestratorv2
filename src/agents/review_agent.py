"""ReviewAgent — reviews generated code for production-readiness issues."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.tools.code_review_tool import review_code


class ReviewAgent(BaseAgent):
    """Review the latest code snippet for bugs, security and clarity.

    Pulls the code from `request.context['code']` (as produced by
    `CodingAgent`) and runs the deterministic `review_code` heuristic.
    """

    name = "ReviewAgent"
    tools = ["review_code"]

    async def handle(self, request: AgentRequest) -> AgentResponse:
        code = request.context.get("code", "")
        result = review_code(code)
        self._log(f"Review completed: {result.summary}")

        if result.findings:
            bullet_list = "\n".join(
                f"- [{f.severity.value}] ({f.category}) {f.message}"
                for f in result.findings
            )
        else:
            bullet_list = "- No issues detected."

        content = f"{result.summary}\n{bullet_list}"
        return AgentResponse(
            agent_name=self.name,
            content=content,
            data={"review": result.model_dump(mode="json")},
        )
