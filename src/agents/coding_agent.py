"""CodingAgent — LLM-driven code generation.

The agent reasons over the request, picks among three deterministic
tools (code_generation_tool, template_loader, file_context_tool), then
the LLM synthesizes the final code block from their outputs.
"""

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
from src.tools.code_generation_tool import generate_skeleton
from src.tools.file_context_tool import find_related_files
from src.tools.template_loader import list_templates, load_template

_SYSTEM_PROMPT = (
    "You are CodingAgent. Decide whether to call code_generation_tool "
    "(for a generic boilerplate skeleton), template_loader (for a known "
    "named template), or file_context_tool (to ground the snippet in "
    "existing project code). You may call multiple. Then produce a "
    "single Python code block satisfying the request."
)

_TEMPLATE_BY_KEYWORD = (
    ("redis", "redis_memory_tool"),
    ("upload", "fastapi_upload_endpoint"),
    ("fastapi", "fastapi_upload_endpoint"),
    ("endpoint", "fastapi_upload_endpoint"),
)


def _select_template_for_query(query: str) -> str | None:
    q = query.lower()
    for keyword, template in _TEMPLATE_BY_KEYWORD:
        if keyword in q:
            return template
    return None


class CodingAgent(ReasoningAgent):
    name = "CodingAgent"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        super().__init__(model=model)
        self.tools = [
            ToolSpec(
                name="code_generation_tool",
                description=(
                    "Return a small Python boilerplate skeleton matched "
                    "to the query. Args: query (str), language (str)."
                ),
                handler=generate_skeleton,
            ),
            ToolSpec(
                name="template_loader",
                description=(
                    "Load a named, pre-written template "
                    f"(available: {', '.join(list_templates())}). "
                    "Args: template_name (str)."
                ),
                handler=load_template,
            ),
            ToolSpec(
                name="file_context_tool",
                description=(
                    "Return previews of project files matching the query "
                    "to ground the snippet in existing code. "
                    "Args: query (str), max_files (int)."
                ),
                handler=find_related_files,
            ),
        ]

    def user_prompt(self, request: AgentRequest) -> str:
        return f"Implementation request: {request.query}"

    def _mock_reason_policy(self, request: AgentRequest):
        template = _select_template_for_query(request.query)

        def policy() -> AgentReasoning:
            if template is not None:
                return AgentReasoning(
                    reasoning=(
                        f"Query matches a known template ({template!r}); "
                        "loading it via template_loader."
                    ),
                    selected_tools=[
                        ToolDecision(
                            tool_name="template_loader",
                            rationale=f"Pre-written template {template!r} fits the request.",
                            arguments={"template_name": template},
                        )
                    ],
                    skipped_tools=["code_generation_tool", "file_context_tool"],
                )
            return AgentReasoning(
                reasoning=(
                    "No named template matches; generating a boilerplate "
                    "skeleton via code_generation_tool."
                ),
                selected_tools=[
                    ToolDecision(
                        tool_name="code_generation_tool",
                        rationale="Need a base skeleton to start from.",
                        arguments={"query": request.query, "language": "python"},
                    )
                ],
                skipped_tools=["template_loader", "file_context_tool"],
            )

        return policy

    def _mock_synthesize(self, request, reasoning, invocations) -> str | None:
        code = self._code_from_invocations(invocations)
        if not code:
            return (
                "[mock-llm] No code produced (no tools succeeded). "
                "Set ANTHROPIC_API_KEY for real code synthesis."
            )
        return (
            "[mock-llm] Template/skeleton produced; set ANTHROPIC_API_KEY "
            f"for real code synthesis.\n```python\n{code}\n```"
        )

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        return {"code": self._code_from_invocations(invocations), "language": "python"}

    @staticmethod
    def _code_from_invocations(invocations: list[ToolInvocation]) -> str:
        """Extract the most-likely code body from any successful tool call."""
        for inv in invocations:
            if not inv.success or not isinstance(inv.result, dict):
                continue
            if inv.tool_name == "template_loader" and inv.result.get("body"):
                return inv.result["body"]
            if inv.tool_name == "code_generation_tool" and inv.result.get("body"):
                return inv.result["body"]
        return ""
