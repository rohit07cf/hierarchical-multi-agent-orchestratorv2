"""EchoAgent: handles string manipulation and text operations."""

from __future__ import annotations

from typing import Any

from agents import function_tool

from agents.base_agent import BaseAgent
from prompts.react_prompt import get_react_prompt
from models.tool_models import ToolResult


@function_tool
def echo_agent_echo_text(text: str) -> str:
    """Echo back the provided text unchanged.

    Args:
        text: The text to echo back.
    """
    return ToolResult.ok(result=text, tool_name="echo_text").model_dump_json()


@function_tool
def reverse_text(text: str) -> str:
    """Reverse the provided text and return it.

    Args:
        text: The text to reverse.
    """
    reversed_text = text[::-1]
    return ToolResult.ok(result=reversed_text, tool_name="reverse_text").model_dump_json()


class EchoAgentDef(BaseAgent):
    """Agent for string manipulation and text operations.

    Equipped with echo_text and reverse_text tools.
    """

    TOOLS = ["echo_text", "reverse_text"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(name="EchoAgent", model=model)

    def _get_system_prompt(self) -> str:
        return get_react_prompt("EchoAgent", self.TOOLS)

    def _register_tools(self) -> list[Any]:
        return [echo_agent_echo_text, reverse_text]
