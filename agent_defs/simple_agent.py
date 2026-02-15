"""SimpleAgent: handles basic mathematical operations and text echoing."""

from __future__ import annotations

from typing import Any

from agents import function_tool

from agent_defs.base_agent import BaseAgent
from prompts.react_prompt import get_react_prompt
from models.tool_models import ToolResult


@function_tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers together and return the result.

    Args:
        a: First number.
        b: Second number.
    """
    result = a + b
    return ToolResult.ok(result=result, tool_name="add_numbers").model_dump_json()


@function_tool
def echo_text(text: str) -> str:
    """Echo back the provided text unchanged.

    Args:
        text: The text to echo back.
    """
    return ToolResult.ok(result=text, tool_name="echo_text").model_dump_json()


class SimpleAgentDef(BaseAgent):
    """Agent for simple mathematical operations and text echoing.

    Equipped with add_numbers and echo_text tools for basic tasks.
    """

    TOOLS = ["add_numbers", "echo_text"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(name="SimpleAgent", model=model)

    def _get_system_prompt(self) -> str:
        return get_react_prompt("SimpleAgent", self.TOOLS)

    def _register_tools(self) -> list[Any]:
        return [add_numbers, echo_text]
