"""MathAgent: handles complex mathematical computations."""

from __future__ import annotations

from typing import Any

from agents import function_tool

from agents.base_agent import BaseAgent
from prompts.react_prompt import get_react_prompt
from models.tool_models import ToolResult


@function_tool
def math_add_numbers(a: float, b: float) -> str:
    """Add two numbers together and return the result.

    Args:
        a: First number.
        b: Second number.
    """
    result = a + b
    return ToolResult.ok(result=result, tool_name="add_numbers").model_dump_json()


@function_tool
def subtract_numbers(a: float, b: float) -> str:
    """Subtract the second number from the first and return the result.

    Args:
        a: Number to subtract from.
        b: Number to subtract.
    """
    result = a - b
    return ToolResult.ok(result=result, tool_name="subtract_numbers").model_dump_json()


@function_tool
def multiply_numbers(a: float, b: float) -> str:
    """Multiply two numbers together and return the result.

    Args:
        a: First number.
        b: Second number.
    """
    result = a * b
    return ToolResult.ok(result=result, tool_name="multiply_numbers").model_dump_json()


class MathAgentDef(BaseAgent):
    """Agent for complex mathematical computations.

    Equipped with add_numbers, subtract_numbers, and multiply_numbers tools.
    """

    TOOLS = ["add_numbers", "subtract_numbers", "multiply_numbers"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(name="MathAgent", model=model)

    def _get_system_prompt(self) -> str:
        return get_react_prompt("MathAgent", self.TOOLS)

    def _register_tools(self) -> list[Any]:
        return [math_add_numbers, subtract_numbers, multiply_numbers]
