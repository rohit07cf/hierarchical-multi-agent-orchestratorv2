"""Mathematical operation tools for the MathAgent and SimpleAgent."""

from __future__ import annotations

import time

from models.tool_models import ToolResult


def add_numbers(a: float, b: float) -> str:
    """Add two numbers together and return the result.

    Args:
        a: First number.
        b: Second number.

    Returns:
        JSON string with the addition result.
    """
    start = time.perf_counter()
    result = a + b
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=result,
        tool_name="add_numbers",
        execution_time_ms=elapsed,
    ).model_dump_json()


def subtract_numbers(a: float, b: float) -> str:
    """Subtract the second number from the first and return the result.

    Args:
        a: Number to subtract from.
        b: Number to subtract.

    Returns:
        JSON string with the subtraction result.
    """
    start = time.perf_counter()
    result = a - b
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=result,
        tool_name="subtract_numbers",
        execution_time_ms=elapsed,
    ).model_dump_json()


def multiply_numbers(a: float, b: float) -> str:
    """Multiply two numbers together and return the result.

    Args:
        a: First number.
        b: Second number.

    Returns:
        JSON string with the multiplication result.
    """
    start = time.perf_counter()
    result = a * b
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=result,
        tool_name="multiply_numbers",
        execution_time_ms=elapsed,
    ).model_dump_json()
