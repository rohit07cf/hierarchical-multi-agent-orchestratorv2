"""Text manipulation tools for the EchoAgent and SimpleAgent."""

from __future__ import annotations

import time

from models.tool_models import ToolResult


def echo_text(text: str) -> str:
    """Echo back the provided text unchanged.

    Args:
        text: The text to echo.

    Returns:
        JSON string with the echoed text.
    """
    start = time.perf_counter()
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=text,
        tool_name="echo_text",
        execution_time_ms=elapsed,
    ).model_dump_json()


def reverse_text(text: str) -> str:
    """Reverse the provided text and return it.

    Args:
        text: The text to reverse.

    Returns:
        JSON string with the reversed text.
    """
    start = time.perf_counter()
    reversed_text = text[::-1]
    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=reversed_text,
        tool_name="reverse_text",
        execution_time_ms=elapsed,
    ).model_dump_json()
