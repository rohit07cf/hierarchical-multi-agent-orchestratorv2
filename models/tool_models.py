"""Tool result and tool call models for structured tool interactions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Structured representation of an LLM-generated tool call.

    Used with OpenAI Structured Outputs to ensure type-safe tool invocations
    with explicit reasoning about why a tool was selected.
    """

    tool_name: str = Field(description="Name of the tool to call")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the tool",
    )
    reasoning: str = Field(
        default="",
        description="LLM's reasoning for selecting this tool and parameters",
    )


class ToolResult(BaseModel):
    """Structured output from a tool execution.

    Provides a consistent interface for all tool results, including
    success/failure status and optional error information.
    """

    success: bool = Field(description="Whether the tool executed successfully")
    result: Any = Field(default=None, description="The tool's output value")
    error: str | None = Field(
        default=None,
        description="Error message if the tool failed",
    )
    tool_name: str = Field(default="", description="Name of the tool that was called")
    execution_time_ms: float | None = Field(
        default=None,
        description="Execution time in milliseconds",
    )

    @classmethod
    def ok(cls, result: Any, tool_name: str = "", execution_time_ms: float | None = None) -> ToolResult:
        """Create a successful tool result."""
        return cls(
            success=True,
            result=result,
            tool_name=tool_name,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def fail(cls, error: str, tool_name: str = "") -> ToolResult:
        """Create a failed tool result."""
        return cls(
            success=False,
            error=error,
            tool_name=tool_name,
        )


class ToolDefinition(BaseModel):
    """Metadata about an available tool for agent registration."""

    name: str = Field(description="Tool function name")
    description: str = Field(description="Human-readable description of the tool")
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for the tool's parameters",
    )
