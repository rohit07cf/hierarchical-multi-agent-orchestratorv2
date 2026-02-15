"""Pydantic validators and validation utilities."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from models.agent_state import AgentState, HITLAction
from models.supervisor_output import SupervisorOutput
from models.tool_models import ToolCall, ToolResult


def validate_agent_state(data: dict[str, Any]) -> AgentState | None:
    """Validate and parse an AgentState from a dictionary.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated AgentState or None if validation fails.
    """
    try:
        return AgentState.model_validate(data)
    except ValidationError:
        return None


def validate_hitl_action(data: dict[str, Any]) -> HITLAction | None:
    """Validate and parse an HITLAction from a dictionary.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated HITLAction or None if validation fails.
    """
    try:
        return HITLAction.model_validate(data)
    except ValidationError:
        return None


def validate_tool_call(data: dict[str, Any]) -> ToolCall | None:
    """Validate and parse a ToolCall from a dictionary.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated ToolCall or None if validation fails.
    """
    try:
        return ToolCall.model_validate(data)
    except ValidationError:
        return None


def validate_tool_result(data: dict[str, Any]) -> ToolResult | None:
    """Validate and parse a ToolResult from a dictionary.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated ToolResult or None if validation fails.
    """
    try:
        return ToolResult.model_validate(data)
    except ValidationError:
        return None


def validate_supervisor_output(data: dict[str, Any]) -> SupervisorOutput | None:
    """Validate and parse a SupervisorOutput from a dictionary.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated SupervisorOutput or None if validation fails.
    """
    try:
        return SupervisorOutput.model_validate(data)
    except ValidationError:
        return None
