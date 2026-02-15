"""Supervisor-specific tools for task decomposition and reasoning."""

from __future__ import annotations

import time

from models.tool_models import ToolResult


def reasoning_step(
    thought: str,
    action_plan: str,
    agents_needed: str,
) -> str:
    """Execute a structured reasoning step for task decomposition.

    This tool is used by the Supervisor agent to reason through task
    decomposition in a structured way, documenting which agents are
    needed and what the execution plan should be.

    Args:
        thought: The supervisor's analysis of the user request.
        action_plan: The planned sequence of agent delegations.
        agents_needed: Comma-separated list of agents required.

    Returns:
        JSON string with the structured reasoning output.
    """
    start = time.perf_counter()

    agents_list = [a.strip() for a in agents_needed.split(",") if a.strip()]

    result = {
        "thought": thought,
        "action_plan": action_plan,
        "agents_needed": agents_list,
        "reasoning_complete": True,
    }

    elapsed = (time.perf_counter() - start) * 1000
    return ToolResult.ok(
        result=result,
        tool_name="reasoning_step",
        execution_time_ms=elapsed,
    ).model_dump_json()
