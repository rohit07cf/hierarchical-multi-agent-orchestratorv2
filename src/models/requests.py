"""Request models used to invoke agents in the hierarchy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """A request handed to any agent in the hierarchy.

    Carries the original user query plus any context produced by upstream
    agents. Workers receive the same model their manager received, so the
    chain is uniform and easy to trace.
    """

    query: str = Field(description="The user-facing query or subtask description")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Context produced by upstream agents (retrieved docs, plans, etc.)",
    )
    parent_agent: str | None = Field(
        default=None,
        description="Name of the agent that issued this request",
    )
