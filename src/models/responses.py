"""Response models returned by agents."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """The standard envelope returned by every agent.

    `content` is the user-visible answer. `data` carries structured
    artifacts (retrieved docs, code blocks, review findings) for the next
    agent in the chain. `trace` (when present) is the AgentTrace dict
    produced by a `ReasoningAgent` so the UI can render reasoning,
    selected/skipped tools, and tool invocations.
    """

    agent_name: str = Field(description="Name of the agent producing the response")
    content: str = Field(description="Free-text answer for humans / downstream agents")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload for downstream agents",
    )
    trace: dict[str, Any] | None = Field(
        default=None,
        description="Serialized AgentTrace — reasoning, tool decisions, invocations",
    )
    success: bool = Field(default=True, description="Whether the agent completed successfully")
    error: str | None = Field(default=None, description="Error message on failure")


class ReviewSeverity(str, Enum):
    """Severity of a single review finding."""

    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class ReviewFinding(BaseModel):
    """A single observation produced by the ReviewAgent."""

    category: str = Field(description="Category, e.g. 'security', 'tests', 'clarity'")
    severity: ReviewSeverity = Field(description="Severity of the finding")
    message: str = Field(description="Human-readable explanation")


class ReviewResult(BaseModel):
    """Structured output produced by the ReviewAgent.

    Aggregates findings into a list and gives an overall verdict so the
    supervisor can decide whether to surface a warning to the user.
    """

    approved: bool = Field(description="True if no blocker-severity findings exist")
    summary: str = Field(description="One-line summary of the review")
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="Ordered list of findings",
    )

    @property
    def has_blockers(self) -> bool:
        """Whether any finding has BLOCKER severity."""
        return any(f.severity == ReviewSeverity.BLOCKER for f in self.findings)
