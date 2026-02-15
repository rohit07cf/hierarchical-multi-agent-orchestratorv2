"""Supervisor output and subtask result models for structured orchestration responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubtaskStatus(str, Enum):
    """Status of a subtask delegated to a child agent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubtaskResult(BaseModel):
    """Result from a child agent's execution of a delegated subtask.

    Captures the agent responsible, the subtask description, execution result,
    and current status for aggregation by the supervisor.
    """

    agent_name: str = Field(description="Name of the child agent that executed this subtask")
    subtask: str = Field(description="Description of the subtask delegated to the agent")
    result: Any = Field(default=None, description="Result produced by the child agent")
    status: SubtaskStatus = Field(
        default=SubtaskStatus.PENDING,
        description="Current execution status of this subtask",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the subtask failed",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool calls made during subtask execution",
    )

    @property
    def is_success(self) -> bool:
        """Check if this subtask completed successfully."""
        return self.status == SubtaskStatus.COMPLETED and self.error is None


class TaskDecomposition(BaseModel):
    """Structured representation of the supervisor's task decomposition plan."""

    original_request: str = Field(description="The original user request")
    reasoning: str = Field(description="Supervisor's reasoning about how to decompose the task")
    subtasks: list[PlannedSubtask] = Field(
        default_factory=list,
        description="Ordered list of planned subtasks",
    )


class PlannedSubtask(BaseModel):
    """A planned subtask before execution."""

    agent_name: str = Field(description="Target agent for this subtask")
    description: str = Field(description="What the agent should do")
    tools_needed: list[str] = Field(
        default_factory=list,
        description="Tools the agent will likely need",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices of subtasks this one depends on",
    )


class SupervisorOutput(BaseModel):
    """Final structured output from the supervisor after orchestrating all child agents.

    Aggregates results from multiple subtask executions into a coherent
    final answer with full traceability.
    """

    final_answer: str = Field(description="The aggregated final answer to the user's request")
    subtasks: list[SubtaskResult] = Field(
        default_factory=list,
        description="Results from all delegated subtasks",
    )
    decomposition: TaskDecomposition | None = Field(
        default=None,
        description="The task decomposition plan used",
    )

    @property
    def all_succeeded(self) -> bool:
        """Check if all subtasks completed successfully."""
        return all(s.is_success for s in self.subtasks)

    @property
    def failed_subtasks(self) -> list[SubtaskResult]:
        """Get list of failed subtasks."""
        return [s for s in self.subtasks if s.status == SubtaskStatus.FAILED]
