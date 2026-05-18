"""State models tracking the lifecycle of an orchestration run."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle status of an individual subtask."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTask(BaseModel):
    """A single unit of work the supervisor assigns to a manager or worker."""

    agent_name: str = Field(description="Target agent (e.g. ResearchManagerAgent)")
    description: str = Field(description="What the agent should do")
    tools_needed: list[str] = Field(default_factory=list, description="Tools the agent will likely use")
    depends_on: list[int] = Field(
        default_factory=list,
        description="Indices of tasks this one depends on (for sequencing)",
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Any = Field(default=None, description="Result payload after execution")
    error: str | None = Field(default=None)

    @property
    def is_success(self) -> bool:
        """Whether the task completed without error."""
        return self.status == TaskStatus.COMPLETED and self.error is None


class ExecutionPlan(BaseModel):
    """The supervisor's decomposition of the user query into tasks."""

    original_request: str = Field(description="The original user query")
    reasoning: str = Field(description="Why the supervisor chose this plan")
    tasks: list[AgentTask] = Field(default_factory=list, description="Ordered list of tasks")

    @property
    def assigned_agents(self) -> list[str]:
        """Distinct agent names involved in the plan, preserving order."""
        seen: list[str] = []
        for task in self.tasks:
            if task.agent_name not in seen:
                seen.append(task.agent_name)
        return seen


class ExecutionStepKind(str, Enum):
    """Categories of events recorded on the execution timeline."""

    TASK_DECOMPOSITION = "task_decomposition"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETE = "subtask_complete"
    ORCHESTRATION_COMPLETE = "orchestration_complete"
    INFO = "info"
    ERROR = "error"


class ExecutionStep(BaseModel):
    """A single record on the execution timeline.

    Streamlit renders these chronologically in the "Execution Steps" panel.
    Each step is intentionally minimal — agent name, what happened, and
    optional structured payload — so the UI can stay generic.
    """

    step_number: int = Field(description="1-based ordinal")
    agent_name: str = Field(description="Agent that emitted the step")
    kind: ExecutionStepKind = Field(description="Category of step")
    message: str = Field(description="Human-readable summary, used as the log line")
    payload: dict[str, Any] = Field(default_factory=dict, description="Optional structured data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrchestratorState(BaseModel):
    """Complete runtime state for one orchestration run.

    Bundles the original request, the plan, the execution timeline and the
    final answer. This is what the Streamlit "State Inspector" reads from.
    """

    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = Field(description="The original user query")
    plan: ExecutionPlan | None = Field(default=None)
    steps: list[ExecutionStep] = Field(default_factory=list)
    status: str = Field(default="initialized")
    current_tool: str | None = Field(default=None)
    tool_path: str = Field(default="RootSupervisorAgent")
    final_answer: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def iteration_count(self) -> int:
        """Number of steps recorded so far — surfaced in the state inspector."""
        return len(self.steps)

    def add_step(
        self,
        agent_name: str,
        kind: ExecutionStepKind,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionStep:
        """Append an execution step and update the state's `updated_at`."""
        step = ExecutionStep(
            step_number=len(self.steps) + 1,
            agent_name=agent_name,
            kind=kind,
            message=message,
            payload=payload or {},
        )
        self.steps.append(step)
        self.updated_at = datetime.now(timezone.utc)
        return step
