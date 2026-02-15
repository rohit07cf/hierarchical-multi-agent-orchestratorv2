"""Agent state and HITL action models for execution tracking and state restoration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HITLActionType(str, Enum):
    """Types of human-in-the-loop interventions."""

    CANCEL = "CANCEL"
    REVISE = "REVISE"
    APPROVE = "APPROVE"


class HITLAction(BaseModel):
    """Represents a human-in-the-loop intervention during agent execution.

    Captures the user's decision when execution is paused for review,
    including any revised inputs or reasoning for the action.
    """

    action: HITLActionType = Field(description="The HITL action taken by the user")
    input: str | None = Field(
        default=None,
        description="Revised input provided by the user (for REVISE actions)",
    )
    reason: str | None = Field(
        default=None,
        description="User's reason for the intervention",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntermediateStep(BaseModel):
    """A single step in the agent's execution trace."""

    step_number: int = Field(description="Ordinal step number")
    agent_name: str = Field(description="Agent that executed this step")
    action: str = Field(description="Action taken (tool call or reasoning)")
    action_input: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters for the action"
    )
    observation: str | None = Field(
        default=None, description="Result or observation from the action"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentState(BaseModel):
    """Complete execution state for an agent, enabling pause/resume and HITL restoration.

    Captures all information needed to serialize, persist, and restore an agent's
    execution context at any point during a multi-step workflow.
    """

    state_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this state snapshot",
    )
    current_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Current input parameters being processed",
    )
    intermediate_steps: list[IntermediateStep] = Field(
        default_factory=list,
        description="Ordered list of execution steps taken so far",
    )
    tool: str | None = Field(
        default=None,
        description="Currently selected or pending tool name",
    )
    tool_path: str = Field(
        default="",
        description="Hierarchical path (e.g. 'Supervisor.ClassifierAgent') for tree traversal",
    )
    iteration_count: int = Field(
        default=0,
        description="Number of ReAct loop iterations completed",
    )
    is_paused: bool = Field(
        default=False,
        description="Whether execution is currently paused for HITL",
    )
    hitl_actions: list[HITLAction] = Field(
        default_factory=list,
        description="History of HITL interventions applied to this state",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def pause(self) -> None:
        """Mark execution as paused for HITL review."""
        self.is_paused = True
        self.updated_at = datetime.now(timezone.utc)

    def resume(self, action: HITLAction) -> None:
        """Resume execution after HITL review with the given action."""
        self.hitl_actions.append(action)
        self.is_paused = False
        self.updated_at = datetime.now(timezone.utc)

        if action.action == HITLActionType.REVISE and action.input:
            self.current_inputs["revised_input"] = action.input

    def add_step(
        self,
        agent_name: str,
        action: str,
        action_input: dict[str, Any] | None = None,
        observation: str | None = None,
    ) -> IntermediateStep:
        """Record a new execution step."""
        step = IntermediateStep(
            step_number=len(self.intermediate_steps) + 1,
            agent_name=agent_name,
            action=action,
            action_input=action_input or {},
            observation=observation,
        )
        self.intermediate_steps.append(step)
        self.iteration_count = len(self.intermediate_steps)
        self.updated_at = datetime.now(timezone.utc)
        return step

    def to_serializable(self) -> dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> AgentState:
        """Reconstruct state from a serialized dictionary."""
        return cls.model_validate(data)
