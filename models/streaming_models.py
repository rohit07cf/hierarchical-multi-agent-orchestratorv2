"""Streaming response models for real-time UI updates."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StreamingStatus(str, Enum):
    """Status of a streaming response step."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    REASONING = "reasoning"
    COMPLETED = "completed"
    ERROR = "error"
    HITL_PAUSED = "hitl_paused"


class StreamingModelResponseStep(BaseModel):
    """A single step in a streaming model response for real-time UI updates.

    Used by the StreamingCallbackHandler to push incremental updates
    to the Streamlit frontend as the agent processes.
    """

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this streaming message",
    )
    status: StreamingStatus = Field(description="Current status of this step")
    name: str = Field(
        default="",
        description="Name of the agent or tool producing this step",
    )
    message_fragment: str = Field(
        default="",
        description="Incremental text fragment for streaming display",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool params, agent info, etc.)",
    )

    @classmethod
    def token(cls, name: str, fragment: str) -> StreamingModelResponseStep:
        """Create a token streaming step."""
        return cls(
            status=StreamingStatus.IN_PROGRESS,
            name=name,
            message_fragment=fragment,
        )

    @classmethod
    def tool_call(cls, name: str, tool_name: str, params: dict[str, Any]) -> StreamingModelResponseStep:
        """Create a tool call notification step."""
        return cls(
            status=StreamingStatus.TOOL_CALLED,
            name=name,
            message_fragment=f"Calling tool: {tool_name}",
            metadata={"tool_name": tool_name, "parameters": params},
        )

    @classmethod
    def tool_result(cls, name: str, tool_name: str, result: Any) -> StreamingModelResponseStep:
        """Create a tool result notification step."""
        return cls(
            status=StreamingStatus.TOOL_COMPLETED,
            name=name,
            message_fragment=f"Tool {tool_name} completed",
            metadata={"tool_name": tool_name, "result": str(result)},
        )

    @classmethod
    def error(cls, name: str, error_msg: str) -> StreamingModelResponseStep:
        """Create an error notification step."""
        return cls(
            status=StreamingStatus.ERROR,
            name=name,
            message_fragment=error_msg,
        )

    @classmethod
    def hitl_pause(cls, name: str, reason: str) -> StreamingModelResponseStep:
        """Create a HITL pause notification step."""
        return cls(
            status=StreamingStatus.HITL_PAUSED,
            name=name,
            message_fragment=f"Paused for human review: {reason}",
        )
