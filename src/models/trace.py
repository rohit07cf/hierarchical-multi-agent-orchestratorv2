"""Models capturing agent reasoning, tool decisions, and execution traces.

These are populated by every `ReasoningAgent` and surfaced on the
Streamlit "Agent Reasoning Trace" panel so the user can see *why* an
agent did what it did — not just what it returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class LLMMode(str, Enum):
    """Whether reasoning is performed by a real LLM or the offline mock."""

    REAL = "real"
    MOCK = "mock"


@dataclass(frozen=True)
class ToolSpec:
    """Static description of a tool an agent can choose to invoke.

    The `handler` is the deterministic Python callable executed when the
    agent (or its LLM) decides to call it. `parameters_schema` mirrors a
    JSON-Schema-like description so the LLM can fill in arguments.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    parameters_schema: dict[str, Any] | None = None


class ToolDecision(BaseModel):
    """A single tool the agent has decided to call, with rationale."""

    tool_name: str = Field(description="Name of the tool to invoke")
    rationale: str = Field(description="Why this tool was selected")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class AgentReasoning(BaseModel):
    """The agent's pre-execution reasoning output.

    Captures the chain-of-thought summary, which tools the agent decided
    to call (and why), and which were considered but skipped. The
    Streamlit trace panel renders these three sections directly.
    """

    reasoning: str = Field(description="Summary of the agent's analysis")
    selected_tools: list[ToolDecision] = Field(
        default_factory=list, description="Tools the agent chose to call, in order"
    )
    skipped_tools: list[str] = Field(
        default_factory=list,
        description="Tools considered but not invoked (with implicit rationale = 'not needed')",
    )


class ToolInvocation(BaseModel):
    """Record of a single tool call executed during agent handling.

    `result` carries the full structured payload (consumed by the next
    agent in the chain). `result_preview` is a short string the UI
    renders verbatim.
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    result: Any = Field(
        default=None,
        description="Full tool result, used by downstream agents",
    )
    result_preview: str = Field(
        default="",
        description="Truncated string preview of the tool result for the UI",
    )
    success: bool = True
    error: str | None = None


class AgentTrace(BaseModel):
    """Full execution trace for a single agent invocation.

    Populated by `ReasoningAgent.handle()` and attached to every
    `AgentResponse.trace`. Persists into `OrchestratorState.steps`
    payloads so the Streamlit state inspector can replay it.
    """

    agent_name: str
    llm_mode: LLMMode
    reasoning: str
    available_tools: list[str]
    selected_tools: list[str]
    skipped_tools: list[str]
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    final_response: str = ""

    @property
    def is_mock(self) -> bool:
        """Whether this trace came from the offline mock LLM."""
        return self.llm_mode == LLMMode.MOCK
