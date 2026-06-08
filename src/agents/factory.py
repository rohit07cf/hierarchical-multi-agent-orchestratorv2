"""Build the SDK-native agent graph (Supervisor → Managers → Workers).

All agents call Claude through LiteLLM's in-process adapter (`LitellmModel`).
Delegation uses `agent.as_tool(...)` (wrapped in `traced_as_tool` so each
sub-agent's reasoning trace is captured for the UI). The supervisor's routing
rules — formerly the deterministic `Router` + `_PLANNER_SYSTEM_PROMPT` — now
live in the supervisor Agent's `instructions`.
"""

from __future__ import annotations

import os

from agents import Agent, Model, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from src.agents.sdk_tools import (
    code_generation_tool,
    code_review_tool,
    file_context_tool,
    load_knowledge_base_tool,
    review_instructions,
    security_review_tool,
    simple_retriever,
    summarizer_instructions,
    template_loader,
    test_gap_tool,
    traced_as_tool,
)

# The SDK's own tracing exporter targets OpenAI; we run no OpenAI key and use
# our own OpenTelemetry pipeline, so disable it once at import.
set_tracing_disabled(True)

DEFAULT_MODEL = "claude-opus-4-8"

# ----------------- Static maps (reused by the result mapper) -----------------

RAG_TOOLS = ["simple_retriever", "load_knowledge_base"]
CODING_TOOLS = ["code_generation_tool", "template_loader", "file_context_tool"]
REVIEW_TOOLS = ["code_review_tool", "security_review_tool", "test_gap_tool"]
RESEARCH_WORKER_TOOLS = ["call_rag_agent", "call_summarizer_agent"]
BUILD_WORKER_TOOLS = ["call_coding_agent", "call_review_agent"]
MANAGER_TOOLS = ["ResearchManagerAgent", "BuildManagerAgent"]

# Which workers sit under each manager (for grouping traces into subtasks).
MANAGER_WORKERS: dict[str, list[str]] = {
    "ResearchManagerAgent": ["RAGAgent", "SummarizerAgent"],
    "BuildManagerAgent": ["CodingAgent", "ReviewAgent"],
}
# Surfaced as PlannedSubtask.tools_needed in the reconstructed decomposition.
TOOLS_BY_MANAGER = MANAGER_WORKERS


# ----------------- Model -----------------


def make_model(model_name: str = DEFAULT_MODEL) -> LitellmModel:
    """Return a LiteLLM-backed model that routes `model_name` to Claude."""
    return LitellmModel(
        model=f"anthropic/{model_name}",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


# ----------------- Instructions -----------------

_SUPERVISOR_INSTRUCTIONS = (
    "You are the RootSupervisorAgent at the top of a 3-layer hierarchy. Route "
    "the user's request to the right manager tool(s):\n"
    "- ResearchManagerAgent: reflective, philosophical, summarization, "
    "explanation, search, knowledge-base, or architecture questions.\n"
    "- BuildManagerAgent: explicit code/implementation requests (build, "
    "implement, fastapi, redis, write code, review code).\n"
    "- Combined requests: call ResearchManagerAgent first, then "
    "BuildManagerAgent.\n"
    "Call each manager at most once. After the manager(s) return, write the "
    "final answer for the user. Do not mention internal agent or tool names."
)

_RESEARCH_INSTRUCTIONS = (
    "You are ResearchManagerAgent. Worker tools: call_rag_agent (retrieves "
    "knowledge-base documents) and call_summarizer_agent (writes the summary). "
    "If the request needs external/project context, call call_rag_agent FIRST "
    "to retrieve documents, THEN call call_summarizer_agent to summarize them. "
    "If the user pasted text to summarize directly, skip retrieval and just "
    "call call_summarizer_agent. Return the summary."
)

_RAG_INSTRUCTIONS = (
    "You are RAGAgent. Call simple_retriever to find the most relevant local "
    "knowledge-base documents for the query (prefer it when the query has "
    "topical keywords). Use load_knowledge_base only for exploratory "
    "'what documents exist' queries."
)

_BUILD_INSTRUCTIONS = (
    "You are BuildManagerAgent. Worker tools: call_coding_agent (generates "
    "code) and call_review_agent (audits it). Call call_coding_agent first, "
    "then call call_review_agent as the default quality gate unless the user "
    "says 'no review'. Return the code plus the review summary."
)

_CODING_INSTRUCTIONS = (
    "You are CodingAgent. Call template_loader for a known named template "
    "(e.g. fastapi_upload_endpoint, redis_memory_tool), code_generation_tool "
    "for a generic skeleton, and/or file_context_tool to ground the code in "
    "existing project files. Return the code."
)


# ----------------- Graph -----------------


def build_supervisor(
    model: Model | None = None, *, hitl: bool = False, hooks: object = None
) -> Agent:
    """Build and return the root supervisor Agent.

    Args:
        model: A `Model` instance applied to every agent (production passes a
            `LitellmModel`). When `None`, agents have no bound model and a
            `RunConfig(model_provider=...)` must supply one at run time (tests).
        hitl: When True, the manager tools require approval (pause/resume).
        hooks: Optional `RunHooks` threaded into every `as_tool` sub-run so the
            observability/streaming hooks fire for managers and workers too.
    """
    rag = Agent(
        name="RAGAgent",
        model=model,
        instructions=_RAG_INSTRUCTIONS,
        tools=[simple_retriever, load_knowledge_base_tool],
        tool_use_behavior="stop_on_first_tool",  # retrieval-only, no synthesis pass
    )
    summarizer = Agent(
        name="SummarizerAgent",
        model=model,
        instructions=summarizer_instructions,  # dynamic: injects retrieved docs
        tools=[],
    )
    coding = Agent(
        name="CodingAgent",
        model=model,
        instructions=_CODING_INSTRUCTIONS,
        tools=[code_generation_tool, template_loader, file_context_tool],
    )
    review = Agent(
        name="ReviewAgent",
        model=model,
        instructions=review_instructions,  # dynamic: injects generated code
        tools=[code_review_tool, security_review_tool, test_gap_tool],
    )

    research_mgr = Agent(
        name="ResearchManagerAgent",
        model=model,
        instructions=_RESEARCH_INSTRUCTIONS,
        tools=[
            traced_as_tool(
                rag,
                tool_name="call_rag_agent",
                tool_description="Retrieve relevant knowledge-base documents.",
                available_tools=RAG_TOOLS,
                hooks=hooks,
                output="retrieval",
            ),
            traced_as_tool(
                summarizer,
                tool_name="call_summarizer_agent",
                tool_description="Summarize the retrieved documents or pasted text.",
                available_tools=[],
                hooks=hooks,
            ),
        ],
    )
    build_mgr = Agent(
        name="BuildManagerAgent",
        model=model,
        instructions=_BUILD_INSTRUCTIONS,
        tools=[
            traced_as_tool(
                coding,
                tool_name="call_coding_agent",
                tool_description="Generate code for the request.",
                available_tools=CODING_TOOLS,
                hooks=hooks,
            ),
            traced_as_tool(
                review,
                tool_name="call_review_agent",
                tool_description="Review the generated code for quality and security.",
                available_tools=REVIEW_TOOLS,
                hooks=hooks,
            ),
        ],
    )

    supervisor = Agent(
        name="RootSupervisorAgent",
        model=model,
        instructions=_SUPERVISOR_INSTRUCTIONS,
        tools=[
            traced_as_tool(
                research_mgr,
                tool_name="ResearchManagerAgent",
                tool_description="Coordinate retrieval + summarization.",
                available_tools=RESEARCH_WORKER_TOOLS,
                hooks=hooks,
                needs_approval=hitl,
            ),
            traced_as_tool(
                build_mgr,
                tool_name="BuildManagerAgent",
                tool_description="Coordinate code generation + review.",
                available_tools=BUILD_WORKER_TOOLS,
                hooks=hooks,
                needs_approval=hitl,
            ),
        ],
    )
    return supervisor
