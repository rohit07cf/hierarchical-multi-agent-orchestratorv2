"""End-to-end tests for the RootSupervisorAgent."""

from __future__ import annotations

import pytest

from src.models.state_models import ExecutionStepKind
from src.orchestrator.supervisor import RootSupervisorAgent


@pytest.mark.asyncio
async def test_research_only_query_creates_single_task() -> None:
    supervisor = RootSupervisorAgent()
    state, response = await supervisor.orchestrate(
        "Summarize the architecture of this project."
    )
    assert state.status == "completed"
    assert state.plan is not None
    assert [t.agent_name for t in state.plan.tasks] == ["ResearchManagerAgent"]
    assert response.content


@pytest.mark.asyncio
async def test_build_only_query_creates_single_task() -> None:
    supervisor = RootSupervisorAgent()
    state, response = await supervisor.orchestrate(
        "Build a FastAPI endpoint for uploading documents."
    )
    assert state.status == "completed"
    assert state.plan is not None
    assert [t.agent_name for t in state.plan.tasks] == ["BuildManagerAgent"]
    assert "fastapi" in response.content.lower()


@pytest.mark.asyncio
async def test_combined_query_runs_both_managers_in_order() -> None:
    supervisor = RootSupervisorAgent()
    state, _ = await supervisor.orchestrate(
        "Search the knowledge base and generate implementation guidance."
    )
    assert state.plan is not None
    assert [t.agent_name for t in state.plan.tasks] == [
        "ResearchManagerAgent",
        "BuildManagerAgent",
    ]
    assert all(t.is_success for t in state.plan.tasks)


@pytest.mark.asyncio
async def test_execution_timeline_records_expected_step_kinds() -> None:
    supervisor = RootSupervisorAgent()
    state, _ = await supervisor.orchestrate("Build a FastAPI endpoint.")
    kinds = [s.kind for s in state.steps]
    assert ExecutionStepKind.TASK_DECOMPOSITION in kinds
    assert ExecutionStepKind.SUBTASK_STARTED in kinds
    assert ExecutionStepKind.SUBTASK_COMPLETE in kinds
    assert ExecutionStepKind.ORCHESTRATION_COMPLETE in kinds
    # Steps should be in chronological order.
    assert [s.step_number for s in state.steps] == list(range(1, len(state.steps) + 1))


@pytest.mark.asyncio
async def test_subtask_complete_steps_carry_agent_trace() -> None:
    """Every manager invocation must surface its reasoning trace on the timeline."""
    supervisor = RootSupervisorAgent()
    state, _ = await supervisor.orchestrate("Build a FastAPI endpoint.")
    completed_steps = [
        s for s in state.steps if s.kind == ExecutionStepKind.SUBTASK_COMPLETE
    ]
    assert completed_steps, "expected at least one subtask_complete step"
    for step in completed_steps:
        trace = step.payload.get("trace")
        assert trace is not None, (
            f"subtask_complete for {step.agent_name} missing reasoning trace"
        )
        assert trace["agent_name"] == step.agent_name
        # Manager trace should expose nested worker traces too.
        # (BuildManagerAgent → CodingAgent / ReviewAgent)
        worker_traces = step.payload.get("trace", {}).get("tool_invocations", [])
        assert isinstance(worker_traces, list)


@pytest.mark.asyncio
async def test_philosophical_input_runs_summarizer_only_end_to_end() -> None:
    """The 'meaning of life' query must not invoke BuildManager OR RAGAgent."""
    supervisor = RootSupervisorAgent()
    state, _ = await supervisor.orchestrate(
        "The meaning of life is to live, to understand, and to create "
        "something meaningful to share with others."
    )
    assert state.plan is not None
    manager_names = [t.agent_name for t in state.plan.tasks]
    assert manager_names == ["ResearchManagerAgent"]

    research_task = state.plan.tasks[0]
    research_trace = research_task.result.get("trace", {})
    assert "call_summarizer_agent" in research_trace["selected_tools"]
    assert "call_rag_agent" not in research_trace["selected_tools"], (
        "RAGAgent must not be invoked for pasted philosophical text"
    )


@pytest.mark.asyncio
async def test_pre_subtask_hook_can_pause_execution() -> None:
    supervisor = RootSupervisorAgent()
    call_count = {"n": 0}

    async def block_first_subtask(idx, task, state) -> bool:
        call_count["n"] += 1
        return False  # never proceed — emulates a HITL pause

    state, response = await supervisor.orchestrate(
        "Build a FastAPI endpoint.", pre_subtask=block_first_subtask
    )
    assert call_count["n"] == 1
    assert state.status == "paused"
    assert response.content == "Execution paused for review."
