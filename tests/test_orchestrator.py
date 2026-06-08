"""Bridge-level tests: SupervisorAgent (the UI contract) over the SDK graph.

`make_model` is monkeypatched to a scripted `StubModel`, so the whole bridge
(build graph → Runner.run → map → state mirror → HITL) runs without a network.
"""

from __future__ import annotations

import agent_defs.supervisor as bridge
from models.agent_state import HITLAction, HITLActionType, HITLCheckpointType
from models.supervisor_output import SupervisorOutput
from orchestration.hitl_manager import HITLManager
from tests.stub_model import StubModel, func_call, message

_RESEARCH = "Summarize the architecture of this project."
_BUILD = "Build a FastAPI endpoint and review it."


def _research_script(final: str = "Final summary.") -> dict:
    return {
        "supervisor": [func_call("ResearchManagerAgent", input=_RESEARCH), message(final)],
        "research": [
            func_call("call_rag_agent", input="architecture"),
            func_call("call_summarizer_agent", input="summarize"),
            message("Research summary."),
        ],
        "rag": [func_call("simple_retriever", query="architecture", top_k=2)],
        "summarizer": [message("This project uses a 3-layer hierarchical pattern.")],
    }


def _build_script() -> dict:
    return {
        "supervisor": [func_call("BuildManagerAgent", input=_BUILD), message("Code + review.")],
        "build": [
            func_call("call_coding_agent", input="fastapi"),
            func_call("call_review_agent", input="review"),
            message("Build summary."),
        ],
        "coding": [
            func_call("template_loader", template_name="fastapi_upload_endpoint"),
            message("Here is the endpoint."),
        ],
        "review": [func_call("code_review_tool", code="def x(): pass"), message("Reviewed.")],
    }


async def test_bridge_research_query(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "make_model", lambda m: StubModel(_research_script()))
    sup = bridge.SupervisorAgent()
    out = await sup.orchestrate_manual(_RESEARCH)
    assert isinstance(out, SupervisorOutput)
    assert out.final_answer == "Final summary."
    assert [s.agent_name for s in out.subtasks] == ["ResearchManagerAgent"]
    # the State Inspector mirror is populated
    assert sup.state.intermediate_steps


async def test_bridge_build_query_routes_to_build_manager(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "make_model", lambda m: StubModel(_build_script()))
    sup = bridge.SupervisorAgent()
    out = await sup.orchestrate_manual(_BUILD)
    assert [s.agent_name for s in out.subtasks] == ["BuildManagerAgent"]
    assert out.final_answer == "Code + review."


async def test_observability_metrics_populate_per_request(monkeypatch) -> None:
    """A run must record per-agent metrics (panel un-hides) + a request summary."""
    monkeypatch.setattr(bridge, "make_model", lambda m: StubModel(_research_script()))
    sup = bridge.SupervisorAgent()
    await sup.orchestrate_manual(_RESEARCH)

    from src.observability.metrics.snapshot import collect_snapshot

    snap = collect_snapshot()
    assert not snap.is_empty, "panel must un-hide after a run"
    invs = snap.counter_by_label("agent_invocations_total", "agent")
    assert {"RootSupervisorAgent", "ResearchManagerAgent", "RAGAgent"} <= set(invs)
    # per-agent LLM token attribution (operation label == agent name)
    assert snap.counter_by_label("llm_tokens_input_total", "operation")
    assert snap.counter_by_label("routing_decisions_total", "next_manager")

    m = sup.last_request_metrics
    assert m and m["total_tokens"] > 0 and m["turns"] == 1 and m["request_id"]
    assert "RAGAgent" in {a["agent"] for a in m["per_agent"]}


async def test_hitl_pause_returns_none_and_captures_state(monkeypatch) -> None:
    """enable_hitl → manager tool needs approval → run pauses before executing."""
    monkeypatch.setattr(bridge, "make_model", lambda m: StubModel(_research_script()))
    sup = bridge.SupervisorAgent()
    hitl = HITLManager(persistence_dir=None)

    out = await sup.orchestrate(_RESEARCH, hitl_manager=hitl, enable_hitl=True)
    assert out is None, "HITL run should pause and return None"

    paused = hitl.get_all_paused()
    assert len(paused) == 1
    state = paused[0]
    assert state.checkpoint_type == HITLCheckpointType.TOOL_EXECUTION
    assert state.pending_data["pending_tool"]["agent_name"] == "ResearchManagerAgent"
    assert "_run_state" in state.pending_data  # serialized SDK RunState


async def test_hitl_resume_after_approval(monkeypatch) -> None:
    """Approving the paused manager resumes the run to a final answer."""
    stub = StubModel(_research_script("Resumed final answer."))
    monkeypatch.setattr(bridge, "make_model", lambda m: stub)  # persistent across resume
    sup = bridge.SupervisorAgent()
    hitl = HITLManager(persistence_dir=None)

    assert await sup.orchestrate(_RESEARCH, hitl_manager=hitl, enable_hitl=True) is None
    state_id = hitl.get_all_paused()[0].state_id

    hitl.apply_action(
        state_id, HITLAction(action=HITLActionType.APPROVE, reason="approved")
    )
    out = await sup.resume_orchestration(hitl, state_id)
    assert isinstance(out, SupervisorOutput)
    assert out.final_answer == "Resumed final answer."
    assert [s.agent_name for s in out.subtasks] == ["ResearchManagerAgent"]
