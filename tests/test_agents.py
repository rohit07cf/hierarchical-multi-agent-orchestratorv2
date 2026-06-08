"""Tests for the deterministic tools and the SDK-native agent graph.

Tool primitives are tested directly (pure functions, unchanged by the SDK
migration). The agent graph is exercised end-to-end against a scripted
``StubModel`` — the real ``@function_tool`` bodies run, so the shared-context
hand-off (RAG→Summarizer, Coder→Reviewer) and trace capture are real; only the
model's tool-selection/output is stubbed (no network).
"""

from __future__ import annotations

import pytest
from agents import Runner

from agents import Agent
from src.agents.factory import build_supervisor
from src.agents.result_mapper import map_run_result
from src.agents.sdk_tools import (
    RunContext,
    _retrieval_line,
    load_knowledge_base_tool,
    simple_retriever,
)
from src.tools.code_review_tool import review_code
from src.tools.security_review_tool import scan_security
from src.tools.simple_retriever import SimpleRetriever
from src.tools.template_loader import load_template
from src.tools.test_gap_tool import scan_test_gaps
from tests.stub_model import StubModel, func_call, message


# ----------------- Tool primitives (deterministic) -----------------


def test_review_code_warns_on_missing_error_handling() -> None:
    result = review_code("def f():\n    return 1\n")
    assert any(f.category == "bugs" for f in result.findings)


def test_security_review_flags_hardcoded_secret() -> None:
    result = scan_security('API_KEY = "abc-123-def"\n')
    assert result["finding_count"] >= 1
    assert any(f["category"] == "security" for f in result["findings"])


def test_test_gap_tool_lists_untested_functions() -> None:
    result = scan_test_gaps("def foo():\n    return 1\nclass Bar:\n    pass\n")
    assert "foo" in result["functions"]
    assert "Bar" in result["classes"]
    assert result["has_tests"] is False
    assert any(f["category"] == "tests" for f in result["findings"])


def test_template_loader_returns_known_template() -> None:
    result = load_template("fastapi_upload_endpoint")
    assert result["found"] is True
    assert "FastAPI" in result["body"]


def test_template_loader_handles_unknown_template() -> None:
    result = load_template("does_not_exist")
    assert result["found"] is False
    assert result["body"] == ""
    assert isinstance(result["available"], list)


def test_simple_retriever_returns_top_k_sorted() -> None:
    retriever = SimpleRetriever({
        "a.md": "supervisor agent orchestration",
        "b.md": "completely unrelated content about cats",
        "c.md": "agent orchestration patterns and routing",
    })
    docs = retriever.retrieve("agent orchestration", top_k=2)
    assert len(docs) == 2
    assert docs[0].score >= docs[1].score
    assert {d.name for d in docs} == {"a.md", "c.md"}


def test_simple_retriever_drops_zero_overlap_docs_when_relevant_exists() -> None:
    retriever = SimpleRetriever({
        "relevant.md": "agent orchestration patterns",
        "irrelevant.md": "completely unrelated content about cats",
    })
    docs = retriever.retrieve("agent orchestration", top_k=5)
    assert {d.name for d in docs} == {"relevant.md"}
    assert all(d.score > 0.0 for d in docs)


def test_simple_retriever_keeps_top_k_when_nothing_overlaps() -> None:
    retriever = SimpleRetriever({
        "x.md": "content about cats",
        "y.md": "content about dogs",
    })
    docs = retriever.retrieve("quantum chromodynamics", top_k=2)
    assert len(docs) == 2


# ----------------- SDK agent graph (StubModel) -----------------


async def _run(script: dict, query: str):
    sup = build_supervisor(model=StubModel(script))
    ctx = RunContext()
    result = await Runner.run(sup, query, context=ctx)
    return result, ctx, map_run_result(result, query, ctx)


@pytest.mark.asyncio
async def test_research_path_hands_off_docs_to_summarizer() -> None:
    """RAG retrieves; the docs reach the summarizer via the shared RunContext."""
    q = "Summarize the architecture of this project."
    script = {
        "supervisor": [func_call("ResearchManagerAgent", input=q), message("Final summary.")],
        "research": [
            func_call("call_rag_agent", input="architecture"),
            func_call("call_summarizer_agent", input="summarize"),
            message("Research summary."),
        ],
        "rag": [func_call("simple_retriever", query="architecture", top_k=2)],
        "summarizer": [message("This project uses a 3-layer hierarchical pattern.")],
    }
    _, ctx, out = await _run(script, q)

    assert ctx.documents, "RAG must populate ctx.documents (the hand-off)"
    assert out.final_answer == "Final summary."
    assert [s.agent_name for s in out.subtasks] == ["ResearchManagerAgent"]
    nested = [tc["agent_trace"]["agent_name"] for tc in out.subtasks[0].tool_calls]
    assert nested == ["ResearchManagerAgent", "RAGAgent", "SummarizerAgent"]


@pytest.mark.asyncio
async def test_rag_agent_selects_retriever_only_no_synthesis() -> None:
    """RAG (stop_on_first_tool) selects the retriever and returns the docs, no prose."""
    q = "Find architecture docs."
    script = {
        "supervisor": [func_call("ResearchManagerAgent", input=q), message("done")],
        "research": [func_call("call_rag_agent", input="architecture"), message("ok")],
        "rag": [func_call("simple_retriever", query="architecture", top_k=2)],
        "summarizer": [message("n/a")],
    }
    _, _, out = await _run(script, q)
    rag_trace = next(
        tc["agent_trace"] for tc in out.subtasks[0].tool_calls
        if tc["agent_trace"]["agent_name"] == "RAGAgent"
    )
    assert rag_trace["selected_tools"] == ["simple_retriever"]
    assert "load_knowledge_base" in rag_trace["skipped_tools"]


@pytest.mark.asyncio
async def test_summarizer_selects_no_tools() -> None:
    q = "Summarize: the cat sat on the mat."
    script = {
        "supervisor": [func_call("ResearchManagerAgent", input=q), message("done")],
        "research": [func_call("call_summarizer_agent", input=q), message("ok")],
        "summarizer": [message("A cat sat on a mat.")],
    }
    _, _, out = await _run(script, q)
    summ = next(
        tc["agent_trace"] for tc in out.subtasks[0].tool_calls
        if tc["agent_trace"]["agent_name"] == "SummarizerAgent"
    )
    assert summ["selected_tools"] == []


class _FakeWrap:
    def __init__(self, ctx): self.context = ctx


class _FakeResult:
    """Minimal RunResult stand-in for unit-testing _retrieval_line."""
    def __init__(self, ctx, final_output): self.context_wrapper = _FakeWrap(ctx); self.final_output = final_output


def test_retrieval_line_reports_docs_even_when_agent_says_none() -> None:
    """Regression: RAG looped and emitted a 'no docs' message, but the retriever
    DID find docs. The as_tool output must reflect the retrieved docs, not the
    agent's prose (the live bug where a real summary was never produced)."""
    ctx = RunContext(documents=[{"name": "architecture_notes.md", "score": 0.5, "text": "x"}])
    line = _retrieval_line(_FakeResult(ctx, "I couldn't find any relevant documentation."))
    assert line == "Retrieved 1 document(s): architecture_notes.md."


def test_retrieval_line_reports_none_when_only_zero_score_docs() -> None:
    ctx = RunContext(documents=[{"name": "x.md", "score": 0.0, "text": "x"}])
    line = _retrieval_line(_FakeResult(ctx, "anything"))
    assert line == "No knowledge-base documents matched the query."


@pytest.mark.asyncio
async def test_retriever_keeps_relevant_docs_across_a_broader_search() -> None:
    """A later zero-overlap search must not clobber relevant docs found earlier."""
    rag = Agent(
        name="RAGAgent",
        model=StubModel({"rag": [
            func_call("simple_retriever", query="agent orchestration patterns", top_k=2),
            func_call("simple_retriever", query="zzz nothing matches here", top_k=2),
            message("done"),
        ]}),
        tools=[simple_retriever, load_knowledge_base_tool],
    )
    ctx = RunContext()
    await Runner.run(rag, "go", context=ctx)
    assert ctx.documents, "relevant docs from the first search must survive"
    assert all(d["score"] > 0.0 for d in ctx.documents)


@pytest.mark.asyncio
async def test_build_path_hands_off_code_to_reviewer() -> None:
    """Coding generates code; it reaches the reviewer via the shared RunContext."""
    q = "Build a FastAPI upload endpoint and review it."
    script = {
        "supervisor": [func_call("BuildManagerAgent", input=q), message("Code + review.")],
        "build": [
            func_call("call_coding_agent", input="fastapi upload"),
            func_call("call_review_agent", input="review the code"),
            message("Build summary."),
        ],
        "coding": [
            func_call("template_loader", template_name="fastapi_upload_endpoint"),
            message("Here is the endpoint."),
        ],
        "review": [func_call("code_review_tool", code="def x(): pass"), message("Reviewed.")],
    }
    _, ctx, out = await _run(script, q)

    assert ctx.generated_code, "CodingAgent must populate ctx.generated_code"
    assert "FastAPI" in ctx.generated_code
    assert [s.agent_name for s in out.subtasks] == ["BuildManagerAgent"]
    nested = [tc["agent_trace"]["agent_name"] for tc in out.subtasks[0].tool_calls]
    assert nested == ["BuildManagerAgent", "CodingAgent", "ReviewAgent"]
