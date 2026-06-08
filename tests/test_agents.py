"""Tests for individual reasoning agents and their underlying tools."""

from __future__ import annotations

import pytest

from src.agents.build_manager import BuildManagerAgent
from src.agents.coding_agent import CodingAgent
from src.agents.rag_agent import RAGAgent
from src.agents.research_manager import ResearchManagerAgent
from src.agents.review_agent import ReviewAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models.requests import AgentRequest
from src.models.trace import AgentReasoning, LLMMode, ToolDecision
from src.tools.code_review_tool import review_code
from src.tools.security_review_tool import scan_security
from src.tools.simple_retriever import SimpleRetriever
from src.tools.template_loader import load_template
from src.tools.test_gap_tool import scan_test_gaps


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
    """Irrelevant (score-0) docs must not pad the result when a match exists."""
    retriever = SimpleRetriever({
        "relevant.md": "agent orchestration patterns",
        "irrelevant.md": "completely unrelated content about cats",
    })
    docs = retriever.retrieve("agent orchestration", top_k=5)
    assert {d.name for d in docs} == {"relevant.md"}
    assert all(d.score > 0.0 for d in docs)


def test_simple_retriever_keeps_top_k_when_nothing_overlaps() -> None:
    """All-zero overlap → keep the top-k fallback so the chain has input."""
    retriever = SimpleRetriever({
        "x.md": "content about cats",
        "y.md": "content about dogs",
    })
    docs = retriever.retrieve("quantum chromodynamics", top_k=2)
    assert len(docs) == 2  # fallback: nothing matched, return top-k anyway


class _SpyLLM:
    """Minimal LLMClient stand-in that counts reason()/synthesize() calls."""

    def __init__(self) -> None:
        self.mode = LLMMode.REAL  # force the real path (mock would skip calls)
        self.reason_calls = 0
        self.synth_calls = 0

    @property
    def enabled(self) -> bool:
        return True

    async def reason(self, *, agent_name, system_prompt, user_prompt,
                     available_tools, mock_policy=None) -> AgentReasoning:
        self.reason_calls += 1
        if available_tools:
            t = available_tools[0]
            return AgentReasoning(
                reasoning="spy",
                selected_tools=[ToolDecision(
                    tool_name=t.name, rationale="spy",
                    arguments={"query": "agent orchestration"},
                )],
                skipped_tools=[],
            )
        return AgentReasoning(reasoning="spy", selected_tools=[], skipped_tools=[])

    async def synthesize(self, *, agent_name, system_prompt, user_prompt,
                         tool_results, mock_policy=None) -> str:
        self.synth_calls += 1
        return "SPY-SYNTH"


@pytest.mark.asyncio
async def test_rag_agent_does_not_call_llm_synthesize() -> None:
    """Fix C: RAGAgent is retrieval-only — no LLM synthesis call."""
    agent = RAGAgent()
    spy = _SpyLLM()
    agent.llm = spy  # type: ignore[assignment]
    response = await agent.handle(AgentRequest(query="agent orchestration patterns"))
    assert spy.synth_calls == 0, "RAGAgent must not invoke the LLM synthesizer"
    assert spy.reason_calls == 1  # it still reasons to pick the retriever
    # Content is the deterministic retrieval summary, not LLM prose.
    assert "Retrieved" in response.content or "No knowledge-base" in response.content
    assert response.data.get("documents") is not None


@pytest.mark.asyncio
async def test_toolless_agent_skips_llm_reason() -> None:
    """Fix B: a tool-less agent (summarizer) skips the reason() LLM call."""
    agent = SummarizerAgent()
    spy = _SpyLLM()
    agent.llm = spy  # type: ignore[assignment]
    response = await agent.handle(AgentRequest(query="Summarize this text please."))
    assert spy.reason_calls == 0, "tool-less agent must not invoke the LLM reasoner"
    assert spy.synth_calls == 1  # it still synthesizes the summary
    assert response.content == "SPY-SYNTH"


# ----------------- Reasoning agent traces -----------------


@pytest.mark.asyncio
async def test_rag_agent_emits_trace_with_retriever_selected() -> None:
    agent = RAGAgent()
    response = await agent.handle(
        AgentRequest(query="agent orchestration patterns")
    )
    assert response.success
    assert response.trace is not None
    trace = response.trace
    assert trace["agent_name"] == "RAGAgent"
    assert trace["llm_mode"] == LLMMode.MOCK.value
    assert "simple_retriever" in trace["selected_tools"]
    assert "load_knowledge_base" in trace["skipped_tools"]
    assert response.data.get("documents")


@pytest.mark.asyncio
async def test_summarizer_calls_no_tools_for_pasted_text() -> None:
    """Pasted text → SummarizerAgent must pick zero tools (the LLM synthesizes)."""
    agent = SummarizerAgent()
    response = await agent.handle(
        AgentRequest(
            query="The meaning of life is to live and create meaning together.",
            context={},  # no retrieved docs
        )
    )
    trace = response.trace
    assert trace is not None
    assert trace["selected_tools"] == []
    assert "[mock-llm]" in response.content


@pytest.mark.asyncio
async def test_coding_agent_picks_template_for_fastapi_request() -> None:
    agent = CodingAgent()
    response = await agent.handle(
        AgentRequest(query="Build a FastAPI endpoint for uploading documents.")
    )
    trace = response.trace
    assert trace is not None
    assert "template_loader" in trace["selected_tools"]
    assert response.data["language"] == "python"
    assert "FastAPI" in response.data["code"]


@pytest.mark.asyncio
async def test_review_agent_invokes_all_three_review_tools() -> None:
    agent = ReviewAgent()
    response = await agent.handle(
        AgentRequest(
            query="Review this code",
            context={"code": 'def f():\n    API_KEY = "abc-secret-1234"\n    return 1\n'},
        )
    )
    trace = response.trace
    assert trace is not None
    assert set(trace["selected_tools"]) == {
        "code_review_tool",
        "security_review_tool",
        "test_gap_tool",
    }
    review = response.data["review"]
    assert any(
        f["category"] == "security" for f in review["findings"]
    ), f"Expected a security finding, got: {review['findings']}"


@pytest.mark.asyncio
async def test_review_agent_skips_tools_when_no_code() -> None:
    agent = ReviewAgent()
    response = await agent.handle(
        AgentRequest(query="Review this code", context={"code": ""})
    )
    trace = response.trace
    assert trace is not None
    assert trace["selected_tools"] == []
    assert set(trace["skipped_tools"]) == {
        "code_review_tool",
        "security_review_tool",
        "test_gap_tool",
    }


# ----------------- Manager-level routing decisions -----------------


@pytest.mark.asyncio
async def test_research_manager_skips_rag_for_philosophical_text() -> None:
    """The 'meaning of life' case from the spec: SummarizerAgent only."""
    manager = ResearchManagerAgent()
    response = await manager.handle(
        AgentRequest(
            query=(
                "The meaning of life is to live, to understand, and to "
                "create something meaningful to share with others."
            )
        )
    )
    trace = response.trace
    assert trace is not None
    assert "call_summarizer_agent" in trace["selected_tools"]
    assert "call_rag_agent" not in trace["selected_tools"], (
        "Philosophical pasted text must not trigger the RAG worker"
    )


@pytest.mark.asyncio
async def test_research_manager_uses_rag_for_lookup_intent() -> None:
    manager = ResearchManagerAgent()
    response = await manager.handle(
        AgentRequest(query="Search the knowledge base for agent orchestration patterns.")
    )
    trace = response.trace
    assert trace is not None
    assert set(trace["selected_tools"]) == {
        "call_rag_agent",
        "call_summarizer_agent",
    }


@pytest.mark.asyncio
async def test_build_manager_runs_review_as_default_quality_gate() -> None:
    manager = BuildManagerAgent()
    response = await manager.handle(
        AgentRequest(query="Build a small FastAPI endpoint.")
    )
    trace = response.trace
    assert trace is not None
    assert set(trace["selected_tools"]) == {
        "call_coding_agent",
        "call_review_agent",
    }


@pytest.mark.asyncio
async def test_build_manager_skips_review_when_user_opts_out() -> None:
    manager = BuildManagerAgent()
    response = await manager.handle(
        AgentRequest(query="Build a small FastAPI endpoint, no review please.")
    )
    trace = response.trace
    assert trace is not None
    assert "call_coding_agent" in trace["selected_tools"]
    assert "call_review_agent" not in trace["selected_tools"]
    assert "call_review_agent" in trace["skipped_tools"]
