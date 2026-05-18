"""Tests for individual worker agents and the tools they wrap."""

from __future__ import annotations

import pytest

from src.agents.coding_agent import CodingAgent
from src.agents.rag_agent import RAGAgent
from src.agents.review_agent import ReviewAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models.requests import AgentRequest
from src.tools.code_review_tool import review_code
from src.tools.simple_retriever import SimpleRetriever


@pytest.mark.asyncio
async def test_rag_agent_returns_documents() -> None:
    agent = RAGAgent()
    response = await agent.handle(AgentRequest(query="hierarchical orchestration patterns"))
    assert response.success
    docs = response.data["documents"]
    assert len(docs) > 0
    assert all("name" in d and "text" in d for d in docs)


@pytest.mark.asyncio
async def test_summarizer_handles_empty_documents() -> None:
    agent = SummarizerAgent()
    response = await agent.handle(AgentRequest(query="anything", context={"documents": []}))
    assert response.success
    assert "nothing to summarize" in response.content.lower()


@pytest.mark.asyncio
async def test_coding_agent_returns_python_code() -> None:
    agent = CodingAgent()
    response = await agent.handle(AgentRequest(query="Build a FastAPI endpoint"))
    assert response.data["language"] == "python"
    assert "fastapi" in response.data["code"].lower()


@pytest.mark.asyncio
async def test_review_agent_formats_findings() -> None:
    agent = ReviewAgent()
    response = await agent.handle(
        AgentRequest(query="review", context={"code": "def f(): return 1"})
    )
    assert "finding" in response.content.lower()
    review = response.data["review"]
    assert "approved" in review
    assert "findings" in review


def test_review_code_flags_hardcoded_secret() -> None:
    result = review_code('API_KEY = "abc-123"\n')
    assert result.has_blockers is True
    assert any(f.category == "security" for f in result.findings)


def test_review_code_warns_on_missing_error_handling() -> None:
    result = review_code("def f():\n    return 1\n")
    assert any(f.category == "bugs" for f in result.findings)


def test_simple_retriever_returns_top_k_sorted() -> None:
    retriever = SimpleRetriever({
        "a.md": "supervisor agent orchestration",
        "b.md": "completely unrelated content about cats",
        "c.md": "agent orchestration patterns and routing",
    })
    docs = retriever.retrieve("agent orchestration", top_k=2)
    assert len(docs) == 2
    # `a.md` and `c.md` both contain "agent" and "orchestration".
    assert docs[0].score >= docs[1].score
    assert {d.name for d in docs} == {"a.md", "c.md"}
