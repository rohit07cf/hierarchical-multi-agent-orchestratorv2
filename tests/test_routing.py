"""Tests for the deterministic supervisor router."""

from __future__ import annotations

import pytest

from src.orchestrator.router import Router


@pytest.fixture
def router() -> Router:
    """A fresh Router for each test."""
    return Router()


def test_research_only_for_summarize_query(router: Router) -> None:
    decision = router.decide("Summarize the architecture of this project.")
    assert decision.use_research is True
    assert decision.use_build is False
    assert decision.managers == ["ResearchManagerAgent"]


def test_build_only_for_implementation_query(router: Router) -> None:
    decision = router.decide("Build a FastAPI endpoint for uploading documents.")
    assert decision.use_research is False
    assert decision.use_build is True
    assert decision.managers == ["BuildManagerAgent"]


def test_both_managers_for_combined_query(router: Router) -> None:
    decision = router.decide(
        "Search the knowledge base for agent orchestration patterns "
        "and generate implementation guidance."
    )
    assert decision.use_research is True
    assert decision.use_build is True
    assert decision.managers == ["ResearchManagerAgent", "BuildManagerAgent"]


def test_unknown_query_falls_back_to_research(router: Router) -> None:
    decision = router.decide("hello there friend")
    assert decision.use_research is True
    assert decision.use_build is False
    assert "default" in decision.reasoning.lower()


def test_review_keyword_triggers_build(router: Router) -> None:
    decision = router.decide("Review this code for production concerns.")
    assert decision.use_build is True
