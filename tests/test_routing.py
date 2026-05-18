"""Tests for the deterministic supervisor router.

Covers the intent-based routing rules: BuildManagerAgent must only be
invoked when the query carries explicit code/implementation intent.
Generic verbs like "create", "review", or "generate" — and reflective
or philosophical text — must route to ResearchManagerAgent alone.
"""

from __future__ import annotations

import pytest

from src.orchestrator.router import Router


@pytest.fixture
def router() -> Router:
    """A fresh Router for each test."""
    return Router()


# --- Original cases (still expected to hold under the new rules) ---


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


# --- New cases: intent-based routing must reject false positives ---


def test_philosophical_text_routes_to_research_only(router: Router) -> None:
    """Reflective prose containing generic verbs must not trigger BuildManager.

    The text below mentions "create", "live", "understand", "purpose" —
    all words that the OLD keyword router would catch — but it is
    clearly a philosophical reflection, not a code request.
    """
    decision = router.decide(
        "The meaning of life is to live, to understand, to create "
        "something meaningful, and to share it with others."
    )
    assert decision.use_build is False, (
        f"Philosophical text must not route to BuildManager, "
        f"got reasoning: {decision.reasoning}"
    )
    assert decision.use_research is True
    assert decision.managers == ["ResearchManagerAgent"]


def test_long_pasted_article_routes_to_research_only(router: Router) -> None:
    """A long pasted article without an explicit code request must stay research-only.

    The article incidentally uses words like "build", "create", "generate",
    and "function" in the linguistic/philosophical sense — none of which
    should be misread as a coding request.
    """
    article = (
        "Modern philosophers have argued at length about how we build "
        "personal identity over time. To create meaning, one must engage "
        "deeply with the world and generate hypotheses about its function. "
        "Consciousness, memory, and perspective all play a role in how a "
        "person reviews their past experiences and reaches new understanding. "
        "There is no single class of answer that satisfies every observer; "
        "instead, each individual finds their own perspective through "
        "reflection. " * 4
    )
    decision = router.decide(article)
    assert decision.use_build is False, (
        f"Long prose without explicit code intent must not trigger "
        f"BuildManager, got reasoning: {decision.reasoning}"
    )
    assert decision.use_research is True
    assert decision.managers == ["ResearchManagerAgent"]


def test_explicit_fastapi_build_routes_to_build(router: Router) -> None:
    """The canonical explicit-build query must route to BuildManagerAgent."""
    decision = router.decide("Build a FastAPI endpoint that uploads files.")
    assert decision.use_build is True
    assert decision.managers[-1] == "BuildManagerAgent"


def test_summarize_then_create_code_routes_to_both(router: Router) -> None:
    """A combined request routes through Research → Build in order."""
    decision = router.decide(
        "Summarize the architecture and then create code for a "
        "small example service."
    )
    assert decision.use_research is True
    assert decision.use_build is True
    assert decision.managers == ["ResearchManagerAgent", "BuildManagerAgent"]


# --- Negative cases for individual generic verbs ---


@pytest.mark.parametrize(
    "query",
    [
        "Summarize this text.",
        "Make this paragraph better.",
        "Explain this idea.",
        "What do you think about this?",
        "Rewrite this in a clearer way.",
    ],
)
def test_reflective_inputs_never_trigger_build(router: Router, query: str) -> None:
    decision = router.decide(query)
    assert decision.use_build is False, (
        f"Reflective input {query!r} must not trigger BuildManager, "
        f"got reasoning: {decision.reasoning}"
    )
    assert decision.use_research is True


@pytest.mark.parametrize(
    "query",
    [
        "Please create a new approach to this problem.",
        "I want to generate fresh ideas about consciousness.",
        "Can you review the perspective in this essay?",
    ],
)
def test_bare_generic_verbs_do_not_trigger_build(
    router: Router, query: str
) -> None:
    """`create`/`generate`/`review` alone (no code object) must not be build."""
    decision = router.decide(query)
    assert decision.use_build is False, (
        f"Bare generic verb in {query!r} should not trigger BuildManager, "
        f"got reasoning: {decision.reasoning}"
    )
