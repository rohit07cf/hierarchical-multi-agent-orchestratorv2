"""Deterministic router used by the supervisor to pick manager agents.

Routing is keyword-based on purpose: it has to be testable without an
LLM, and the heuristics double as documentation for what each manager
actually does.
"""

from __future__ import annotations

from dataclasses import dataclass

_RESEARCH_KEYWORDS = (
    "summarize", "summary", "explain", "describe", "what", "how",
    "search", "lookup", "find", "knowledge", "context", "architecture",
    "pattern", "patterns", "documentation",
)

_BUILD_KEYWORDS = (
    "build", "implement", "create", "generate", "write code", "endpoint",
    "fastapi", "function", "class", "service", "snippet", "tool",
    "review", "audit", "refactor",
)


@dataclass
class RoutingDecision:
    """Which managers the supervisor should invoke, in order."""

    use_research: bool
    use_build: bool
    reasoning: str

    @property
    def managers(self) -> list[str]:
        """Return the ordered list of manager agent names to invoke."""
        managers: list[str] = []
        if self.use_research:
            managers.append("ResearchManagerAgent")
        if self.use_build:
            managers.append("BuildManagerAgent")
        return managers


class Router:
    """Decide which managers should handle a query based on keyword signals."""

    def decide(self, query: str) -> RoutingDecision:
        """Return the routing decision for `query`.

        If the query matches neither manager's keywords, default to
        research-only — it's the safer fallback because the
        ResearchManager always succeeds (it just returns "no documents"
        when nothing is found).
        """
        q = query.lower()
        use_research = any(k in q for k in _RESEARCH_KEYWORDS)
        use_build = any(k in q for k in _BUILD_KEYWORDS)

        if not use_research and not use_build:
            return RoutingDecision(
                use_research=True,
                use_build=False,
                reasoning="No build keywords detected; defaulting to research-only routing.",
            )

        reasoning_parts: list[str] = []
        if use_research:
            reasoning_parts.append("query mentions research/lookup intent")
        if use_build:
            reasoning_parts.append("query mentions build/implementation intent")
        return RoutingDecision(
            use_research=use_research,
            use_build=use_build,
            reasoning="; ".join(reasoning_parts),
        )
