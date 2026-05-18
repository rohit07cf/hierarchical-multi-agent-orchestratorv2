"""Deterministic, intent-based router used by the supervisor.

The router is conservative on purpose: BuildManagerAgent is invoked
*only* when the query carries an explicit code/implementation intent.
Reflective, philosophical, summarization, or rewriting requests route
to ResearchManagerAgent alone — even if they happen to contain generic
verbs like "create", "generate", or "review".

Two signal classes drive the build decision:

1. **Strong tokens** — substrings that almost never appear outside of a
   code/implementation request (`fastapi`, `redis`, `endpoint`,
   `implement`, `snippet`, `debug`, `refactor`).
2. **Explicit phrases** — verb+object compounds that pin down intent
   ("build a", "write a function", "review this code", "create code",
   etc.). Bare verbs like "create" or "review" do **not** match.

If neither signal fires the router defaults to research-only, which is
the safe fallback for arbitrary text the user pastes in.
"""

from __future__ import annotations

from dataclasses import dataclass

# Research signals — kept broad because research is the safe default.
_RESEARCH_KEYWORDS = (
    "summarize", "summary", "summarise", "explain", "describe",
    "rewrite", "paraphrase", "improve",
    "search", "lookup", "knowledge", "context", "architecture",
    "pattern", "patterns", "documentation",
    "meaning of", "purpose of", "perspective", "reflect",
)

_RESEARCH_PHRASES = (
    "what do you think", "make this better", "make this paragraph",
    "make this clearer", "what is", "explain this", "rewrite this",
)

# Strong unambiguous build signals — these substrings rarely appear in
# non-code prose, so a single hit is enough to invoke BuildManager.
_BUILD_TOKENS = (
    "fastapi", "redis", "endpoint", "snippet", "debug",
    "refactor", "implement",  # also catches "implementation", "implements"
)

# Explicit verb+object phrases that signal an unambiguous code intent.
# Bare verbs (`create`, `generate`, `review`, `build` alone) are
# intentionally NOT here — they trigger too many false positives on
# philosophical or general-purpose text.
_BUILD_PHRASES = (
    "build a", "build an", "build the", "build me", "build this",
    "create a function", "create an api", "create a class", "create code",
    "write a function", "write code", "write tests", "write a test",
    "write unit test",
    "generate code", "generate a function", "generate an api",
    "generate fastapi", "generate a fastapi", "generate api",
    "code review", "review code", "review the code", "review this code",
    "review the solution", "code snippet", "code generation",
    "deployment file", "modify project", "modify the project",
    "fastapi endpoint",
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
    """Decide which managers should handle a query based on intent signals."""

    def decide(self, query: str) -> RoutingDecision:
        """Return the routing decision for `query`.

        Build is opt-in (explicit code intent required); research is the
        default fallback so any free-form text still produces a
        meaningful response.
        """
        q = query.lower()

        research_hits = [k for k in _RESEARCH_KEYWORDS if k in q]
        research_hits += [p for p in _RESEARCH_PHRASES if p in q]
        use_research = bool(research_hits)

        build_token_hits = [t for t in _BUILD_TOKENS if t in q]
        build_phrase_hits = [p for p in _BUILD_PHRASES if p in q]
        use_build = bool(build_token_hits or build_phrase_hits)

        if not use_research and not use_build:
            return RoutingDecision(
                use_research=True,
                use_build=False,
                reasoning=(
                    "No explicit code intent or research keywords detected; "
                    "defaulting to research-only routing."
                ),
            )

        parts: list[str] = []
        if use_research:
            sample = ", ".join(sorted(set(research_hits))[:3])
            parts.append(f"research signal ({sample})")
        if use_build:
            sample = ", ".join(sorted(set(build_token_hits + build_phrase_hits))[:3])
            parts.append(f"explicit code intent ({sample})")
        return RoutingDecision(
            use_research=use_research,
            use_build=use_build,
            reasoning="; ".join(parts),
        )
