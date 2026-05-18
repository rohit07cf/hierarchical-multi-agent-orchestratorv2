"""Find related files in the local project tree.

Useful for the `CodingAgent` when it wants to ground a code generation
request in the existing codebase (e.g. "look at how we structure other
agents before generating a new one"). Returns small previews — never
the full file — to keep the LLM prompt bounded.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXTENSIONS = (".py", ".md")
_MAX_PREVIEW_CHARS = 400
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".hitl_states"}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def find_related_files(query: str, max_files: int = 3) -> dict:
    """Return up to `max_files` project files whose path/contents match `query`.

    The match is a simple substring scan over filenames *and* the first
    few KB of each file; scoring counts hits in either location.
    """
    query_tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(query)}
    if not query_tokens:
        return {"query": query, "matches": []}

    scored: list[tuple[int, Path, str]] = []
    for path in _iter_project_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            continue
        haystack = (str(path.relative_to(_PROJECT_ROOT)) + "\n" + text).lower()
        score = sum(1 for t in query_tokens if t in haystack)
        if score > 0:
            scored.append((score, path, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [
        {
            "path": str(path.relative_to(_PROJECT_ROOT)),
            "score": score,
            "preview": text[:_MAX_PREVIEW_CHARS],
        }
        for score, path, text in scored[:max_files]
    ]
    return {"query": query, "matches": matches}


def _iter_project_files():
    """Yield project files matching the default extensions, skipping noise dirs."""
    for path in _PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix not in _DEFAULT_EXTENSIONS:
            continue
        yield path
