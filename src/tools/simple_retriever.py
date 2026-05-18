"""Lightweight keyword-overlap retriever for the local knowledge base.

This is intentionally not a vector store. The project is meant to be
explainable in interviews and runnable offline; a deterministic
token-overlap score serves the same architectural purpose without
pulling in an embedding service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "is", "are", "be", "this", "that", "it", "as", "by", "from", "at",
    "i", "you", "we", "they", "what", "how", "why", "when", "do", "does",
})


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in _STOPWORDS]


@dataclass
class RetrievedDoc:
    """A retrieved document with a normalized relevance score in [0, 1]."""

    name: str
    text: str
    score: float


class SimpleRetriever:
    """Keyword-overlap retriever over an in-memory document map."""

    def __init__(self, documents: dict[str, str]) -> None:
        self._documents = documents
        self._token_index: dict[str, set[str]] = {
            name: set(_tokenize(text)) for name, text in documents.items()
        }

    @property
    def is_empty(self) -> bool:
        """Whether the retriever has any documents loaded."""
        return not self._documents

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        """Return the top `top_k` documents ranked by token overlap with `query`.

        The score is `|query ∩ doc| / |query|`, clamped to `[0, 1]`. When no
        documents share any tokens with the query, the highest-scoring docs
        are still returned with score 0 so the chain has something to
        summarize.
        """
        query_tokens = set(_tokenize(query))
        if not self._documents:
            return []

        scored: list[RetrievedDoc] = []
        denom = max(len(query_tokens), 1)
        for name, tokens in self._token_index.items():
            overlap = len(query_tokens & tokens)
            score = overlap / denom
            scored.append(RetrievedDoc(name=name, text=self._documents[name], score=score))

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]
