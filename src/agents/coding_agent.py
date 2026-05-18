"""CodingAgent — generates Python / FastAPI implementation snippets."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse

_CODING_SYSTEM = (
    "You are a senior Python engineer. Produce a single, focused, "
    "production-quality code snippet that satisfies the request. Include "
    "type hints, error handling and brief comments only where the "
    "rationale is non-obvious."
)


class CodingAgent(BaseAgent):
    """Generate a code snippet for the request.

    Uses the LLM when available; otherwise falls back to one of a small
    set of pre-canned snippets chosen by keyword match. The deterministic
    path makes the demo runnable offline while keeping the output shape
    identical to the LLM path.
    """

    name = "CodingAgent"
    tools = ["llm_codegen"]

    async def handle(self, request: AgentRequest) -> AgentResponse:
        if self.llm.enabled:
            code = await self.llm.complete(request.query, system=_CODING_SYSTEM)
        else:
            code = _canned_snippet(request.query)

        self._log("Code snippet generated")
        return AgentResponse(
            agent_name=self.name,
            content=f"```python\n{code}\n```",
            data={"code": code, "language": "python"},
        )


def _canned_snippet(query: str) -> str:
    """Return a deterministic snippet by keyword matching the query."""
    q = query.lower()
    if "redis" in q:
        return _REDIS_SNIPPET
    if "fastapi" in q or "endpoint" in q or "upload" in q:
        return _FASTAPI_SNIPPET
    return _GENERIC_SNIPPET


_FASTAPI_SNIPPET = '''from fastapi import FastAPI, UploadFile, HTTPException

app = FastAPI()


@app.post("/documents")
async def upload_document(file: UploadFile) -> dict:
    """Upload a document; validate type and size before reading."""
    allowed = {"text/plain", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(status_code=413, detail="File too large")
    return {"filename": file.filename, "size": len(contents)}
'''

_REDIS_SNIPPET = '''from typing import Any
import redis.asyncio as redis


class RedisMemory:
    """Async Redis-backed key/value memory with TTL."""

    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(key)
        except redis.RedisError as e:
            raise RuntimeError(f"Redis get failed: {e}") from e

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        try:
            await self._client.set(key, str(value), ex=ttl_seconds)
        except redis.RedisError as e:
            raise RuntimeError(f"Redis set failed: {e}") from e
'''

_GENERIC_SNIPPET = '''def solve(query: str) -> str:
    """Placeholder implementation generated from the query."""
    return f"Implementation for: {query!r}"
'''
