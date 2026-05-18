"""Template loader — returns named code/skeleton templates.

Templates are deterministic primitives the `CodingAgent`'s LLM can
choose to load when it wants a starting scaffold rather than generating
code from scratch.
"""

from __future__ import annotations

_TEMPLATES: dict[str, str] = {
    "fastapi_upload_endpoint": '''from fastapi import FastAPI, UploadFile, HTTPException

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
''',
    "redis_memory_tool": '''from typing import Any
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
''',
    "python_function_skeleton": '''def TODO_function_name(arg: str) -> str:
    """Replace with a concise docstring describing what this returns."""
    raise NotImplementedError("Implement TODO_function_name")
''',
}


def list_templates() -> list[str]:
    """Return all available template names."""
    return sorted(_TEMPLATES.keys())


def load_template(template_name: str) -> dict:
    """Return the template body keyed by `template_name`.

    Unknown names return an empty body with `found=False` instead of
    raising — keeps the tool friendly for an LLM that hallucinates a
    name.
    """
    body = _TEMPLATES.get(template_name, "")
    return {
        "template_name": template_name,
        "found": bool(body),
        "body": body,
        "available": list_templates(),
    }
