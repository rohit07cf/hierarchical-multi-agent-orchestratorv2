# FastAPI Examples

## Document upload endpoint

```python
from fastapi import FastAPI, UploadFile, HTTPException

app = FastAPI()


@app.post("/documents")
async def upload_document(file: UploadFile) -> dict:
    if file.content_type not in {"text/plain", "application/pdf"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(status_code=413, detail="File too large")
    return {"filename": file.filename, "bytes": len(contents)}
```

Production considerations:

- Validate content type and size *before* reading the file fully into memory.
- Stream large uploads to disk or object storage instead of `await file.read()`.
- Add an authentication dependency (e.g. `Depends(verify_token)`).
- Add structured logging and a request ID middleware.

## Simple Redis memory tool

```python
from typing import Any
import redis.asyncio as redis


class RedisMemory:
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        await self._client.set(key, str(value), ex=ttl_seconds)
```

Production considerations:

- Always set a TTL so the memory layer never grows unbounded.
- Wrap calls in a retry/backoff for transient network errors.
- Use a connection pool and close it on shutdown.
- Don't store secrets without encryption.
