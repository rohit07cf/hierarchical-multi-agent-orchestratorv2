"""Deterministic boilerplate generator used by the CodingAgent's LLM.

The real code generation is done by the LLM at synthesis time; this
tool exists so the agent has *something* to call when it wants a
language-appropriate skeleton without inventing one from scratch. In
mock mode the skeleton IS the code the user sees.
"""

from __future__ import annotations

_PYTHON_SKELETON = '''from __future__ import annotations


def TODO_function_name(arg: str) -> str:
    """Summarize what this function returns in one short line."""
    raise NotImplementedError("Implement TODO_function_name")
'''

_PYTHON_FASTAPI_SKELETON = '''from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/healthz")
async def healthz() -> dict:
    """Simple health-check endpoint — replace with real handler."""
    return {"status": "ok"}
'''

_PYTHON_CLASS_SKELETON = '''from __future__ import annotations


class TODOClassName:
    """One-line docstring describing what this class models."""

    def __init__(self, arg: str) -> None:
        self.arg = arg
'''


def generate_skeleton(query: str, language: str = "python") -> dict:
    """Return a small boilerplate skeleton matched against `query`.

    Args:
        query: Natural-language description of what the user wants built.
        language: Target language. Only "python" is supported today;
            unknown languages return an empty skeleton.

    Returns:
        Dict with `language`, `body`, and `skeleton_kind` describing
        which template was selected.
    """
    if language != "python":
        return {
            "language": language,
            "skeleton_kind": "none",
            "body": "",
            "note": f"language {language!r} not supported",
        }

    q = query.lower()
    if "fastapi" in q or "endpoint" in q or "api" in q:
        return {
            "language": language,
            "skeleton_kind": "fastapi_app",
            "body": _PYTHON_FASTAPI_SKELETON,
        }
    if "class" in q:
        return {
            "language": language,
            "skeleton_kind": "python_class",
            "body": _PYTHON_CLASS_SKELETON,
        }
    return {
        "language": language,
        "skeleton_kind": "python_function",
        "body": _PYTHON_SKELETON,
    }
