"""Lightweight LLM client used by all agents.

The project must run without an API key. This wrapper detects whether
`OPENAI_API_KEY` is available and either calls the OpenAI Chat Completions
API or returns a deterministic mock string. Agents never call OpenAI
directly so swapping in another provider — or the offline mock — is a
single-file change.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    """Thin wrapper that returns a string completion for a prompt."""

    model: str = "gpt-4.1-nano"
    enabled: bool = False

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return a completion for `prompt`, or a deterministic mock if disabled.

        Args:
            prompt: User-side prompt.
            system: Optional system message.

        Returns:
            String response. Never raises — falls back to a mock on any error.
        """
        if not self.enabled:
            return _mock_completion(prompt, system)
        try:
            return await self._call_openai(prompt, system)
        except Exception as e:
            logger.warning("LLM call failed, falling back to mock: %s", e)
            return _mock_completion(prompt, system)

    async def _call_openai(self, prompt: str, system: str | None) -> str:
        """Invoke the OpenAI Chat Completions API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


def get_llm_client(model: str = "gpt-4.1-nano") -> LLMClient:
    """Return an LLMClient, enabling real calls only if an API key is set."""
    enabled = bool(os.environ.get("OPENAI_API_KEY"))
    return LLMClient(model=model, enabled=enabled)


def _mock_completion(prompt: str, system: str | None) -> str:
    """Return a deterministic, useful-looking completion when no key is set.

    The mock is intentionally short and clearly labeled so demos run
    offline without giving the false impression of a real LLM response.
    """
    snippet = prompt.strip().splitlines()[0][:120] if prompt.strip() else ""
    return (
        "[mock-llm] "
        f"{snippet}"
        " — (offline mock; set OPENAI_API_KEY for real completions)"
    )
