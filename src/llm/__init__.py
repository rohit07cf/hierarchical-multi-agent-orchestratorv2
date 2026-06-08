"""LLM client abstraction (real Anthropic/Claude or deterministic mock)."""

from src.llm.client import DEFAULT_MODEL, LLMClient, get_llm_client

__all__ = ["DEFAULT_MODEL", "LLMClient", "get_llm_client"]
