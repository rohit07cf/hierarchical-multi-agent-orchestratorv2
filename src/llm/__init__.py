"""LLM client abstraction (real OpenAI or deterministic mock)."""

from src.llm.client import LLMClient, get_llm_client

__all__ = ["LLMClient", "get_llm_client"]
