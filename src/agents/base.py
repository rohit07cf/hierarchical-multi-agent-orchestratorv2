"""Base class for agents in the new hierarchical architecture."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.llm.client import LLMClient, get_llm_client
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Common scaffolding for every agent in the system.

    Subclasses implement `handle()` and receive a shared `LLMClient` and a
    consistent name/tools surface so the orchestrator can introspect them
    for the UI's agent-hierarchy display.
    """

    name: str = "BaseAgent"
    tools: list[str] = []

    def __init__(self, model: str = "gpt-4.1-nano", llm: LLMClient | None = None) -> None:
        self.model = model
        self.llm = llm or get_llm_client(model)

    @abstractmethod
    async def handle(self, request: AgentRequest) -> AgentResponse:
        """Process the request and return an `AgentResponse`."""

    def _log(self, message: str) -> None:
        """Emit a uniformly-formatted log line — used for orchestration traces."""
        logger.info("[%s] %s", self.name, message)
