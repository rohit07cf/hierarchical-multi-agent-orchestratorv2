"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files.

    All configuration is centralized here and accessed via get_settings().
    """

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4.1-nano", description="Default OpenAI model")

    # Temporal
    temporal_host: str = Field(default="localhost", description="Temporal server host")
    temporal_port: int = Field(default=7233, description="Temporal server port")
    temporal_task_queue: str = Field(
        default="agent-orchestration-queue",
        description="Temporal task queue name",
    )

    # Application
    app_name: str = Field(
        default="Hierarchical Multi-Agent Orchestrator",
        description="Application name",
    )
    log_level: str = Field(default="INFO", description="Logging level")
    hitl_persistence_dir: str = Field(
        default=".hitl_states",
        description="Directory for HITL state persistence",
    )

    # Streamlit
    streamlit_port: int = Field(default=8501, description="Streamlit server port")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()
