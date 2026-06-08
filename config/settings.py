"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files.

    All configuration is centralized here and accessed via get_settings().
    """

    # Anthropic (Claude)
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-opus-4-8", description="Default Claude model"
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

    # Observability (see src/observability/config.py for the runtime resolver).
    # These mirror the OBS_* env vars so settings can be inspected centrally;
    # the observability package reads the env directly at init time.
    obs_enabled: bool = Field(default=False, description="Enable tracing + metrics export")
    obs_trace_exporter: str = Field(default="console", description="otlp | console | none")
    obs_otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP gRPC endpoint")
    obs_metrics_port: int = Field(default=9108, description="Prometheus /metrics port")
    obs_log_json: bool = Field(default=True, description="Emit JSON structured logs")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()
