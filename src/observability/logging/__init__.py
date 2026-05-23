"""Structured JSON logging correlated to traces and requests."""

from src.observability.logging.setup import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
