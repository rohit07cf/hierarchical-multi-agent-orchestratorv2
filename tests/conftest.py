"""Shared pytest fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` and the project root importable from any test.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
