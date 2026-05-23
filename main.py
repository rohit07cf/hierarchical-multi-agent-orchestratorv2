"""Entry point for the Hierarchical Multi-Agent Orchestrator.

Run with: streamlit run main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.observability import init_observability
from ui.streamlit_app import run_app

# Initialise observability once at process start. No-op (beyond structured
# logging) unless OBS_ENABLED=true, so the offline demo is unaffected; when
# enabled it installs the tracer and starts the Prometheus /metrics server.
init_observability()

if __name__ == "__main__":
    run_app()
else:
    # When run via `streamlit run main.py`, __name__ is "__main__" for the
    # script but Streamlit re-executes it. This branch ensures the app
    # runs regardless.
    run_app()
