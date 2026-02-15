"""Entry point for the Hierarchical Multi-Agent Orchestrator.

Run with: streamlit run main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.streamlit_app import run_app

if __name__ == "__main__":
    run_app()
else:
    # When run via `streamlit run main.py`, __name__ is "__main__" for the
    # script but Streamlit re-executes it. This branch ensures the app
    # runs regardless.
    run_app()
