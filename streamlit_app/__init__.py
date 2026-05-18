"""Top-level Streamlit application package.

The existing app entrypoint lives at `ui/streamlit_app.py` to preserve
the historical `streamlit run main.py` flow. This package mirrors the
new project layout requested in the refactor (`streamlit_app/{components,
pages}`) and re-exports the same UI helpers so future pages can be added
under `streamlit_app/pages/` without touching `ui/`.
"""

from ui.streamlit_app import run_app

__all__ = ["run_app"]
