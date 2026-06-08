"""agent_defs package — the legacy-shaped supervisor bridge.

The refactored system lives under `src/`. This package keeps the
`SupervisorAgent` name that the Streamlit UI imports, delegating the real
work to the new hierarchy in `src/`.
"""

from agent_defs.supervisor import SupervisorAgent

__all__ = ["SupervisorAgent"]
