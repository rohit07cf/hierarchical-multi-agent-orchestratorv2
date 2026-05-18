"""Legacy agent_defs package — preserved as a thin import surface.

The refactored system lives under `src/`. This module keeps the old
public names so any external caller that imported them continues to
work, while degrading gracefully if the optional `openai-agents` SDK
is not installed (the new system does not require it).
"""

from agent_defs.supervisor import SupervisorAgent

__all__ = ["SupervisorAgent"]

# Legacy worker agents are only available when the `openai-agents` SDK is
# installed — the new architecture does not need them. Expose them when
# importable so backward-compatible callers still resolve the symbols.
try:
    from agent_defs.base_agent import BaseAgent  # noqa: F401
    from agent_defs.classifier_agent import ClassifierAgentDef  # noqa: F401
    from agent_defs.echo_agent import EchoAgentDef  # noqa: F401
    from agent_defs.math_agent import MathAgentDef  # noqa: F401
    from agent_defs.simple_agent import SimpleAgentDef  # noqa: F401

    __all__ += [
        "BaseAgent",
        "ClassifierAgentDef",
        "EchoAgentDef",
        "MathAgentDef",
        "SimpleAgentDef",
    ]
except ImportError:
    # openai-agents SDK not installed — only the new architecture is
    # available. This is the expected path for the offline demo.
    pass
