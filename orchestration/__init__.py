"""Orchestration layer for the hierarchical multi-agent orchestrator.

Imports are guarded so the package degrades gracefully when the optional
`openai-agents` SDK or Temporal SDK are not installed — the offline
refactored demo only depends on `AgentTree` and `HITLManager`.
"""

from orchestration.agent_tree import AgentNode, AgentTree
from orchestration.hitl_manager import HITLManager

__all__ = ["AgentNode", "AgentTree", "HITLManager"]

try:
    from orchestration.streaming_handler import StreamingCallbackHandler  # noqa: F401

    __all__.append("StreamingCallbackHandler")
except ImportError:
    pass

try:
    from orchestration.temporal_workflow import AgentOrchestrationWorkflow  # noqa: F401

    __all__.append("AgentOrchestrationWorkflow")
except ImportError:
    pass
