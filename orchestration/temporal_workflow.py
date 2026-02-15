"""Temporal workflow definitions for durable agent orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

logger = logging.getLogger(__name__)


@dataclass
class WorkflowParams:
    """Parameters for the agent orchestration workflow."""

    user_input: str
    model: str = "gpt-4.1-nano"
    use_manual_orchestration: bool = False
    timeout_seconds: int = 300


@dataclass
class WorkflowResult:
    """Result from the agent orchestration workflow."""

    final_answer: str
    subtasks: list[dict[str, Any]]
    success: bool
    error: str | None = None


@activity.defn
async def execute_agent_orchestration_activity(params: WorkflowParams) -> WorkflowResult:
    """Temporal activity that runs the supervisor agent orchestration.

    This activity encapsulates the agent execution within Temporal's
    durability framework, providing retry logic, heartbeat monitoring,
    and failure recovery.

    Args:
        params: Workflow parameters including user input and configuration.

    Returns:
        WorkflowResult with the orchestration output.
    """
    # Import here to avoid circular imports at module level
    from agent_defs.supervisor import SupervisorAgent

    logger.info("Starting agent orchestration activity for: %s", params.user_input[:100])

    try:
        supervisor = SupervisorAgent(model=params.model)

        # Send heartbeat to indicate activity is alive
        activity.heartbeat("Starting orchestration")

        if params.use_manual_orchestration:
            output = await supervisor.orchestrate_manual(params.user_input)
        else:
            output = await supervisor.orchestrate(params.user_input)

        activity.heartbeat("Orchestration complete")

        return WorkflowResult(
            final_answer=output.final_answer,
            subtasks=[s.model_dump(mode="json") for s in output.subtasks],
            success=output.all_succeeded,
        )

    except Exception as e:
        error_msg = f"Orchestration activity failed: {e}"
        logger.error(error_msg, exc_info=True)
        return WorkflowResult(
            final_answer="",
            subtasks=[],
            success=False,
            error=error_msg,
        )


@workflow.defn
class AgentOrchestrationWorkflow:
    """Temporal workflow for durable multi-agent orchestration.

    Provides workflow-level durability, retry logic, and monitoring
    for the agent orchestration pipeline. Failed activities are
    automatically retried according to the configured retry policy.
    """

    @workflow.run
    async def run(self, params: WorkflowParams) -> WorkflowResult:
        """Execute the agent orchestration workflow.

        Args:
            params: Workflow parameters including user input and model config.

        Returns:
            WorkflowResult from the orchestration activity.
        """
        workflow.logger.info("Workflow started for: %s", params.user_input[:100])

        result = await workflow.execute_activity(
            execute_agent_orchestration_activity,
            params,
            start_to_close_timeout=timedelta(seconds=params.timeout_seconds),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        workflow.logger.info("Workflow completed, success=%s", result.success)
        return result


async def start_temporal_workflow(
    user_input: str,
    model: str = "gpt-4.1-nano",
    use_manual: bool = False,
    temporal_host: str = "localhost",
    temporal_port: int = 7233,
) -> WorkflowResult:
    """Start a Temporal workflow for agent orchestration.

    Connects to the Temporal server and starts a new workflow execution
    for the given user input.

    Args:
        user_input: The user's message to process.
        model: OpenAI model to use.
        use_manual: Whether to use manual orchestration mode.
        temporal_host: Temporal server hostname.
        temporal_port: Temporal server port.

    Returns:
        WorkflowResult from the completed workflow.
    """
    from temporalio.client import Client

    client = await Client.connect(f"{temporal_host}:{temporal_port}")

    params = WorkflowParams(
        user_input=user_input,
        model=model,
        use_manual_orchestration=use_manual,
    )

    result = await client.execute_workflow(
        AgentOrchestrationWorkflow.run,
        params,
        id=f"agent-orchestration-{hash(user_input) % 100000}",
        task_queue="agent-orchestration-queue",
    )

    return result


async def run_temporal_worker(
    temporal_host: str = "localhost",
    temporal_port: int = 7233,
) -> None:
    """Start a Temporal worker that processes agent orchestration workflows.

    Args:
        temporal_host: Temporal server hostname.
        temporal_port: Temporal server port.
    """
    from temporalio.client import Client
    from temporalio.worker import Worker

    client = await Client.connect(f"{temporal_host}:{temporal_port}")

    worker = Worker(
        client,
        task_queue="agent-orchestration-queue",
        workflows=[AgentOrchestrationWorkflow],
        activities=[execute_agent_orchestration_activity],
    )

    logger.info("Temporal worker started on queue: agent-orchestration-queue")
    await worker.run()
