"""Supervisor agent: orchestrates task decomposition and delegates to child agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import Agent, Runner, function_tool

from agent_defs.base_agent import BaseAgent
from agent_defs.simple_agent import SimpleAgentDef
from agent_defs.math_agent import MathAgentDef
from agent_defs.echo_agent import EchoAgentDef
from agent_defs.classifier_agent import ClassifierAgentDef
from models.agent_state import HITLCheckpointType
from models.supervisor_output import (
    SubtaskResult,
    SubtaskStatus,
    SupervisorOutput,
    TaskDecomposition,
    PlannedSubtask,
)
from models.tool_models import ToolResult
from orchestration.hitl_manager import HITLManager
from prompts.supervisor_prompt import get_supervisor_prompt

logger = logging.getLogger(__name__)


# Sentinel returned when orchestration pauses for HITL review
HITL_PAUSED_SENTINEL = "__HITL_PAUSED__"


@function_tool
def reasoning_step(thought: str, action_plan: str, agents_needed: str) -> str:
    """Execute a structured reasoning step for task decomposition.

    Use this tool to reason through how to decompose a user request
    into subtasks and which agents should handle each subtask.

    Args:
        thought: Your analysis of the user request.
        action_plan: The planned sequence of agent delegations.
        agents_needed: Comma-separated list of agents required.
    """
    agents_list = [a.strip() for a in agents_needed.split(",") if a.strip()]
    return ToolResult.ok(
        result={
            "thought": thought,
            "action_plan": action_plan,
            "agents_needed": agents_list,
            "reasoning_complete": True,
        },
        tool_name="reasoning_step",
    ).model_dump_json()


class SupervisorAgent(BaseAgent):
    """Supervisor agent that decomposes tasks and delegates to specialized child agents.

    Maintains a registry of child agents and orchestrates multi-step workflows
    by routing subtasks to the appropriate child agent based on the task requirements.

    Supports two HITL checkpoint types:
    - DECOMPOSITION: Pauses after task decomposition for plan review
    - TOOL_EXECUTION: Pauses before each child agent runs for tool confirmation
    """

    CHILD_AGENT_MAP: dict[str, type[BaseAgent]] = {
        "SimpleAgent": SimpleAgentDef,
        "MathAgent": MathAgentDef,
        "EchoAgent": EchoAgentDef,
        "ClassifierAgent": ClassifierAgentDef,
    }

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        super().__init__(name="Supervisor", model=model)
        self._child_agents: dict[str, BaseAgent] = {}
        self._initialize_children()

    def _initialize_children(self) -> None:
        """Instantiate all child agents."""
        for name, agent_cls in self.CHILD_AGENT_MAP.items():
            self._child_agents[name] = agent_cls(model=self.model)

    @property
    def child_agents(self) -> dict[str, BaseAgent]:
        """Access the registry of child agents."""
        return self._child_agents

    def _get_system_prompt(self) -> str:
        return get_supervisor_prompt()

    def _register_tools(self) -> list[Any]:
        return [reasoning_step]

    def _build_agent(self) -> Agent:
        """Build the supervisor agent with tools for task decomposition."""
        self._tools = self._register_tools()
        return Agent(
            name=self.name,
            instructions=self._get_system_prompt(),
            tools=self._tools,
            model=self.model,
        )

    async def orchestrate(
        self,
        user_input: str,
        hitl_manager: HITLManager | None = None,
        enable_hitl: bool = False,
    ) -> SupervisorOutput | None:
        """Run the full orchestration pipeline: decompose, delegate, aggregate.

        When enable_hitl=True, execution pauses at two checkpoints:
        1. After decomposition — user reviews the planned subtasks
        2. Before each tool execution — user confirms each agent delegation

        Returns None when paused for HITL (result will come on resume).

        Args:
            user_input: The user's message or task.
            hitl_manager: HITLManager for state persistence (required if enable_hitl=True).
            enable_hitl: Whether to pause at HITL checkpoints.

        Returns:
            SupervisorOutput with aggregated results, or None if paused for HITL.
        """
        self.reset_state()
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Supervisor starting orchestration for: %s", user_input[:100])

        try:
            # Step 1: Decompose the task into subtasks
            decomposition = await self._decompose_task(user_input)
            self._state.add_step(
                agent_name=self.name,
                action="task_decomposition",
                action_input={"user_input": user_input},
                observation=json.dumps(decomposition.model_dump(), default=str),
            )

            # HITL Checkpoint 1: Decomposition review
            if enable_hitl and hitl_manager:
                self._state.pause(
                    checkpoint_type=HITLCheckpointType.DECOMPOSITION,
                    pending_data={
                        "decomposition": decomposition.model_dump(mode="json"),
                        "subtask_index": 0,
                    },
                )
                hitl_manager.capture_state(self._state)
                logger.info("HITL: Paused for decomposition review")
                return None

            # Step 2 + 3: Execute subtasks and aggregate
            return await self._execute_and_aggregate(
                user_input, decomposition, hitl_manager, enable_hitl,
            )

        except Exception as e:
            error_msg = f"Orchestration failed: {e}"
            logger.error(error_msg, exc_info=True)
            return SupervisorOutput(
                final_answer=error_msg,
                subtasks=[
                    SubtaskResult(
                        agent_name="Supervisor",
                        subtask=user_input,
                        result=None,
                        status=SubtaskStatus.FAILED,
                        error=str(e),
                    )
                ],
            )

    async def resume_orchestration(
        self,
        hitl_manager: HITLManager,
        state_id: str,
    ) -> SupervisorOutput | None:
        """Resume a paused orchestration after HITL action.

        Picks up from where execution paused — either after decomposition
        review or after tool execution confirmation.

        Args:
            hitl_manager: The HITLManager holding the paused state.
            state_id: ID of the paused state to resume.

        Returns:
            SupervisorOutput if orchestration completes, or None if paused again.
        """
        state = hitl_manager.get_state(state_id)
        if state is None:
            logger.error("Cannot resume: state %s not found", state_id)
            return SupervisorOutput(
                final_answer=f"Error: State {state_id} not found",
                subtasks=[],
            )

        # Restore internal state
        self._state = state
        user_input = state.current_inputs.get("user_input", "")
        pending = state.pending_data

        # Check the last HITL action
        last_action = state.hitl_actions[-1] if state.hitl_actions else None
        if last_action and last_action.action.value == "CANCEL":
            return SupervisorOutput(
                final_answer="Orchestration cancelled by user.",
                subtasks=[],
            )

        # Handle revision: user may have changed the input
        if last_action and last_action.action.value == "REVISE" and last_action.input:
            user_input = last_action.input
            state.current_inputs["user_input"] = user_input

        checkpoint = state.checkpoint_type

        try:
            if checkpoint == HITLCheckpointType.DECOMPOSITION:
                # User approved/revised the decomposition — now execute subtasks
                decomposition_data = pending.get("decomposition", {})
                decomposition = TaskDecomposition.model_validate(decomposition_data)

                # If revised, re-decompose with new input
                if last_action and last_action.action.value == "REVISE":
                    decomposition = await self._decompose_task(user_input)
                    self._state.add_step(
                        agent_name=self.name,
                        action="task_re_decomposition",
                        action_input={"revised_input": user_input},
                        observation=json.dumps(decomposition.model_dump(), default=str),
                    )

                return await self._execute_and_aggregate(
                    user_input, decomposition, hitl_manager, enable_hitl=True,
                )

            elif checkpoint == HITLCheckpointType.TOOL_EXECUTION:
                # User approved a tool execution — run it and continue
                decomposition_data = pending.get("decomposition", {})
                decomposition = TaskDecomposition.model_validate(decomposition_data)
                subtask_index = pending.get("subtask_index", 0)
                completed_results = pending.get("completed_results", [])

                # Restore already-completed subtask results
                subtask_results = [
                    SubtaskResult.model_validate(r) for r in completed_results
                ]

                # Continue from where we left off
                return await self._execute_and_aggregate(
                    user_input,
                    decomposition,
                    hitl_manager,
                    enable_hitl=True,
                    start_index=subtask_index,
                    prior_results=subtask_results,
                )

            else:
                logger.error("Unknown checkpoint type: %s", checkpoint)
                return SupervisorOutput(
                    final_answer="Error: Unknown HITL checkpoint type",
                    subtasks=[],
                )

        except Exception as e:
            error_msg = f"Resume failed: {e}"
            logger.error(error_msg, exc_info=True)
            return SupervisorOutput(
                final_answer=error_msg,
                subtasks=[
                    SubtaskResult(
                        agent_name="Supervisor",
                        subtask=user_input,
                        result=None,
                        status=SubtaskStatus.FAILED,
                        error=str(e),
                    )
                ],
            )

    async def _execute_and_aggregate(
        self,
        user_input: str,
        decomposition: TaskDecomposition,
        hitl_manager: HITLManager | None = None,
        enable_hitl: bool = False,
        start_index: int = 0,
        prior_results: list[SubtaskResult] | None = None,
    ) -> SupervisorOutput | None:
        """Execute subtasks and aggregate, with optional HITL pauses before each tool.

        Args:
            user_input: The user's original request.
            decomposition: The task decomposition plan.
            hitl_manager: HITLManager for HITL state persistence.
            enable_hitl: Whether to pause before each tool execution.
            start_index: Index to start execution from (for resume).
            prior_results: Already-completed results from previous run (for resume).

        Returns:
            SupervisorOutput if all complete, or None if paused.
        """
        subtask_results = list(prior_results) if prior_results else []

        for i, planned in enumerate(decomposition.subtasks[start_index:], start=start_index):
            # HITL Checkpoint 2: Tool execution confirmation
            if enable_hitl and hitl_manager:
                self._state.pause(
                    checkpoint_type=HITLCheckpointType.TOOL_EXECUTION,
                    pending_data={
                        "decomposition": decomposition.model_dump(mode="json"),
                        "subtask_index": i,
                        "completed_results": [r.model_dump(mode="json") for r in subtask_results],
                        "pending_tool": {
                            "agent_name": planned.agent_name,
                            "description": planned.description,
                            "tools_needed": planned.tools_needed,
                        },
                    },
                )
                self._state.tool = planned.agent_name
                self._state.tool_path = f"Supervisor.{planned.agent_name}"
                hitl_manager.capture_state(self._state)
                logger.info(
                    "HITL: Paused for tool confirmation — %s: %s",
                    planned.agent_name,
                    planned.description[:60],
                )
                return None

            # Execute the subtask
            result = await self._execute_subtask(planned)
            subtask_results.append(result)
            self._state.add_step(
                agent_name=planned.agent_name,
                action=f"subtask_complete:{planned.description[:60]}",
                observation=str(result.result) if result.result else str(result.error),
            )

        # All subtasks complete — aggregate
        final_answer = self._aggregate_results(user_input, subtask_results)

        self._state.add_step(
            agent_name=self.name,
            action="orchestration_complete",
            observation=final_answer,
        )

        return SupervisorOutput(
            final_answer=final_answer,
            subtasks=subtask_results,
            decomposition=decomposition,
        )

    async def orchestrate_manual(self, user_input: str) -> SupervisorOutput:
        """Run orchestration with explicit manual decomposition and delegation.

        Unlike orchestrate() which supports HITL checkpoints, this method
        runs to completion without pausing. Useful for non-interactive contexts.

        Args:
            user_input: The user's message or task.

        Returns:
            SupervisorOutput with results from each individually-executed subtask.
        """
        self.reset_state()
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Supervisor starting manual orchestration for: %s", user_input[:100])

        # Step 1: Use supervisor to decompose the task
        decomposition = await self._decompose_task(user_input)
        self._state.add_step(
            agent_name=self.name,
            action="task_decomposition",
            action_input={"user_input": user_input},
            observation=json.dumps(decomposition.model_dump(), default=str),
        )

        # Step 2: Execute each subtask via the appropriate child agent
        subtask_results: list[SubtaskResult] = []
        for planned in decomposition.subtasks:
            result = await self._execute_subtask(planned)
            subtask_results.append(result)

        # Step 3: Aggregate results
        final_answer = self._aggregate_results(user_input, subtask_results)

        self._state.add_step(
            agent_name=self.name,
            action="aggregation",
            observation=final_answer,
        )

        return SupervisorOutput(
            final_answer=final_answer,
            subtasks=subtask_results,
            decomposition=decomposition,
        )

    async def _decompose_task(self, user_input: str) -> TaskDecomposition:
        """Use the supervisor LLM to decompose a task into subtasks."""
        decomposition_prompt = f"""Analyze this user request and decompose it into subtasks.
For each subtask, specify which agent should handle it.

User request: {user_input}

Available agents:
- SimpleAgent: [add_numbers, echo_text]
- MathAgent: [add_numbers, subtract_numbers, multiply_numbers]
- EchoAgent: [echo_text, reverse_text]
- ClassifierAgent: [classify_intent, detect_sentiment]

Respond in this JSON format:
{{
    "reasoning": "your analysis here",
    "subtasks": [
        {{"agent_name": "AgentName", "description": "what to do", "tools_needed": ["tool1"]}}
    ]
}}"""

        try:
            result = await Runner.run(
                Agent(
                    name="TaskDecomposer",
                    instructions="You decompose tasks into subtasks. Always respond with valid JSON only.",
                    model=self.model,
                ),
                input=decomposition_prompt,
            )
            raw = str(result.final_output)

            # Parse JSON from the response
            try:
                # Try to extract JSON from potential markdown code blocks
                if "```" in raw:
                    json_str = raw.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    json_str = json_str.strip()
                else:
                    json_str = raw.strip()

                parsed = json.loads(json_str)
                planned_subtasks = [
                    PlannedSubtask(
                        agent_name=s.get("agent_name", "SimpleAgent"),
                        description=s.get("description", ""),
                        tools_needed=s.get("tools_needed", []),
                    )
                    for s in parsed.get("subtasks", [])
                ]
                return TaskDecomposition(
                    original_request=user_input,
                    reasoning=parsed.get("reasoning", ""),
                    subtasks=planned_subtasks,
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning("Could not parse decomposition JSON, creating fallback")
                return self._fallback_decomposition(user_input)

        except Exception as e:
            logger.error("Task decomposition failed: %s", e)
            return self._fallback_decomposition(user_input)

    def _fallback_decomposition(self, user_input: str) -> TaskDecomposition:
        """Create a simple fallback decomposition when LLM parsing fails."""
        return TaskDecomposition(
            original_request=user_input,
            reasoning="Fallback: routing entire request to SimpleAgent",
            subtasks=[
                PlannedSubtask(
                    agent_name="SimpleAgent",
                    description=user_input,
                    tools_needed=[],
                )
            ],
        )

    async def _execute_subtask(self, planned: PlannedSubtask) -> SubtaskResult:
        """Execute a single planned subtask using the designated child agent."""
        agent_name = planned.agent_name
        child = self._child_agents.get(agent_name)

        if child is None:
            return SubtaskResult(
                agent_name=agent_name,
                subtask=planned.description,
                result=None,
                status=SubtaskStatus.FAILED,
                error=f"Unknown agent: {agent_name}",
            )

        self._state.tool_path = f"Supervisor.{agent_name}"
        logger.info("Delegating to %s: %s", agent_name, planned.description[:80])

        try:
            output = await child.run(planned.description)
            return SubtaskResult(
                agent_name=agent_name,
                subtask=planned.description,
                result=output,
                status=SubtaskStatus.COMPLETED,
            )
        except Exception as e:
            return SubtaskResult(
                agent_name=agent_name,
                subtask=planned.description,
                result=None,
                status=SubtaskStatus.FAILED,
                error=str(e),
            )

    def _aggregate_results(
        self, user_input: str, results: list[SubtaskResult]
    ) -> str:
        """Combine subtask results into a coherent final answer."""
        parts = []
        for r in results:
            if r.is_success:
                parts.append(f"[{r.agent_name}] {r.subtask}: {r.result}")
            else:
                parts.append(f"[{r.agent_name}] {r.subtask}: FAILED - {r.error}")

        if not parts:
            return "No subtasks were executed."

        return "\n".join(parts)
