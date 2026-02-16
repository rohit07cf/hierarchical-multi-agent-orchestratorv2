"""Supervisor agent: orchestrates task decomposition and delegates to child agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import Agent, Runner, function_tool
from agents.items import (
    HandoffOutputItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
)

from agent_defs.base_agent import BaseAgent
from agent_defs.simple_agent import SimpleAgentDef
from agent_defs.math_agent import MathAgentDef
from agent_defs.echo_agent import EchoAgentDef
from agent_defs.classifier_agent import ClassifierAgentDef
from models.agent_state import AgentState
from models.supervisor_output import (
    SubtaskResult,
    SubtaskStatus,
    SupervisorOutput,
    TaskDecomposition,
    PlannedSubtask,
)
from models.tool_models import ToolResult
from prompts.supervisor_prompt import get_supervisor_prompt

logger = logging.getLogger(__name__)


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
        """Build the supervisor agent with bidirectional handoffs to child agents.

        Creates the supervisor first, then rebuilds each child agent with a
        handoff back to the supervisor so control returns after each subtask.
        """
        self._tools = self._register_tools()

        # Build the supervisor agent first (without handoffs yet)
        supervisor_agent = Agent(
            name=self.name,
            instructions=self._get_system_prompt(),
            tools=self._tools,
            handoffs=[],  # will be set after child agents are wired
            model=self.model,
        )

        # Rebuild each child agent with a handoff back to the supervisor
        child_sdk_agents = []
        for child in self._child_agents.values():
            child_agent = Agent(
                name=child.name,
                instructions=child._get_system_prompt(),
                tools=child._register_tools(),
                handoffs=[supervisor_agent],
                model=child.model,
            )
            # Update the child's cached agent reference
            child._agent = child_agent
            child_sdk_agents.append(child_agent)

        # Now set the supervisor's handoffs to point to the rebuilt child agents
        supervisor_agent.handoffs = child_sdk_agents

        return supervisor_agent

    async def orchestrate(self, user_input: str) -> SupervisorOutput:
        """Run the full orchestration pipeline: decompose, delegate, aggregate.

        This is the primary entry point for processing user requests through
        the supervisor hierarchy. It uses the OpenAI Agents SDK's built-in
        handoff mechanism for delegation.

        Args:
            user_input: The user's message or task.

        Returns:
            SupervisorOutput with aggregated results from all subtask executions.
        """
        self.reset_state()
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Supervisor starting orchestration for: %s", user_input[:100])

        subtask_results: list[SubtaskResult] = []

        try:
            # Run the supervisor agent which will use handoffs to delegate
            result = await Runner.run(self.agent, input=user_input, max_turns=25)
            final_output = str(result.final_output)

            # Extract per-agent subtask results from the run trace
            subtask_results, planned_subtasks = self._extract_subtask_results(result)

            # Track which agents were involved via the run result
            self._state.add_step(
                agent_name=self.name,
                action="orchestration_complete",
                observation=final_output,
            )

            return SupervisorOutput(
                final_answer=final_output,
                subtasks=subtask_results,
                decomposition=TaskDecomposition(
                    original_request=user_input,
                    reasoning="Delegated via agent handoffs",
                    subtasks=planned_subtasks,
                ),
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

    def _extract_subtask_results(
        self, result: Any
    ) -> tuple[list[SubtaskResult], list[PlannedSubtask]]:
        """Extract per-agent subtask results from the SDK run trace.

        Walks through result.new_items to reconstruct which child agents
        were invoked, what tools they called, and what they produced.

        Returns:
            Tuple of (subtask_results, planned_subtasks).
        """
        # Track per-agent data: agent_name -> {tool_calls, result_text}
        agent_data: dict[str, dict[str, Any]] = {}
        current_child: str | None = None

        for item in result.new_items:
            agent_name = item.agent.name if item.agent else "Unknown"

            # Track handoffs to child agents
            if isinstance(item, HandoffOutputItem):
                target_name = item.target_agent.name
                if target_name != self.name:
                    current_child = target_name
                    if target_name not in agent_data:
                        agent_data[target_name] = {"tool_calls": [], "result": None}
                else:
                    current_child = None

            # Track tool calls made by child agents (not the Supervisor itself)
            elif isinstance(item, ToolCallItem) and agent_name != self.name:
                if agent_name not in agent_data:
                    agent_data[agent_name] = {"tool_calls": [], "result": None}
                tool_info = {"tool": getattr(item.raw_item, "name", "unknown")}
                if hasattr(item.raw_item, "arguments"):
                    tool_info["arguments"] = item.raw_item.arguments
                agent_data[agent_name]["tool_calls"].append(tool_info)

            # Track tool call outputs from child agents
            elif isinstance(item, ToolCallOutputItem) and agent_name != self.name:
                if agent_name not in agent_data:
                    agent_data[agent_name] = {"tool_calls": [], "result": None}
                agent_data[agent_name]["result"] = str(item.output)

            # Track message outputs from child agents
            elif isinstance(item, MessageOutputItem) and agent_name != self.name:
                if agent_name not in agent_data:
                    agent_data[agent_name] = {"tool_calls": [], "result": None}
                text_parts = []
                for content in item.raw_item.content:
                    if hasattr(content, "text"):
                        text_parts.append(content.text)
                if text_parts:
                    agent_data[agent_name]["result"] = " ".join(text_parts)

        # Build SubtaskResult and PlannedSubtask lists
        subtask_results: list[SubtaskResult] = []
        planned_subtasks: list[PlannedSubtask] = []

        for agent_name, data in agent_data.items():
            tool_names = [tc.get("tool", "") for tc in data["tool_calls"]]
            subtask_results.append(
                SubtaskResult(
                    agent_name=agent_name,
                    subtask=f"Delegated to {agent_name}",
                    result=data["result"],
                    status=SubtaskStatus.COMPLETED if data["result"] else SubtaskStatus.FAILED,
                    tool_calls=data["tool_calls"],
                )
            )
            planned_subtasks.append(
                PlannedSubtask(
                    agent_name=agent_name,
                    description=f"Delegated to {agent_name}",
                    tools_needed=tool_names,
                )
            )

        # If no child agents were tracked (e.g. supervisor handled it directly),
        # fall back to a single supervisor result
        if not subtask_results:
            subtask_results.append(
                SubtaskResult(
                    agent_name="Supervisor",
                    subtask="Direct response",
                    result=str(result.final_output),
                    status=SubtaskStatus.COMPLETED,
                )
            )

        return subtask_results, planned_subtasks

    async def orchestrate_manual(self, user_input: str) -> SupervisorOutput:
        """Run orchestration with explicit manual decomposition and delegation.

        Unlike orchestrate() which relies on SDK handoffs, this method explicitly
        calls the supervisor for planning, then individually runs each child agent.
        This provides more control and better state tracking for HITL support.

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
