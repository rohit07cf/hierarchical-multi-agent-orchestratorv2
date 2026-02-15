# Hierarchical Multi-Agent Orchestrator

A production-grade hierarchical multi-agent orchestration system built with the OpenAI Agents SDK, featuring supervisor-driven task decomposition, HITL (Human-In-The-Loop) state management, Temporal workflow durability, and real-time streaming.

## Architecture

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │  Agent       │
                    │ [reasoning]  │
                    └──────┬───────┘
           ┌───────┬───────┼───────┬────────┐
           ▼       ▼       ▼       ▼        ▼
     ┌─────────┐┌──────┐┌──────┐┌───────────┐
     │ Simple  ││ Math ││ Echo ││Classifier │
     │ Agent   ││Agent ││Agent ││  Agent    │
     │[add,    ││[add, ││[echo,││[classify, │
     │ echo]   ││sub,  ││rev]  ││ sentiment]│
     │         ││mul]  ││      ││           │
     └─────────┘└──────┘└──────┘└───────────┘
```

### Design Patterns

- **Supervisor-Child Agent Pattern**: One supervisor orchestrates multiple specialized child agents
- **Template Method Pattern**: Base agent class defines execution pipeline; subclasses customize behavior
- **ReAct Pattern**: Child agents use Reasoning + Acting loops for multi-step execution
- **HITL State Restoration**: Pause, review, revise, and resume agent execution at any checkpoint

## Features

- **Hierarchical Agent Management**: AgentTree and AgentNode for tree-based agent organization with visualization
- **Task Decomposition**: Supervisor analyzes requests and routes subtasks to specialized agents
- **4 Specialized Child Agents**: SimpleAgent, MathAgent, EchoAgent, ClassifierAgent
- **HITL Support**: Pause/resume with Cancel, Revise, and Approve actions
- **Real-time Streaming**: StreamingCallbackHandler with async queue for live UI updates
- **Temporal Workflows**: Durable orchestration with retry logic and heartbeat monitoring
- **Structured Outputs**: Pydantic models throughout for type safety
- **Streamlit UI**: Interactive interface with agent tree visualization, reasoning panel, and state inspector

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4.1-nano
LOG_LEVEL=INFO
```

### Running

```bash
streamlit run main.py
```

### With Docker (includes Temporal)

```bash
docker-compose up
```

## Project Structure

```
├── agents/
│   ├── base_agent.py          # Abstract base with Template Method pattern
│   ├── supervisor.py          # Supervisor: task decomposition & delegation
│   ├── simple_agent.py        # SimpleAgent: add_numbers, echo_text
│   ├── math_agent.py          # MathAgent: add, subtract, multiply
│   ├── echo_agent.py          # EchoAgent: echo_text, reverse_text
│   └── classifier_agent.py   # ClassifierAgent: classify_intent, detect_sentiment
├── tools/
│   ├── math_tools.py          # Mathematical operation tools
│   ├── text_tools.py          # Text manipulation tools
│   ├── classification_tools.py # NLP classification tools
│   └── supervisor_tools.py    # Supervisor reasoning tool
├── models/
│   ├── agent_state.py         # AgentState, HITLAction, IntermediateStep
│   ├── supervisor_output.py   # SupervisorOutput, SubtaskResult, TaskDecomposition
│   ├── streaming_models.py    # StreamingModelResponseStep
│   └── tool_models.py         # ToolResult, ToolCall, ToolDefinition
├── orchestration/
│   ├── agent_tree.py          # AgentTree & AgentNode with Graphviz visualization
│   ├── streaming_handler.py   # StreamingCallbackHandler (AgentHooks)
│   ├── hitl_manager.py        # HITL state capture, persist, restore
│   └── temporal_workflow.py   # Temporal workflow & activity definitions
├── prompts/
│   ├── supervisor_prompt.py   # Supervisor task decomposition prompt
│   ├── react_prompt.py        # ReAct pattern prompt template
│   └── tool_selection_prompt.py # Tool selection guidance prompt
├── ui/
│   ├── streamlit_app.py       # Main Streamlit application
│   ├── components.py          # Reusable UI components
│   └── visualizations.py      # Charts, tables, tree rendering
├── hooks/
│   └── run_hooks.py           # AgentHooks & ToolHooksImpl
├── config/
│   └── settings.py            # Pydantic Settings configuration
├── utils/
│   ├── logging.py             # Structured logging setup
│   ├── validators.py          # Pydantic validation helpers
│   └── serializers.py         # JSON serialization with enhanced type support
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Temporal + app containers
└── README.md
```

## Data Models

### AgentState
Captures complete execution state for pause/resume and HITL:
- `current_inputs`: Active input parameters
- `intermediate_steps`: Ordered execution trace
- `tool_path`: Hierarchical path (e.g., `Supervisor.ClassifierAgent`)
- `iteration_count`: ReAct loop iterations
- `is_paused`: HITL pause flag
- `hitl_actions`: History of user interventions

### SupervisorOutput
Structured output from orchestration:
- `final_answer`: Aggregated response
- `subtasks`: List of SubtaskResult with per-agent outcomes
- `decomposition`: TaskDecomposition plan

### StreamingModelResponseStep
Real-time UI update events:
- Token streaming, tool calls, tool results, errors, HITL pauses

## Example Interaction

```
User: "What's the sentiment of 'I love this product'? Also multiply 5 * 3"

Supervisor:
  REASON: User wants sentiment analysis AND math calculation
  ACTION: Decompose into 2 subtasks

Subtask 1 -> ClassifierAgent.detect_sentiment:
  Result: sentiment = "positive", confidence = 0.70

Subtask 2 -> MathAgent.multiply_numbers:
  Result: 5 x 3 = 15

Final Answer: Sentiment is positive (70% confidence). 5 x 3 = 15.
```

## Tech Stack

- **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)**: Agent framework with handoffs, tools, and hooks
- **[Temporal](https://temporal.io)**: Workflow orchestration and durability
- **[Pydantic](https://docs.pydantic.dev)**: Data validation and structured outputs
- **[Streamlit](https://streamlit.io)**: Interactive UI
- **[Graphviz](https://graphviz.org)**: Agent hierarchy visualization
