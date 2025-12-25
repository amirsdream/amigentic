# Magentic Architecture

## Overview

Magentic is a dynamic meta-agent system that uses AI to generate unique agent topologies for each query. The system features a modern web UI with real-time WebSocket streaming for live execution visualization.

## System Flow

```
User Query (Web UI or CLI)
    ↓
WebSocket Connection / Direct API
    ↓
Meta-Coordinator (AI analyzes complexity)
    ↓
Dynamic Agent Plan (unique per query)
    ↓
LangGraph Execution (state + checkpointing)
    ↓
Layer-Based Parallel Execution
    ↓  (Real-time WebSocket updates)
Final Output → Web UI or Terminal
```

## Architecture Layers

### 1. Presentation Layer (Frontend)
- **React Application**: Modern SPA with Vite build system
- **WebSocket Client**: Real-time bidirectional communication
- **Markdown Renderer**: Beautiful formatting with syntax highlighting
- **State Management**: React hooks for UI state
- **Components**: Message display, execution visualization, agent steps

### 2. API Layer (Backend)
- **FastAPI Server**: Async web framework with WebSocket support
- **REST Endpoints**: `/health`, `/query`, `/memory`
- **WebSocket Endpoint**: `/ws` for real-time streaming
- **Progress Streaming**: Events for plan, agent_start, agent_complete, complete, error

### 3. Coordination Layer
- **Meta-Coordinator**: AI-driven planning with complexity assessment
- **Plan Generation**: Creates ExecutionPlan with agents and dependencies
- **Complexity Scaling**: Simple (1 agent), Medium (1-2), Complex (2+ with synthesizer)

### 4. Execution Layer
- **LangGraph Executor**: DAG-based parallel execution
- **State Management**: Persistent state across agent executions
- **Checkpointing**: Resume from failures
- **Layer Execution**: Topological sort for optimal parallelization

### 5. Agent Layer
- **Role Library**: 8 specialized agent roles
- **Dynamic Instantiation**: Agents created per execution plan
- **Tool Integration**: DuckDuckGo search for researchers
- **Memory**: Conversation history (last 4 messages)

## Core Components

### 1. Meta-Coordinator
- **Purpose**: AI-driven planning engine
- **Input**: User query + conversation context
- **Output**: ExecutionPlan with dynamic agent topology
- **Temperature**: 0.3 (for consistent JSON)

### 2. ExecutionPlan
- **Agents**: List of {role, task, depends_on}
- **Roles**: researcher, analyzer, planner, writer, coder, critic, synthesizer, coordinator
- **Dependencies**: Agent indices for execution order
- **Layers**: Topological sort for parallel execution

### 3. LangGraph Executor
- **State Management**: MagenticState (query, agent_outputs, trace, etc.)
- **Checkpointing**: MemorySaver for crash recovery
- **Graph Building**: Dynamic nodes/edges from AI plan
- **Execution**: Async with proper dependency resolution

### 4. Meta-Agent System
- **Agent Execution**: LLM + role-based system prompts
- **Tools**: DuckDuckGo search for researchers
- **Memory**: Conversation history (last 4 messages)
- **Observability**: Phoenix tracing per agent

## Execution Example

**Query**: "Compare Python and Rust"

**AI Plan**:
```json
{
  "agents": [
    {"role": "researcher", "task": "Python info", "depends_on": []},
    {"role": "researcher", "task": "Rust info", "depends_on": []},
    {"role": "analyzer", "task": "Compare", "depends_on": [0, 1]},
    {"role": "synthesizer", "task": "Report", "depends_on": [2]}
  ]
}
```

**Graph**:
```
START → [researcher_0, researcher_1] (parallel)
researcher_0 → analyzer_2
researcher_1 → analyzer_2
analyzer_2 → synthesizer_3
synthesizer_3 → END
```

**Execution**: 3 layers, 4 agents, parallel researchers

## Key Features

- **Dynamic**: Each query gets unique topology
- **Parallel**: Independent agents run concurrently
- **Stateful**: LangGraph preserves state across agents
- **Resumable**: Checkpoints enable crash recovery
- **Observable**: Phoenix dashboard for tracing
- **Validated**: Auto-fixes illogical dependencies

## State Schema

```python
class MagenticState(TypedDict):
    query: str                    # User query
    agent_outputs: Dict[str, Any] # Agent results
    execution_trace: List[Dict]   # Event timeline
    current_layer: int            # Execution progress
    total_layers: int             # Total layers
    messages: List[BaseMessage]   # Inter-agent msgs
    session_id: str               # Unique session
    start_time: str               # Timestamp
    final_output: Optional[str]   # Final result
```

## Complexity Scaling

| Score | Complexity | Agents | Depth |
|-------|------------|--------|-------|
| < 1   | Very Simple | 1-2   | 1     |
| 1-2   | Simple     | 2-4   | 2     |
| 3-4   | Moderate   | 4-6   | 3     |
| 5-7   | Complex    | 6-8   | 4     |
| 8+    | Very Complex| 8-12+ | 5     |
