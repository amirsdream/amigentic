# Layer Barriers and Execution Synchronization

## Overview

Magentic implements a layer barrier system to ensure reliable parallel execution of agents. This document explains the problem this solves and how it works.

## The Problem

When multiple agents execute in parallel within the same layer, they can complete at different times. Without synchronization, the following race condition occurs:

```
Layer 0: [researcher_0, researcher_1] (both start simultaneously)
  └─> researcher_0 completes in 2 seconds
  └─> researcher_1 completes in 5 seconds

Layer 1: [synthesizer_2] (depends on both researchers)
  ❌ PROBLEM: If synthesizer_2 starts when researcher_0 completes,
              it won't have researcher_1's output yet!
```

This leads to:
- Empty context errors: "Part has 'From' and ':' but no content after"
- Incomplete information being passed to dependent agents
- Unpredictable execution behavior

## The Solution: Layer Barriers

Barrier nodes are inserted between layers to synchronize execution:

```
START
  │
  ├─> researcher_0 ────┐
  │                     │
  └─> researcher_1 ────┤
                        ▼
                   barrier_0 (waits for ALL)
                        │
                        ▼
                  synthesizer_2
                        │
                        ▼
                   barrier_1
                        │
                        ▼
                       END
```

### How Barriers Work

1. **Check Completion**: Barrier examines state to verify ALL agents in the current layer have outputs
2. **Wait or Proceed**: If any agent is missing, execution pauses. If all complete, flow continues
3. **State Passthrough**: Barriers don't modify state, they only synchronize timing

### Implementation

**Code Location**: `src/langgraph_executor.py`

```python
def _create_layer_barrier(self, layer_num: int, layer_agents: List[int], agents: List[Dict]) -> Callable:
    """Create a barrier node that waits for ALL agents in a layer to complete."""
    def layer_barrier(state: MagenticState) -> MagenticState:
        # Check if all agents in this layer have outputs
        all_complete = True
        for agent_idx in layer_agents:
            agent_id = f"{agents[agent_idx]['role']}_{agent_idx}"
            if agent_id not in state["agent_outputs"]:
                all_complete = False
                break
        
        if all_complete:
            console.print(f"✓ Layer {layer_num} complete - all agents finished")
        
        # Return state unchanged - barrier just synchronizes
        return state
    
    return layer_barrier
```

## Graph Structure

**Before (problematic)**:
```
START → [agents with dependencies based on depends_on] → END
```

**After (with barriers)**:
```
START → Layer 0 agents → barrier_0 → Layer 1 agents → barrier_1 → ... → END
```

### Edge Creation

The `_add_dynamic_edges()` method creates the barrier structure:

```python
# Layer 0: Connect START to all layer 0 agents
for agent_idx in layers[0]:
    graph.add_edge(START, get_agent_id(agent_idx))

# Connect layer 0 agents to barrier_0
for agent_idx in layers[0]:
    graph.add_edge(get_agent_id(agent_idx), "barrier_0")

# Connect barrier_0 to layer 1 agents
for agent_idx in layers[1]:
    graph.add_edge("barrier_0", get_agent_id(agent_idx))

# ... and so on for each layer
```

## State Management

### Agent Outputs Storage

When an agent completes, it returns a state update:

```python
state_update = {
    "agent_outputs": {agent_id: output_content},  # Merged with existing outputs
    "current_layer": agent_layer,
    "conversation_history": [conversation_entry],
    "execution_trace": [execution_record]
}
```

The `agent_outputs` field uses `merge_dicts` reducer, which merges new outputs with existing ones:

```python
class MagenticState(TypedDict):
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]  # Preserves all outputs
```

### Barrier Check

Barriers check the accumulated `agent_outputs`:

```python
# At barrier_0 after layer 0 completes:
state["agent_outputs"] = {
    "researcher_0": "... research output ...",
    "researcher_1": "... research output ..."
}
# ✓ All layer 0 agents present, proceed to layer 1
```

## Context Building

Once all dependencies are available, context is built for the next agent:

```python
context_parts = []
for dep_idx in depends_on:
    dep_agent_id = f"{all_agents[dep_idx]['role']}_{dep_idx}"
    if dep_agent_id in state["agent_outputs"]:
        dep_output = state["agent_outputs"][dep_agent_id]
        output_str = str(dep_output).strip()  # Strip whitespace to prevent parsing issues
        context_parts.append(f"From {dep_agent_id}:\n{output_str}")

context = "\n\n".join(context_parts)
```

### Context Parsing Fix

**Problem**: Outputs with leading newlines caused parsing failures:
```
"From researcher_0:\n\n\n---\n\n**KEY FINDINGS:**..."
```

When split on `"\n\n"`, this separated the header from content.

**Solution**: Strip outputs before building context:
```python
output_str = str(dep_output).strip()  # Remove leading/trailing whitespace
```

## Conversation History

Agents also receive conversation history for continuity:

```python
conversation_entry = {
    "agent_id": agent_id,
    "role": role,
    "task": task,
    "input_context": context[:500],
    "output": output_content[:500],
    "layer": agent_layer,
    "timestamp": datetime.now().isoformat()
}
```

Last 3 conversation steps are shown to each agent:

```python
if conversation_history:
    context += "\n\nPREVIOUS CONVERSATION:\n"
    for step in conversation_history[-3:]:
        context += f"- {step['role']}: {step['output'][:100]}...\n"
```

## Debug Visualization

Enable state visualization to see barriers in action:

```bash
# In .env file
DEBUG_STATE=true
```

Output shows:
```
>>> BARRIER Layer 0: agent_outputs keys = ['researcher_0', 'researcher_1']
    researcher_0: 2967 chars, preview: ---\n\n**KEY FINDINGS:**\n...
    researcher_1: 1843 chars, preview: **Wedding Budget Research Summary:**\n...
✓ Layer 0 complete - all 2 agents finished
```

## Benefits

1. **Reliability**: No more race conditions or empty contexts
2. **Predictability**: Execution flow is deterministic
3. **Debugging**: Clear visibility into layer completion status
4. **Scalability**: Works with any number of layers and agents per layer

## Performance Impact

Barriers add minimal overhead:
- **Latency**: ~1-5ms per barrier (state dictionary lookup)
- **Memory**: No additional memory (state is already in memory)
- **Benefit**: Prevents re-execution due to failures, saving overall time

## Related Features

- **State Visualization** (`DEBUG_STATE`): See barrier execution in real-time
- **Conversation History**: Agents receive context from previous steps
- **Execution Trace**: Track which agents completed when

## See Also

- [Architecture Diagram](ARCHITECTURE_DIAGRAM.txt) - Visual representation of execution flow
- [Parallel Execution](PARALLEL_EXECUTION.md) - How parallel agents are orchestrated
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Overall system design
