"""
LangGraph-based execution engine for Magentic meta-agent system.

This module provides state management, checkpointing, and message passing
while preserving the dynamic meta-agent behavior. The AI coordinator still
generates unique agent topologies per query - LangGraph just provides better
execution infrastructure.
"""

from typing import TypedDict, Dict, Any, List, Optional, Annotated, cast, TYPE_CHECKING
from datetime import datetime
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from rich.console import Console

if TYPE_CHECKING:
    from .meta_coordinator import ExecutionPlan

console = Console()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries for LangGraph state updates."""
    return {**left, **right}


def max_value(left: int, right: int) -> int:
    """Return maximum of two integers (for current_layer tracking)."""
    return max(left, right)


# ============================================================================
# STATE SCHEMA
# ============================================================================

class MagenticState(TypedDict):
    """
    State shared across all agents in the execution graph.
    
    This preserves all data between agent executions and enables:
    - Agents to access outputs from dependencies
    - Checkpointing for crash recovery
    - Progress tracking and visualization
    - Message passing between agents
    """
    # Input
    query: str
    
    # Execution tracking
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]  # Agent ID -> output
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]  # Timeline of events
    current_layer: Annotated[int, max_value]  # Current layer (max from parallel agents)
    total_layers: int
    agent_to_layer: Dict[str, Any]  # Maps agent_id to {layer, index}
    
    # Agent communication
    messages: Annotated[List[BaseMessage], operator.add]  # For inter-agent messages
    
    # Metadata
    session_id: str
    start_time: str
    
    # Final result
    final_output: Optional[str]


# ============================================================================
# DYNAMIC GRAPH BUILDER
# ============================================================================

class MagenticGraphBuilder:
    """
    Builds LangGraph execution graphs dynamically from AI-generated plans.
    
    CRITICAL: This does NOT hardcode agent workflows. Each query gets a unique
    graph structure determined by the meta-coordinator's AI-driven planning.
    """
    
    def __init__(self, meta_system):
        """
        Initialize with reference to the meta-agent system.
        
        Args:
            meta_system: MetaAgentSystem instance with coordinator and tools
        """
        self.meta_system = meta_system
        self.checkpointer = MemorySaver()
    
    def build_dynamic_graph(self, execution_plan: "ExecutionPlan") -> CompiledStateGraph:
        """
        Build a LangGraph StateGraph from AI-generated execution plan.
        
        This converts the coordinator's plan into a graph structure while
        preserving the dynamic topology. Each call creates a unique graph!
        
        Args:
            execution_plan: Plan from meta_coordinator.create_execution_plan()
                Contains: agents list with roles, dependencies, and layers
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Extract plan details
        agents = execution_plan.agents
        layers = execution_plan.get_execution_layers()
        
        # Build mapping of agent_id to (layer_num, agent_idx)
        agent_to_layer = {}
        for layer_num, layer_agents in enumerate(layers):
            for agent_idx in layer_agents:
                agent_id = f"{agents[agent_idx]['role']}_{agent_idx}"
                agent_to_layer[agent_id] = {'layer': layer_num, 'index': agent_idx}
        
        # Store in class for use in state initialization
        self.agent_to_layer = agent_to_layer
        self.total_layers = len(layers)
        self.total_agents = len(agents)
        
        # Create new graph
        graph = StateGraph(MagenticState)
        
        # Add agent nodes dynamically
        for idx, agent_config in enumerate(agents):
            agent_id = f"{agent_config['role']}_{idx}"
            
            # Create node function for this agent
            node_func = self._create_agent_node(agent_id, agent_config, agents)
            graph.add_node(agent_id, node_func)
        
        # Add edges based on dependencies
        self._add_dynamic_edges(graph, agents, layers)
        
        # Compile with checkpointing
        compiled = graph.compile(checkpointer=self.checkpointer)
        
        return compiled
    
    def _create_agent_node(self, agent_id: str, agent_config: Dict[str, Any], all_agents: List[Dict]):
        """
        Create a node function for a specific agent.
        
        The node function receives state, executes the agent, and updates state.
        
        Args:
            agent_id: Unique identifier for this agent (e.g., "researcher_0")
            agent_config: Agent configuration from ExecutionPlan
            all_agents: Full list of agents to resolve dependencies
        """
        role = agent_config["role"]
        task = agent_config["task"]
        depends_on = agent_config.get("depends_on", [])
        
        async def agent_node(state: MagenticState) -> Dict[str, Any]:
            """Execute this agent and update state."""
            console.print(f"\n[yellow]→ Executing {agent_id} ({role})...[/yellow]")
            
            # Get this agent's layer and index from mapping
            agent_info = state.get("agent_to_layer", {}).get(agent_id, {'layer': 0, 'index': 0})
            agent_layer = agent_info['layer']
            agent_idx = agent_info['index']
            total_layers = state.get("total_layers", 1)
            total_agents = len(all_agents)
            
            # Gather context from dependencies (depends_on contains agent indices)
            context_parts = []
            for dep_idx in depends_on:
                # Generate the dependency agent's ID from its index and role
                dep_agent_id = f"{all_agents[dep_idx]['role']}_{dep_idx}"
                if dep_agent_id in state["agent_outputs"]:
                    dep_output = state["agent_outputs"][dep_agent_id]
                    context_parts.append(f"From {dep_agent_id}:\n{dep_output}")
                    console.print(f"  [dim]Using output from {dep_agent_id}[/dim]")
            
            # Pass dependency outputs (empty string means no prior agent outputs)
            context = "\n\n".join(context_parts)
            
            # Execute agent using meta_system
            try:
                console.print(f"  [dim]Task: {task[:80]}...[/dim]")
                result = await self.meta_system.execute_agent_for_langgraph(
                    agent_id=agent_id,
                    role=role,
                    task=task,
                    context=context,
                    original_query=state["query"],  # Pass the actual user query
                    layer=agent_layer,
                    total_layers=total_layers,
                    agent_number=agent_idx + 1,  # 1-indexed for display
                    total_agents=total_agents
                )
                
                console.print(f"[green]✓ {agent_id} completed ({len(str(result))} chars)[/green]")
                
                # Update state
                return {
                    "agent_outputs": {agent_id: result},
                    "current_layer": agent_layer,  # Update to this agent's layer
                    "execution_trace": [{
                        "agent_id": agent_id,
                        "role": role,
                        "layer": agent_layer,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed",
                        "output_length": len(str(result))
                    }]
                }
            
            except Exception as e:
                console.print(f"[red]Error in {agent_id}: {e}[/red]")
                return {
                    "agent_outputs": {agent_id: f"Error: {str(e)}"},
                    "execution_trace": [{
                        "agent_id": agent_id,
                        "role": role,
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed",
                        "error": str(e)
                    }]
                }
        
        return agent_node
    
    def _add_dynamic_edges(self, graph: StateGraph, agents: List[Dict], layers: List[List[int]]):
        """
        Add edges to graph based on AI-generated dependency structure.
        
        This creates the execution flow dynamically:
        - Agents with no dependencies connect from START
        - Dependencies create edges between agents
        - Final agents connect to END
        """
        # Helper to generate agent ID from index
        def get_agent_id(idx: int) -> str:
            return f"{agents[idx]['role']}_{idx}"
        
        # Layer 0 agents (no dependencies) connect from START
        if layers:
            for agent_idx in layers[0]:
                agent_id = get_agent_id(agent_idx)
                graph.add_edge(START, agent_id)
                console.print(f"  Edge: START → {agent_id}")
        
        # Add edges based on dependencies
        for idx, agent in enumerate(agents):
            agent_id = get_agent_id(idx)
            depends_on = agent.get("depends_on", [])
            
            # If has dependencies, connect from each dependency
            for dep_idx in depends_on:
                dep_id = get_agent_id(dep_idx)
                graph.add_edge(dep_id, agent_id)
                console.print(f"  Edge: {dep_id} → {agent_id}")
        
        # Last layer agents connect to END
        if layers:
            for agent_idx in layers[-1]:
                agent_id = get_agent_id(agent_idx)
                graph.add_edge(agent_id, END)
                console.print(f"  Edge: {agent_id} → END")
    
    def create_initial_state(self, query: str) -> MagenticState:
        """
        Create initial state for graph execution.
        
        Args:
            query: User query to process
        
        Returns:
            Initial MagenticState with query and empty collections
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return MagenticState(
            query=query,
            agent_outputs={},
            execution_trace=[],
            current_layer=0,
            total_layers=0,  # Will be set after graph build
            agent_to_layer={},  # Will be set after graph build
            messages=[],
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            final_output=None
        )


# ============================================================================
# EXECUTION WRAPPER
# ============================================================================

class LangGraphExecutor:
    """
    High-level wrapper for executing queries with LangGraph infrastructure.
    
    This class maintains the meta-agent dynamic behavior while providing
    state management, checkpointing, and streaming capabilities.
    """
    
    def __init__(self, meta_system):
        """
        Initialize executor with meta-agent system.
        
        Args:
            meta_system: MetaAgentSystem instance
        """
        self.meta_system = meta_system
        self.graph_builder = MagenticGraphBuilder(meta_system)
    
    async def execute_query(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Execute a query with full LangGraph infrastructure.
        
        Process:
        1. AI coordinator generates dynamic execution plan
        2. Build LangGraph from plan (unique per query!)
        3. Execute with state management and checkpointing
        4. Return results with full execution trace
        
        Args:
            query: User query to process
            stream: If True, stream updates in real-time
        
        Returns:
            Dict with final_output, execution_trace, and metadata
        """
        # Step 1: AI generates dynamic plan
        console.print("[cyan]→ AI Coordinator analyzing query...[/cyan]")
        execution_plan = self.meta_system.coordinator.create_execution_plan(query)
        console.print(f"[green]✓ Plan created: {len(execution_plan.agents)} agents in {len(execution_plan.get_execution_layers())} layers[/green]")
        
        # Step 2: Build dynamic graph from AI plan
        console.print("[cyan]→ Building execution graph...[/cyan]")
        graph = self.graph_builder.build_dynamic_graph(execution_plan)
        
        # Step 3: Create initial state
        initial_state = self.graph_builder.create_initial_state(query)
        initial_state["total_layers"] = self.graph_builder.total_layers
        initial_state["agent_to_layer"] = self.graph_builder.agent_to_layer
        console.print(f"[green]✓ Graph ready[/green]\n")
        
        # Step 4: Execute with checkpointing
        config: RunnableConfig = {"configurable": {"thread_id": initial_state["session_id"]}}  # type: ignore
        
        if stream:
            # Stream execution
            final_state: Optional[MagenticState] = None
            async for update in graph.astream(initial_state, config):
                final_state = update  # type: ignore
        else:
            # Batch execution
            final_state = await graph.ainvoke(initial_state, config)  # type: ignore
        
        console.print("\n[bold green]✓ Complete![/bold green]\n")
        
        # Get final agent output (last agent in execution)
        final_output = ""
        if final_state and "agent_outputs" in final_state:
            outputs = final_state["agent_outputs"]
            if outputs:
                # Get last output
                final_output = list(outputs.values())[-1]
        
        # Update conversation history in meta_system
        self.meta_system.conversation_history.append({"role": "user", "content": query})
        self.meta_system.conversation_history.append({"role": "assistant", "content": final_output})
        
        return {
            "query": query,
            "final_output": final_output,
            "execution_trace": final_state.get("execution_trace", []) if final_state else [],
            "session_id": initial_state["session_id"],
            "agent_count": len(execution_plan.agents),
            "layer_count": len(execution_plan.get_execution_layers()),
            "execution_plan": execution_plan
        }
    
    def execute_query_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute_query.
        
        Args:
            query: User query to process
        
        Returns:
            Execution results
        """
        import asyncio
        return asyncio.run(self.execute_query(query))
