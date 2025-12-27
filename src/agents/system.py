"""Meta-agent system - dynamically creates and executes agents based on coordinator's plan."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

from langchain_core.tools import BaseTool

from ..config import Config
from ..role_library import RoleLibrary
from ..coordinator import MetaCoordinator
from ..ui import ExecutionVisualizer
from .llm_factory import create_llm
from .executor import AgentExecutor

from ..tools.manager import ToolManager

if TYPE_CHECKING:
    from ..services.rag import RAGService

logger = logging.getLogger(__name__)


class MetaAgentSystem:
    """Dynamic meta-agent system."""

    def __init__(
        self,
        config: Config,
        tools: List[BaseTool],
        rag_service: Optional["RAGService"] = None,
        tool_manager: Optional[ToolManager] = None,
    ):
        """Initialize meta-agent system.

        Args:
            config: Application configuration.
            tools: Available tools.
            rag_service: Optional RAG service for active knowledge retrieval.
            tool_manager: Optional tool manager for role-specific MCP tools.
        """
        self.config = config
        self.tools = tools
        self.rag_service = rag_service
        self.tool_manager = tool_manager
        self.role_library = RoleLibrary()

        # Initialize LLM
        self.llm = create_llm(config)
        logger.info(f"✓ Initialized {config.llm_provider} LLM: {self.llm.__class__.__name__}")

        # Initialize coordinator with RAG support
        self.coordinator = MetaCoordinator(config, self.llm, rag_service=rag_service)

        # Initialize executor
        self.agent_executor = AgentExecutor(
            llm=self.llm,
            tools=tools,
            role_library=self.role_library,
            ui_display_limit=config.ui_display_limit,
            tool_manager=tool_manager,
        )

        # Conversation memory
        self.conversation_history: List[Dict[str, str]] = []

        # Visualization
        self.visualizer = ExecutionVisualizer()

        # Execution settings
        self.absolute_max_depth = 5
        self.max_parallel_agents = config.max_parallel_agents
        self._semaphore = asyncio.Semaphore(self.max_parallel_agents)

    def process_query(
        self, query: str, depth: int = 0, max_depth: int | None = None
    ) -> Dict[str, Any]:
        """Process a query using dynamic agent creation.

        Args:
            query: User's query.
            depth: Current execution depth (for hierarchical agents).
            max_depth: Maximum depth for this query branch.

        Returns:
            Result dictionary with final answer and execution trace.
        """
        if depth >= self.absolute_max_depth:
            logger.warning(f"🛑 Max depth {self.absolute_max_depth} reached")
            return {
                "final_answer": f"Maximum execution depth ({self.absolute_max_depth}) reached.",
                "trace": [],
                "plan": {
                    "description": "Depth limit exceeded",
                    "agents": [],
                    "execution_layers": 0,
                },
            }

        if max_depth is None:
            max_depth = self._analyze_query_complexity(query)
            logger.info(f"🎯 Query complexity analysis: max_depth={max_depth}")

        max_depth = min(max_depth, self.absolute_max_depth)

        indent = "  " * depth
        logger.info(f"{indent}🚀 Processing query (depth {depth}/{max_depth}): {query[:100]}...")

        context = self._build_context() if depth == 0 else ""
        plan = self.coordinator.create_execution_plan(
            query, context, depth=depth, max_depth=max_depth
        )

        if depth == 0:
            self.visualizer.display_plan_tree(
                plan.description, plan.agents, depth=depth, max_depth=max_depth
            )

        # Execute plan
        execution_layers = plan.get_execution_layers()
        self._log_execution_layers(execution_layers, plan.agents)

        trace = []
        outputs = {}

        for layer_idx, agent_indices in enumerate(execution_layers):
            logger.info(f"\n{'='*60}")
            logger.info(
                f"🔀 LAYER {layer_idx + 1}/{len(execution_layers)}: Executing {len(agent_indices)} agents"
            )
            logger.info(f"{'='*60}")

            if len(agent_indices) > 1:
                self.visualizer.display_parallel_agents_start(
                    [plan.agents[i] for i in agent_indices], layer_idx + 1, len(execution_layers)
                )

            if len(agent_indices) == 1:
                output = self._execute_single_agent(
                    agent_indices[0],
                    plan.agents[agent_indices[0]],
                    plan.agents,
                    outputs,
                    query,
                    depth,
                    max_depth,
                    trace,
                    layer_idx,
                    len(execution_layers),
                )
                outputs[agent_indices[0]] = output
            else:
                layer_outputs = asyncio.run(
                    self._execute_layer_parallel(
                        agent_indices,
                        plan.agents,
                        outputs,
                        query,
                        depth,
                        max_depth,
                        trace,
                        layer_idx,
                        len(execution_layers),
                    )
                )
                outputs.update(layer_outputs)

            if len(agent_indices) > 1:
                logger.info(f"✅ Layer {layer_idx + 1} complete")
                self.visualizer.console.print(
                    f"\n[bold green]✅ Layer {layer_idx + 1}/{len(execution_layers)} complete[/bold green]\n"
                )

        final_answer = outputs[len(plan.agents) - 1] if outputs else "No output generated"

        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        result = {
            "final_answer": final_answer,
            "trace": trace,
            "plan": {
                "description": plan.description,
                "agents": [a["role"] for a in plan.agents],
                "execution_layers": len(execution_layers),
                "parallelization": f"{sum(len(layer) for layer in execution_layers)} total in {len(execution_layers)} layers",
            },
            "agents_spec": plan.agents,
            "execution_layers": execution_layers,
        }

        self.visualizer.display_summary(result)
        return result

    def _log_execution_layers(self, layers: List[List[int]], agents: List[Dict[str, Any]]) -> None:
        """Log execution layer details."""
        logger.info("")
        logger.info("🔀" + "=" * 70)
        logger.info(f"🔀 PARALLEL EXECUTION: {len(layers)} layers total")
        logger.info("🔀" + "=" * 70)
        for layer_idx, layer in enumerate(layers):
            layer_agents = [agents[i]["role"] for i in layer]
            if len(layer) > 1:
                logger.info(
                    f"🔀 Layer {layer_idx}: ⚡ {len(layer)} agents IN PARALLEL - {layer_agents}"
                )
            else:
                logger.info(f"🔀 Layer {layer_idx}: 1 agent (sequential) - {layer_agents}")
        logger.info("🔀" + "=" * 70)
        logger.info("")

    def _execute_single_agent(
        self,
        agent_index: int,
        agent_spec: Dict[str, Any],
        all_agents: List[Dict[str, Any]],
        completed_outputs: Dict[int, str],
        query: str,
        depth: int,
        max_depth: int,
        trace: List[Dict[str, Any]],
        layer_idx: int = 0,
        total_layers: int = 1,
    ) -> str:
        """Execute a single agent and update trace."""
        role_name = agent_spec.get("role")
        task = agent_spec.get("task")

        if not role_name or not task:
            logger.error(f"Invalid agent spec: {agent_spec}")
            return ""

        logger.info(f"🤖 Agent {agent_index}: {role_name.upper()}")
        logger.info(f"   Task: {task}")

        self.visualizer.display_execution_progress(
            current_step=agent_index + 1,
            total_steps=len(all_agents),
            role=role_name,
            task=task,
            status="running",
            layer=layer_idx + 1,
            total_layers=total_layers,
        )

        role = self.role_library.get_role(role_name)
        if not role:
            error_msg = f"Unknown role '{role_name}' - valid: {self.role_library.list_roles()}"
            logger.error(f"❌ {error_msg}")
            return f"[ERROR: {error_msg}]"

        depends_on = agent_spec.get("depends_on", [])
        previous_outputs = [completed_outputs[i] for i in depends_on if i in completed_outputs]

        result = self.agent_executor.execute(
            role=role,
            task=task,
            original_query=query,
            previous_outputs=previous_outputs,
            conversation_history=self.conversation_history,
            depth=depth,
            max_depth=max_depth,
            process_query_callback=self.process_query,
        )

        output = result.get("content", str(result)) if isinstance(result, dict) else str(result)

        trace.append(
            {
                "step": agent_index,
                "role": role_name,
                "task": task,
                "depends_on": depends_on,
                "parallel": False,
                "output": output[:200] + "..." if len(output) > 200 else output,
            }
        )

        return output

    async def _execute_layer_parallel(
        self,
        agent_indices: List[int],
        all_agents: List[Dict[str, Any]],
        completed_outputs: Dict[int, str],
        query: str,
        depth: int,
        max_depth: int,
        trace: List[Dict[str, Any]],
        layer_idx: int = 0,
        total_layers: int = 1,
    ) -> Dict[int, str]:
        """Execute multiple agents in parallel."""
        logger.info(
            f"⚡ Executing {len(agent_indices)} agents in parallel (max {self.max_parallel_agents} concurrent)..."
        )

        tasks = []
        for i in agent_indices:
            task = asyncio.create_task(
                self._execute_agent_with_limit(
                    i,
                    all_agents[i],
                    all_agents,
                    completed_outputs,
                    query,
                    depth,
                    max_depth,
                    layer_idx,
                    total_layers,
                )
            )
            tasks.append((i, task))

        results = {}
        for i, task in tasks:
            output = await task
            results[i] = output

            trace.append(
                {
                    "step": i,
                    "role": all_agents[i].get("role"),
                    "task": all_agents[i].get("task"),
                    "depends_on": all_agents[i].get("depends_on", []),
                    "parallel": True,
                    "output": output[:200] + "..." if len(output) > 200 else output,
                }
            )

        return results

    async def _execute_agent_with_limit(
        self,
        agent_index: int,
        agent_spec: Dict[str, Any],
        all_agents: List[Dict[str, Any]],
        completed_outputs: Dict[int, str],
        query: str,
        depth: int,
        max_depth: int,
        layer_idx: int = 0,
        total_layers: int = 1,
    ) -> str:
        """Execute agent with semaphore to limit concurrency."""
        async with self._semaphore:
            logger.info(f"🔓 Agent {agent_index} acquired semaphore slot")
            result = await self._execute_agent_async(
                agent_index,
                agent_spec,
                all_agents,
                completed_outputs,
                query,
                depth,
                max_depth,
                layer_idx,
                total_layers,
            )
            logger.info(f"🔒 Agent {agent_index} released semaphore slot")
            return result

    async def _execute_agent_async(
        self,
        agent_index: int,
        agent_spec: Dict[str, Any],
        all_agents: List[Dict[str, Any]],
        completed_outputs: Dict[int, str],
        query: str,
        depth: int,
        max_depth: int,
        layer_idx: int = 0,
        total_layers: int = 1,
    ) -> str:
        """Async wrapper for executing an agent."""
        role_name = agent_spec.get("role")
        task = agent_spec.get("task")

        if not role_name or not task:
            logger.error(f"Invalid agent spec: {agent_spec}")
            return ""

        logger.info(f"⚡ [PARALLEL Layer {layer_idx + 1}] Agent {agent_index}: {role_name.upper()}")

        role = self.role_library.get_role(role_name)
        if not role:
            logger.error(f"Unknown role: {role_name}")
            return ""

        depends_on = agent_spec.get("depends_on", [])
        previous_outputs = [completed_outputs[i] for i in depends_on if i in completed_outputs]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.agent_executor.execute,
            role,
            task,
            query,
            previous_outputs,
            self.conversation_history,
            depth,
            max_depth,
            self.process_query,
        )

        output = result.get("content", str(result)) if isinstance(result, dict) else str(result)
        logger.info(f"✅ [PARALLEL] Agent {agent_index} completed: {role_name.upper()}")
        return output

    async def execute_agent_for_langgraph(
        self,
        agent_id: str,
        role: str,
        task: str,
        context: str,
        original_query: str,
        layer: int = 0,
        total_layers: int = 1,
        agent_number: int = 1,
        total_agents: int = 1,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute a single agent for LangGraph integration."""
        role_obj = self.role_library.get_role(role)
        if not role_obj:
            logger.error(f"Unknown role: {role}")
            return {"content": f"[ERROR: Unknown role '{role}']", "tool_calls": []}

        self.visualizer.display_execution_progress(
            current_step=agent_number,
            total_steps=total_agents,
            role=role,
            task=task,
            status="running",
            layer=layer + 1,
            total_layers=total_layers,
        )

        # Parse context to extract previous outputs
        previous_outputs = self._parse_context(context, agent_id)

        conv_hist = conversation_history if conversation_history is not None else []

        # Set current agent for token tracking
        self.agent_executor.set_current_agent(agent_id, role)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.agent_executor.execute,
            role_obj,
            task,
            original_query,
            previous_outputs,
            conv_hist,
            0,
            3,
            self.process_query,
        )

        logger.info(f"✅ {agent_id} ({role}) completed")
        return result

    def _parse_context(self, context: str, agent_id: str) -> List[str]:
        """Parse context string to extract previous outputs."""
        previous_outputs = []
        if not context:
            logger.info(f"Agent {agent_id} has no context (first agent)")
            return previous_outputs

        logger.info(f"Agent {agent_id} received context: {context[:500]}...")

        parts = context.split("\n\n")
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.startswith("From "):
                if ":\n" in part:
                    _, output = part.split(":\n", 1)
                    if output:
                        previous_outputs.append(output)
                elif ":" in part:
                    _, rest = part.split(":", 1)
                    rest = rest.strip()
                    if rest:
                        previous_outputs.append(rest)
            elif not part.startswith(("Original question:", "===")):
                previous_outputs.append(part)

        logger.info(f"Total previous outputs extracted: {len(previous_outputs)}")
        return previous_outputs

    def _build_context(self) -> str:
        """Build conversation context from history."""
        if not self.conversation_history:
            return ""

        recent = self.conversation_history[-4:]
        lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines) if lines else ""

    def clear_memory(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("💾 Conversation memory cleared")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get conversation memory summary."""
        return {
            "message_count": len(self.conversation_history),
            "exchanges": len(self.conversation_history) // 2,
            "preview": self.conversation_history[-2:] if self.conversation_history else [],
        }

    def generate_execution_graph(self, result: Dict[str, Any], auto_open: bool = True) -> str:
        """Generate interactive HTML graph of last execution."""
        graph_path = self.visualizer.create_execution_graph(
            plan_description=result["plan"]["description"],
            agents=result.get(
                "agents_spec",
                [{"role": r, "task": "", "depends_on": []} for r in result["plan"]["agents"]],
            ),
            trace=result["trace"],
            execution_layers=result.get("execution_layers"),
        )

        if auto_open:
            import webbrowser

            webbrowser.open(f"file://{Path(graph_path).absolute()}")
            logger.info("🌐 Opened graph in browser")

        return graph_path

    def show_memory_visualization(self) -> None:
        """Display conversation memory visualization."""
        self.visualizer.show_memory_visualization(self.conversation_history)

    def _analyze_query_complexity(self, query: str) -> int:
        """Analyze query to determine appropriate max execution depth."""
        word_count = len(query.split())

        if word_count < 10:
            depth, level = 2, "Simple"
        elif word_count < 25:
            depth, level = 3, "Moderate"
        else:
            depth, level = 4, "Complex"

        logger.info(f"📊 Initial assessment: {level} ({word_count} words) → max_depth: {depth}")
        return depth
