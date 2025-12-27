"""Agent execution logic."""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..role_library import RoleLibrary, AgentRole as Role
from ..config import Config
from .token_tracker import get_tracker, TokenUsage

# Import metrics (optional)
try:
    from ..metrics import (
        PROMETHEUS_AVAILABLE,
        AGENT_EXECUTIONS_TOTAL,
        AGENT_EXECUTION_DURATION,
        AGENTS_IN_PROGRESS,
        TOOL_CALLS_TOTAL,
        TOOL_CALL_DURATION,
        record_error,
        record_llm_request,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    AGENT_EXECUTIONS_TOTAL = None
    AGENT_EXECUTION_DURATION = None
    AGENTS_IN_PROGRESS = None
    TOOL_CALLS_TOTAL = None
    TOOL_CALL_DURATION = None
    record_error = None
    record_llm_request = None

if TYPE_CHECKING:
    from ..tools.manager import ToolManager

logger = logging.getLogger(__name__)

# Load config for limits
_config = Config()


class AgentExecutor:
    """Handles execution of individual agents with MCP gateway integration."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        role_library: RoleLibrary,
        ui_display_limit: int = 200,
        tool_manager: Optional["ToolManager"] = None,
    ):
        """Initialize agent executor.

        Args:
            llm: Language model instance
            tools: Default available tools (fallback)
            role_library: Role definitions library
            ui_display_limit: Max characters for output display
            tool_manager: Optional tool manager for role-specific tools
        """
        self.llm = llm
        self.tools = tools  # Default/fallback tools
        self.role_library = role_library
        self.ui_display_limit = ui_display_limit
        self.tool_manager = tool_manager
        self._role_tools_cache: Dict[str, List[BaseTool]] = {}

        # Context limits from config
        self.context_limit = _config.agent_context_limit
        self.history_limit = _config.agent_history_limit

    def _record_agent_start(self) -> float:
        """Record agent execution start for metrics."""
        if PROMETHEUS_AVAILABLE and AGENTS_IN_PROGRESS:
            AGENTS_IN_PROGRESS.inc()
        return time.perf_counter()

    def _record_agent_end(self, agent_type: str, start_time: float, success: bool):
        """Record agent execution completion for metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        duration = time.perf_counter() - start_time
        if AGENTS_IN_PROGRESS:
            AGENTS_IN_PROGRESS.dec()
        if AGENT_EXECUTIONS_TOTAL:
            AGENT_EXECUTIONS_TOTAL.labels(
                agent_type=agent_type,
                status='success' if success else 'error'
            ).inc()
        if AGENT_EXECUTION_DURATION:
            AGENT_EXECUTION_DURATION.labels(agent_type=agent_type).observe(duration)

    def _record_tool_call(self, tool_name: str, duration: float, success: bool, error_type: Optional[str] = None):
        """Record tool call for metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        if TOOL_CALLS_TOTAL:
            TOOL_CALLS_TOTAL.labels(
                tool_name=tool_name,
                status='success' if success else 'error'
            ).inc()
        if TOOL_CALL_DURATION:
            TOOL_CALL_DURATION.labels(tool_name=tool_name).observe(duration)
        # Optionally record the error
        if not success and error_type and record_error:
            record_error(error_type, f'tool_{tool_name}')

    def _record_llm_request(self, duration: float, success: bool):
        """Record LLM request metrics."""
        if not PROMETHEUS_AVAILABLE or not record_llm_request:
            return
        # Get provider and model from config
        provider = _config.llm_provider
        model = _config.get_model_name()
        record_llm_request(provider, model, duration, success)

    def _invoke_llm(self, messages: list, config: RunnableConfig, llm: Any = None) -> Any:
        """Invoke LLM with metrics tracking.
        
        Args:
            messages: List of messages to send to the LLM
            config: RunnableConfig for the invocation
            llm: Optional LLM or LLM with tools to use (defaults to self.llm)
            
        Returns:
            LLM response
        """
        llm_to_use = llm or self.llm
        start_time = time.time()
        try:
            response = llm_to_use.invoke(messages, config=config)
            self._record_llm_request(time.time() - start_time, True)
            return response
        except Exception as e:
            self._record_llm_request(time.time() - start_time, False)
            if PROMETHEUS_AVAILABLE and record_error:
                record_error(type(e).__name__, 'llm')
            raise

    def _track_tokens(self, response: Any, agent_id: str = "", role: str = "") -> TokenUsage:
        """Track tokens from an LLM response.

        Args:
            response: LLM response object
            agent_id: Agent identifier
            role: Agent role name

        Returns:
            TokenUsage extracted from response
        """
        tracker = get_tracker()
        if agent_id and role:
            return tracker.add_agent_usage(agent_id, role, response)
        return tracker.extract_usage_from_response(response)

    async def pre_cache_role_tools(self) -> None:
        """Pre-cache tools for all known roles.

        Call this during async initialization to avoid sync/async issues later.
        """
        if not self.tool_manager:
            return

        for role_name in self.role_library.list_roles():
            try:
                tools = await self.get_tools_for_role(role_name)
                logger.debug(f"Pre-cached {len(tools)} tools for role '{role_name}'")
            except Exception as e:
                logger.warning(f"Failed to pre-cache tools for role '{role_name}': {e}")

    async def get_tools_for_role(self, role_name: str) -> List[BaseTool]:
        """Get tools appropriate for a specific role.

        Uses tool_manager for role-specific MCP tools if available,
        otherwise falls back to default tools.

        Args:
            role_name: Name of the agent role

        Returns:
            List of tools for the role
        """
        # Check cache first
        if role_name in self._role_tools_cache:
            return self._role_tools_cache[role_name]

        # Try to get role-specific tools from tool manager
        if self.tool_manager:
            try:
                role_tools = await self.tool_manager.get_tools_for_role(role_name)
                if role_tools:
                    self._role_tools_cache[role_name] = role_tools
                    logger.info(f"Role '{role_name}' using {len(role_tools)} role-specific tools")
                    return role_tools
            except Exception as e:
                logger.warning(f"Failed to get role-specific tools for {role_name}: {e}")

        # Fall back to default tools
        self._role_tools_cache[role_name] = self.tools
        return self.tools

    def get_tools_for_role_sync(self, role_name: str) -> List[BaseTool]:
        """Synchronous wrapper to get tools for a role.

        Args:
            role_name: Name of the agent role

        Returns:
            List of tools for the role
        """
        # Check cache first (fastest path)
        if role_name in self._role_tools_cache:
            return self._role_tools_cache[role_name]

        # Try to run async function from sync context
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're inside a running loop - can't use run_until_complete
                # Use nest_asyncio or run in executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.get_tools_for_role(role_name))
                    )
                    return future.result(timeout=30)
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.get_tools_for_role(role_name))
        except Exception as e:
            logger.warning(f"Failed to get role tools synchronously: {e}, using defaults")
            return self.tools

    def execute(
        self,
        role: Role,
        task: str,
        original_query: str,
        previous_outputs: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        depth: int = 0,
        max_depth: int = 3,
        process_query_callback: Optional[Callable] = None,
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Execute a single agent.

        Args:
            role: Agent role definition
            task: Specific task for this agent
            original_query: Original user query
            previous_outputs: Outputs from previous agents
            conversation_history: Conversation history
            depth: Current execution depth
            max_depth: Maximum execution depth
            process_query_callback: Callback for recursive delegation
            agent_id: Agent identifier for token tracking

        Returns:
            Dict with 'content' (text output) and 'tool_calls' (list of tools used)
        """
        # Start metrics tracking
        start_time = self._record_agent_start()
        success = False
        
        try:
            result = self._execute_internal(
                role, task, original_query, previous_outputs,
                conversation_history, depth, max_depth,
                process_query_callback, agent_id
            )
            success = True
            return result
        except Exception as e:
            if PROMETHEUS_AVAILABLE and record_error:
                record_error(type(e).__name__, f'agent_{role.name}')
            raise
        finally:
            self._record_agent_end(role.name, start_time, success)

    def _execute_internal(
        self,
        role: Role,
        task: str,
        original_query: str,
        previous_outputs: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        depth: int = 0,
        max_depth: int = 3,
        process_query_callback: Optional[Callable] = None,
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Internal execute implementation."""
        # Build context
        context = self._build_context(original_query, previous_outputs, conversation_history)

        # Build messages
        output_limit_instruction = (
            f"\n\nIMPORTANT: Keep your response concise and under "
            f"{self.ui_display_limit} characters. Be direct and focused."
        )
        system_msg = SystemMessage(content=role.system_prompt + output_limit_instruction)
        task_msg = HumanMessage(content=f"{context}\n\nYour task: {task}")

        logger.info(f"Task message content (first 500 chars): {task_msg.content[:500]}...")

        # Check if agent can and should delegate
        if role.can_delegate and depth < max_depth:
            task_msg = self._build_delegation_prompt(context, task)

        # Add metadata for Phoenix tracing
        config = self._create_run_config(role, task)

        # Execute with or without tools
        if role.needs_tools:
            return self._execute_with_tools(role, system_msg, task_msg, config, agent_id)
        else:
            return self._execute_without_tools(
                role, system_msg, task_msg, config, task, depth, max_depth, process_query_callback, agent_id
            )

    def _build_context(
        self,
        original_query: str,
        previous_outputs: List[str],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build context string from query and previous outputs."""
        context_parts = [f"Original question: {original_query}"]

        if conversation_history:
            context_parts.append("\n=== Previous Agent Conversation Steps ===")
            for i, step in enumerate(conversation_history[-3:], 1):
                context_parts.append(
                    f"\nStep {i} - {step.get('role', 'unknown')} ({step.get('agent_id', '')}):"
                )
                context_parts.append(f"  Task: {step.get('task', '')[:100]}")
                step_output = step.get("output", "")[: self.history_limit]
                context_parts.append(f"  Output: {step_output}...")

        if previous_outputs:
            logger.info(f"Agent has {len(previous_outputs)} previous outputs to incorporate")
            context_parts.append("\n=== Outputs from Previous Agents ===")
            for i, output in enumerate(previous_outputs, 1):
                output_display = (
                    output[: self.context_limit] + "..."
                    if len(output) > self.context_limit
                    else output
                )
                context_parts.append(f"\nAgent {i} output:\n{output_display}")
        else:
            context_parts.append("\n=== You are the first agent ===")
            context_parts.append("No prior agent outputs available yet.")

        return "\n".join(context_parts)

    def _build_delegation_prompt(self, context: str, task: str) -> HumanMessage:
        """Build delegation prompt for agents that can delegate."""
        delegation_prompt = f"""
{context}

Your task: {task}

You have the ability to delegate work to specialized sub-agents. 
If this task would benefit from delegation, respond with JSON:
{{
  "needs_delegation": true,
  "subtasks": [
    {{"role": "role_name", "task": "specific task"}},
    ...
  ]
}}

Otherwise, complete the task directly and respond with your normal output (not JSON).

IMPORTANT: If completing directly, keep your response under {self.ui_display_limit} characters.
"""
        return HumanMessage(content=delegation_prompt)

    def _create_run_config(self, role: Role, task: str) -> RunnableConfig:
        """Create run configuration for tracing."""
        metadata = {"agent_role": role.name, "agent_task": task, "has_tools": role.needs_tools}
        return {
            "run_name": f"{role.name}_agent",
            "metadata": metadata,
            "tags": [role.name, "meta_agent"],
        }  # type: ignore

    def _execute_with_tools(
        self, role: Role, system_msg: SystemMessage, task_msg: HumanMessage, config: RunnableConfig, agent_id: str = ""
    ) -> Dict[str, Any]:
        """Execute agent with tool access."""
        # Get role-specific tools
        role_tools = self.get_tools_for_role_sync(role.name)

        if not role_tools:
            logger.error(f"⚠️ {role.name} needs tools but no tools are available!")
            response = self._invoke_llm([system_msg, task_msg], config)
            self._track_tokens(response, agent_id, role.name)  # Track token usage
            return {"content": str(response.content), "tool_calls": []}

        logger.info(f"🔧 {role.name} has access to {len(role_tools)} tools")

        # Special handling for researcher role
        if role.name == "researcher":
            return self._execute_researcher(role, system_msg, task_msg, config, role_tools, agent_id)

        # Standard tool calling for other roles
        return self._execute_standard_tool_calling(role, system_msg, task_msg, config, role_tools, agent_id)

    def _execute_researcher(
        self,
        role: Role,
        system_msg: SystemMessage,
        task_msg: HumanMessage,
        config: RunnableConfig,
        role_tools: List[BaseTool],
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Execute researcher agent with web search."""
        logger.info(f"🔍 {role.name} will perform web search")

        # Get search queries from LLM
        search_prompt = SystemMessage(
            content="""You are a research specialist planning a web search.
Based on the task, provide 1-3 search queries (one per line) that would give the best results.
Just output the search queries, nothing else. No explanations, no numbering.
IMPORTANT: Keep search queries short and focused."""
        )

        search_response = self._invoke_llm([search_prompt, task_msg], config)
        self._track_tokens(search_response, agent_id, role.name)  # Track token usage
        search_queries = str(search_response.content).strip().split("\n")
        search_queries = [q.strip() for q in search_queries if q.strip()][:3]

        logger.info(f"   └─ Search queries: {search_queries}")

        # Find search tool - prefer MCP websearch if available
        search_tool = self._find_search_tool(role_tools)
        if not search_tool:
            logger.error("   └─ No search tool found!")
            response = self._invoke_llm([system_msg, task_msg], config)
            self._track_tokens(response, agent_id, role.name)  # Track token usage
            return {"content": str(response.content), "tool_calls": []}

        # Execute searches
        tool_results = self._execute_searches(search_tool, search_queries)

        # Generate response with search results
        if tool_results:
            tool_context = "\n\n".join(
                [f"Search result {i+1}:\n{r}" for i, r in enumerate(tool_results)]
            )
            logger.info(f"📥 {role.name} processing {len(tool_results)} search result(s)")

            final_response = self._invoke_llm(
                [
                    system_msg,
                    task_msg,
                    AIMessage(
                        content=f"Based on these search results:\n\n{tool_context}\n\n"
                        f"Provide a comprehensive research summary. Keep it under "
                        f"{self.ui_display_limit} characters - be concise and focused on key findings."
                    ),
                ],
                config,
            )
            self._track_tokens(final_response, agent_id, role.name)  # Track token usage

            return {
                "content": str(final_response.content),
                "tool_calls": [
                    {"name": search_tool.name, "args": {"query": q}} for q in search_queries
                ],
            }

        # No results - fallback
        response = self._invoke_llm([system_msg, task_msg], config)
        self._track_tokens(response, agent_id, role.name)  # Track token usage
        return {"content": str(response.content), "tool_calls": []}

    def _find_search_tool(self, role_tools: List[BaseTool]) -> Optional[BaseTool]:
        """Find search tool - prefers MCP websearch over DuckDuckGo.

        Args:
            role_tools: List of available tools for the role

        Returns:
            Search tool if found
        """
        # First try MCP websearch
        for tool in role_tools:
            if "mcp_websearch_search" in tool.name:
                logger.info(f"   └─ Using MCP websearch tool: {tool.name}")
                return tool

        # Fall back to DuckDuckGo
        for tool in role_tools:
            if tool.name in ["duckduckgo_search", "ddg-search"]:
                logger.info(f"   └─ Using DuckDuckGo search tool: {tool.name}")
                return tool

        return None

    def _execute_searches(self, search_tool: BaseTool, queries: List[str]) -> List[str]:
        """Execute search queries and return results."""
        results = []
        for query in queries:
            query = query.lstrip("0123456789.-) ").strip()
            if not query:
                continue
            start_time = time.time()
            try:
                logger.info(f"   └─ Searching: {query}")
                result = search_tool.invoke({"query": query})
                duration = time.time() - start_time
                success = result is not None
                self._record_tool_call(search_tool.name, duration, success)
                if result:
                    results.append(result)
                    logger.info(f"🔍 SEARCH RESULT for '{query}': {result[:200]}...")
            except Exception as e:
                duration = time.time() - start_time
                self._record_tool_call(search_tool.name, duration, False, type(e).__name__)
                logger.error(f"   └─ Search error for '{query}': {e}")
                results.append(f"Search failed for '{query}': {e}")
        return results

    def _execute_standard_tool_calling(
        self,
        role: Role,
        system_msg: SystemMessage,
        task_msg: HumanMessage,
        config: RunnableConfig,
        role_tools: List[BaseTool],
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Execute agent with standard tool calling."""
        llm_with_tools = self.llm.bind_tools(role_tools)
        logger.info(
            f"🔧 {role.name} bound with {len(role_tools)} tools: {[t.name for t in role_tools[:5]]}..."
        )

        response = self._invoke_llm([system_msg, task_msg], config, llm=llm_with_tools)
        self._track_tokens(response, agent_id, role.name)  # Track token usage

        if hasattr(response, "tool_calls") and response.tool_calls:
            return self._process_tool_calls(
                role, system_msg, task_msg, response, config, role_tools, agent_id
            )

        return {"content": str(response.content), "tool_calls": []}

    def _process_tool_calls(
        self,
        role: Role,
        system_msg: SystemMessage,
        task_msg: HumanMessage,
        response,
        config: RunnableConfig,
        role_tools: List[BaseTool],
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Process tool calls from LLM response."""
        logger.info(f"🔍 {role.name} is calling {len(response.tool_calls)} tool(s)")

        recorded_calls = []
        tool_results = []

        for tool_call in response.tool_calls:
            tool_name, tool_args = self._parse_tool_call(tool_call)
            recorded_calls.append({"name": tool_name, "args": tool_args})

            logger.info(f"   └─ Tool: {tool_name}, Args: {tool_args}")

            result = self._execute_tool(tool_name, tool_args, role_tools)
            if result:
                tool_results.append(result)

        if tool_results:
            tool_context = "\n\n".join([f"Result {i+1}:\n{r}" for i, r in enumerate(tool_results)])
            final_response = self._invoke_llm(
                [
                    system_msg,
                    task_msg,
                    AIMessage(
                        content=f"Based on these results:\n\n{tool_context}\n\nProvide a comprehensive answer."
                    ),
                ],
                config,
            )
            self._track_tokens(final_response, agent_id, role.name)  # Track token usage
            return {"content": str(final_response.content), "tool_calls": recorded_calls}

        return {"content": str(response.content), "tool_calls": recorded_calls}

    def _parse_tool_call(self, tool_call) -> tuple[str, dict]:
        """Parse tool call into name and arguments."""
        if isinstance(tool_call, dict):
            return tool_call.get("name", "unknown"), tool_call.get("args", {})
        return (
            getattr(tool_call, "name", None) or getattr(tool_call, "type", "unknown"),
            getattr(tool_call, "args", None) or getattr(tool_call, "arguments", {}),
        )

    def _execute_tool(
        self, tool_name: str, tool_args: dict, role_tools: List[BaseTool]
    ) -> Optional[str]:
        """Execute a single tool by name from role-specific tools."""
        for tool in role_tools:
            if tool.name == tool_name:
                start_time = time.time()
                try:
                    logger.info(f"   └─ Executing {tool_name}...")
                    result = tool.invoke(tool_args)
                    duration = time.time() - start_time
                    self._record_tool_call(tool_name, duration, True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_tool_call(tool_name, duration, False, type(e).__name__)
                    logger.error(f"   └─ Tool error: {e}")
                    return f"Error executing {tool_name}: {e}"

        logger.warning(f"   └─ Tool '{tool_name}' not found in role tools")
        return None

    def _execute_without_tools(
        self,
        role: Role,
        system_msg: SystemMessage,
        task_msg: HumanMessage,
        config: RunnableConfig,
        task: str,
        depth: int,
        max_depth: int,
        process_query_callback: Optional[Callable],
        agent_id: str = "",
    ) -> Dict[str, Any]:
        """Execute agent without tools."""
        response = self._invoke_llm([system_msg, task_msg], config)
        self._track_tokens(response, agent_id, role.name)  # Track token usage
        response_content = str(response.content)

        # Check for delegation
        if role.can_delegate and depth < max_depth and process_query_callback:
            delegation_result = self._handle_delegation(
                response_content, task, system_msg, config, depth, max_depth, process_query_callback, agent_id, role.name
            )
            if delegation_result:
                return delegation_result

        return {"content": response_content, "tool_calls": []}

    def _handle_delegation(
        self,
        response_content: str,
        task: str,
        system_msg: SystemMessage,
        config: RunnableConfig,
        depth: int,
        max_depth: int,
        process_query_callback: Callable,
        agent_id: str = "",
        role_name: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Handle delegation requests from agents."""
        try:
            delegation_data = json.loads(response_content)
            if not delegation_data.get("needs_delegation") or not delegation_data.get("subtasks"):
                return None

            logger.info(f"🔀 Delegating to {len(delegation_data['subtasks'])} sub-agents")

            sub_results = []
            for subtask_spec in delegation_data["subtasks"]:
                sub_role_name = subtask_spec.get("role")
                sub_task = subtask_spec.get("task")

                if not sub_role_name or not sub_task:
                    continue

                logger.info(f"  └─ Delegating to {sub_role_name}: {sub_task[:60]}...")
                sub_result = process_query_callback(sub_task, depth=depth + 1, max_depth=max_depth)
                sub_results.append(f"{sub_role_name}: {sub_result['final_answer']}")

            if sub_results:
                synthesis_msg = HumanMessage(
                    content=f"""Original task: {task}

Sub-agent results:
{chr(10).join([f"{i+1}. {r}" for i, r in enumerate(sub_results)])}

Combine these results to complete your original task."""
                )

                final_response = self._invoke_llm([system_msg, synthesis_msg], config)
                self._track_tokens(final_response, agent_id, role_name)  # Track token usage
                return {"content": str(final_response.content), "tool_calls": []}

        except json.JSONDecodeError:
            pass

        return None
