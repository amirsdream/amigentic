"""Meta-coordinator for execution planning."""

import logging
import json
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel

from ..config import Config
from ..role_library import RoleLibrary
from ..agents.token_tracker import get_tracker
from .plan import ExecutionPlan
from .prompts import COORDINATOR_SYSTEM_PROMPT
from .validators import fix_synthesizer_dependencies, validate_plan_logic, fix_plan_logic

if TYPE_CHECKING:
    from ..services.rag import RAGService

logger = logging.getLogger(__name__)


class MetaCoordinator:
    """Meta-coordinator that plans and manages dynamic agent execution."""

    def __init__(
        self,
        config: Config,
        llm: BaseChatModel,
        rag_service: Optional["RAGService"] = None,
    ):
        """Initialize meta-coordinator.

        Args:
            config: Application configuration.
            llm: The configured language model to use.
            rag_service: Optional RAG service for active knowledge retrieval.
        """
        self.config = config
        self.llm = llm
        self.rag_service = rag_service
        self.role_library = RoleLibrary()
        self._warmed_up = False

    def warmup(self) -> None:
        """Warm up the LLM with a simple test call to reduce first-query latency."""
        if self._warmed_up:
            return

        try:
            logger.info("🔥 Warming up coordinator LLM...")
            self.llm.invoke(
                [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Hello"),
                ]
            )
            self._warmed_up = True
            logger.info("✓ Coordinator warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def create_execution_plan(
        self, query: str, conversation_history: str = "", depth: int = 0, max_depth: int = 3
    ) -> ExecutionPlan:
        """Create an execution plan for the query.

        Args:
            query: User's query.
            conversation_history: Recent conversation history for context.
            depth: Current nesting depth.
            max_depth: Maximum nesting depth allowed.

        Returns:
            Execution plan with agents to create and sequence.
        """
        if not self._warmed_up:
            self.warmup()

        logger.info(f"📋 Creating execution plan for: {query[:100]}...")

        # === ACTIVE RAG: Auto-inject relevant knowledge base context ===
        rag_context = None
        if self.rag_service:
            rag_context = self.rag_service.get_relevant_context_for_planning(
                query, k=3, min_score=0.5
            )
            if rag_context:
                logger.info("📚 RAG context injected into planning phase")

        # Build system prompt
        roles_str = ", ".join(self.role_library.list_roles())
        system_prompt = COORDINATOR_SYSTEM_PROMPT.format(roles=roles_str)

        # Build human message with optional RAG context
        human_content = query
        if rag_context:
            human_content = f"{rag_context}\n\n{query}"
        if conversation_history:
            human_content = f"CONVERSATION HISTORY:\n{conversation_history}\n\nCURRENT QUESTION:\n{human_content}"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]

        config: RunnableConfig = {
            "run_name": "meta_coordinator_planning",
            "metadata": {"query": query[:100]},
            "tags": ["coordinator", "planning", "meta_agent"],
        }  # type: ignore

        try:
            # Get response
            response = self._invoke_llm(messages, config)
            content = str(response.content) if response.content else ""

            logger.info(f"📝 Raw LLM response:\n{content}")

            # Parse JSON
            plan_data = self._parse_json_response(content)

            if not plan_data.get("agents"):
                raise ValueError("No agents in plan")

            # Limit agents
            agent_count = len(plan_data.get("agents", []))
            if agent_count > 12:
                logger.warning(f"⚠️ Plan has {agent_count} agents, limiting to 12")
                plan_data["agents"] = plan_data["agents"][:12]

            # Build and validate agents
            agents = self._build_agents(plan_data)

            if not agents:
                logger.error("✗ No valid agents in plan, using fallback")
                return self._create_fallback_plan(query)

            # Fix dependencies
            logger.info("📊 Dependencies BEFORE auto-fix:")
            for i, agent in enumerate(agents):
                logger.info(f"   Agent {i} ({agent['role']}): {agent.get('depends_on', [])}")

            agents = fix_synthesizer_dependencies(agents)

            logger.info("📊 Dependencies AFTER auto-fix:")
            for i, agent in enumerate(agents):
                logger.info(f"   Agent {i} ({agent['role']}): {agent.get('depends_on', [])}")

            # Validate logic
            if not validate_plan_logic(agents):
                logger.warning("⚠️ Plan has logical issues, attempting to fix...")
                agents = fix_plan_logic(agents)

            plan = ExecutionPlan(
                description=plan_data.get("description", "Dynamic execution plan"),
                agents=agents,
                depth=depth,
            )

            logger.info(f"✓ Created plan: {plan.description}")
            logger.info(f"✓ Agents: {[a['role'] for a in plan.agents]}")

            return plan

        except Exception as e:
            logger.error(f"✗ Failed to create plan: {e}")
            return self._create_fallback_plan(query)

    def _invoke_llm(self, messages: List, config: RunnableConfig):
        """Invoke LLM with appropriate settings."""
        llm_class_name = self.llm.__class__.__name__
        if "OpenAI" in llm_class_name:
            response = self.llm.invoke(
                messages, config=config, response_format={"type": "json_object"}
            )
        else:
            response = self.llm.invoke(messages, config=config)

        # Track planning tokens
        tracker = get_tracker()
        tracker.add_planning_usage(response)

        return response

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        cleaned = content.strip()
        if "```json" in cleaned or "```" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("⚠️ Response is not valid JSON, attempting extraction...")

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = cleaned[start:end]

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try repairs
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r"\}\s*\n\s*\{", "},\n    {", json_str)
                    json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)
                    return json.loads(json_str)

            raise ValueError("No JSON found in response")

    def _build_agents(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build agent list from plan data."""
        agents = []
        invalid_roles = []

        for agent_spec in plan_data.get("agents", []):
            role_name = agent_spec.get("role")
            task = agent_spec.get("task")

            if not role_name or not task:
                continue

            role_name_lower = role_name.lower()
            role = self.role_library.get_role(role_name_lower)

            if not role:
                invalid_roles.append(role_name)
                logger.warning(f"⚠️ Rejecting undefined role: {role_name}")
                continue

            depends_on = agent_spec.get("depends_on", [])
            if isinstance(depends_on, (int, str)):
                depends_on = [depends_on]
            try:
                depends_on = [int(d) if isinstance(d, str) else d for d in depends_on]
            except (ValueError, TypeError):
                depends_on = []

            agents.append(
                {
                    "role": role_name_lower,
                    "task": task,
                    "can_delegate": role.can_delegate,
                    "depends_on": depends_on,
                }
            )

        if invalid_roles:
            logger.error(f"✗ Invalid roles rejected: {invalid_roles}")

        return agents

    def _create_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create a simple fallback plan."""
        logger.warning("Using fallback plan")

        needs_web = any(
            word in query.lower()
            for word in ["current", "latest", "today", "news", "weather", "2024", "2025", "now"]
        )

        if needs_web:
            agents = [
                {
                    "role": "researcher",
                    "task": "Search for current information",
                    "can_delegate": False,
                    "depends_on": [],
                },
                {
                    "role": "synthesizer",
                    "task": "Create final answer",
                    "can_delegate": False,
                    "depends_on": [0],
                },
            ]
        else:
            agents = [
                {
                    "role": "analyzer",
                    "task": "Answer the question",
                    "can_delegate": False,
                    "depends_on": [],
                }
            ]

        return ExecutionPlan(description="Fallback plan", agents=agents, depth=0)
