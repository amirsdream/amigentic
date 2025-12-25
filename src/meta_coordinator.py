"""Meta-coordinator that dynamically decides which agents to create and how to sequence them."""

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel

from .config import Config
from .role_library import RoleLibrary

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Plan for executing a task with dynamic agents."""
    description: str
    agents: List[Dict[str, Any]]  # List of {role, task, can_delegate, depends_on}
    depth: int = 0  # Nesting level (0 = root)
    
    def get_dependency_graph(self) -> Dict[int, List[int]]:
        """Build dependency graph from agent specifications.
        
        Returns:
            Dict mapping agent index to list of dependency indices.
        """
        graph = {}
        for i, agent in enumerate(self.agents):
            depends_on = agent.get('depends_on', [])
            # Convert to list if single value
            if isinstance(depends_on, int):
                depends_on = [depends_on]
            elif isinstance(depends_on, str):
                # Handle string that might be a single number
                try:
                    depends_on = [int(depends_on)]
                except ValueError:
                    depends_on = []
            # Convert all elements to integers and filter valid dependencies
            try:
                depends_on_ints = [int(d) for d in depends_on]
                graph[i] = [d for d in depends_on_ints if 0 <= d < len(self.agents) and d != i]
            except (ValueError, TypeError):
                logger.warning(f"Invalid depends_on for agent {i}: {depends_on}")
                graph[i] = []
        return graph
    
    def get_execution_layers(self) -> List[List[int]]:
        """Get agents grouped by execution layer (topological sort).
        
        Returns:
            List of layers, where each layer contains agent indices that can run in parallel.
        """
        graph = self.get_dependency_graph()
        n = len(self.agents)
        
        # Calculate in-degrees
        in_degree = {i: 0 for i in range(n)}
        for deps in graph.values():
            for dep in deps:
                in_degree[dep] = in_degree.get(dep, 0)
        
        for i in range(n):
            for dep in graph.get(i, []):
                in_degree[i] += 1
        
        # Topological sort with layers
        layers = []
        remaining = set(range(n))
        
        while remaining:
            # Find all nodes with in-degree 0
            current_layer = [i for i in remaining if in_degree[i] == 0]
            
            if not current_layer:
                # Cycle detected or invalid dependencies - fallback to sequential
                logger.warning("Cycle detected in dependencies, falling back to sequential execution")
                return [[i] for i in sorted(remaining)]
            
            layers.append(current_layer)
            
            # Remove current layer and update in-degrees
            for node in current_layer:
                remaining.remove(node)
                for i in remaining:
                    if node in graph.get(i, []):
                        in_degree[i] -= 1
        
        return layers
    
    
class MetaCoordinator:
    """Meta-coordinator that plans and manages dynamic agent execution."""
    
    def __init__(self, config: Config, llm: BaseChatModel):
        """Initialize meta-coordinator.
        
        Args:
            config: Application configuration.
            llm: The configured language model to use.
        """
        self.config = config
        self.llm = llm
        self.role_library = RoleLibrary()
    
    def create_execution_plan(self, query: str, conversation_history: str = "", depth: int = 0, max_depth: int = 3) -> ExecutionPlan:
        """Create an execution plan for the query.
        
        Args:
            query: User's query.
            conversation_history: Recent conversation history for context.
            depth: Current nesting depth.
            max_depth: Maximum nesting depth allowed.
            
        Returns:
            Execution plan with agents to create and sequence.
        """
        logger.info(f"📋 Creating execution plan for: {query[:100]}...")
        
        # Build the planning prompt - keep it concise for speed
        system_prompt = f"""You are a meta-coordinator creating execution plans. Output ONLY valid JSON.

Available roles: {', '.join(self.role_library.list_roles())}

ROLE SELECTION RULES:
- "researcher": ONLY for web search - current info, facts, news
- "analyzer": Analysis, explanations, comparisons, breakdowns
- "writer": Articles, stories, summaries, documentation
- "coder": ONLY for programming/code tasks
- "planner": Step-by-step plans, strategies
- "critic": Review and improve existing content
- "synthesizer": REQUIRED as final agent when you have 2+ agents

JSON format:
{{{{
  "description": "brief plan",
  "agents": [
    {{"role": "ROLE_NAME", "task": "specific task", "depends_on": []}}
  ]
}}}}

Dependencies:
- "depends_on": [] → runs immediately
- "depends_on": [0] → waits for agent 0
- "depends_on": [0, 1] → waits for agents 0 and 1

Examples:
"Explain X" → {{"description": "Explanation", "agents": [{{"role": "analyzer", "task": "Explain X clearly", "depends_on": []}}]}}

"Compare X vs Y" → {{"description": "Comparison", "agents": [
  {{"role": "researcher", "task": "Research X", "depends_on": []}},
  {{"role": "researcher", "task": "Research Y", "depends_on": []}},
  {{"role": "synthesizer", "task": "Compare X and Y based on research", "depends_on": [0, 1]}}
]}}

Output ONLY JSON - no explanations, no markdown blocks."""

        # Build human message with conversation context if available
        human_content = query
        if conversation_history:
            human_content = f"CONVERSATION HISTORY:\n{conversation_history}\n\nCURRENT QUESTION:\n{query}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        # Add config for Phoenix tracing
        config: RunnableConfig = {
            "run_name": "meta_coordinator_planning",
            "metadata": {"query": query[:100]},
            "tags": ["coordinator", "planning", "meta_agent"]
        }  # type: ignore
        
        # Get the plan - use appropriate method based on LLM type
        content = ""
        try:
            # For OpenAI, we can use structured output mode
            llm_class_name = self.llm.__class__.__name__
            if "OpenAI" in llm_class_name:
                # OpenAI supports response_format for JSON
                response = self.llm.invoke(
                    messages, 
                    config=config,
                    response_format={"type": "json_object"}  # type: ignore
                )
            else:
                # For Ollama and Claude, use regular invoke
                response = self.llm.invoke(messages, config=config)
            
            content = str(response.content) if response.content else ""
            
            # Log raw response for debugging
            logger.debug(f"📝 Raw LLM response (first 500 chars): {content[:500]}...")
            
            # Parse JSON
            try:
                plan_data = json.loads(content)
                logger.info("✓ Successfully parsed JSON response")
                logger.debug(f"📋 Parsed plan: {json.dumps(plan_data, indent=2)}")
            except json.JSONDecodeError:
                # Extract JSON from text
                logger.warning("⚠️ Response is not pure JSON, attempting to extract...")
                
                # Try to remove markdown code blocks first
                cleaned_content = content
                if "```json" in content or "```" in content:
                    logger.info("Found markdown code blocks, removing...")
                    cleaned_content = content.replace("```json", "").replace("```", "")
                
                start = cleaned_content.find('{')
                end = cleaned_content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = cleaned_content[start:end]
                    logger.info(f"📄 Extracted JSON (first 200 chars): {json_str[:200]}...")
                    
                    # Try to parse
                    try:
                        plan_data = json.loads(json_str)
                        logger.info("✓ Successfully extracted and parsed JSON")
                    except json.JSONDecodeError as e:
                        # Try common fixes
                        logger.warning(f"⚠️ JSON parse error: {e}, attempting repairs...")
                        
                        # Fix 1: Replace single quotes with double quotes
                        json_str = json_str.replace("'", '"')
                        
                        # Fix 2: Add missing commas between objects (common LLM error)
                        import re
                        # Pattern: }\n    { (object end, newline, object start)
                        json_str = re.sub(r'\}\s*\n\s*\{', '},\n    {', json_str)
                        # Pattern: ]\n  } (array end, newline, object end)
                        json_str = re.sub(r'\]\s*\n\s*\}', ']\n  }', json_str)
                        
                        # Fix 3: Remove trailing commas before ] or }
                        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                        
                        try:
                            plan_data = json.loads(json_str)
                            logger.info("✓ Successfully repaired and parsed JSON")
                        except json.JSONDecodeError as e2:
                            logger.error(f"✗ JSON repair failed: {e2}")
                            logger.error(f"✗ Attempted JSON (first 1000 chars): {json_str[:1000]}...")
                            raise ValueError(f"Invalid JSON after repair attempts: {e2}")
                else:
                    logger.error(f"✗ No JSON found in response (first 500 chars): {content[:500]}...")
                    raise ValueError("No JSON in response")
            
            # Validate plan
            if not plan_data.get("agents"):
                raise ValueError("No agents in plan")
            
            # GUARDRAIL: Limit number of agents based on complexity
            agent_count = len(plan_data.get("agents", []))
            max_agents = 12  # Soft limit, trust the model but prevent runaway
            
            if agent_count > max_agents:
                logger.warning(f"⚠️ Plan has {agent_count} agents, limiting to {max_agents} for safety")
                plan_data["agents"] = plan_data["agents"][:max_agents]
            
            # Create execution plan with delegation info AND VALIDATION
            agents = []
            invalid_roles = []
            for agent_spec in plan_data.get("agents", []):
                role_name = agent_spec.get("role")
                task = agent_spec.get("task")
                
                if role_name and task:
                    # Normalize role name to lowercase
                    role_name_lower = role_name.lower()
                    
                    # Get role definition to check delegation capability
                    role = self.role_library.get_role(role_name_lower)
                    
                    if not role:
                        # Role doesn't exist - reject it
                        invalid_roles.append(role_name)
                        logger.warning(f"⚠️ Rejecting undefined role: {role_name}")
                        continue
                    
                    can_delegate = role.can_delegate
                    depends_on = agent_spec.get("depends_on", [])
                    
                    # Normalize depends_on to list of integers
                    if isinstance(depends_on, (int, str)):
                        depends_on = [depends_on]
                    # Convert all to integers
                    try:
                        depends_on = [int(d) if isinstance(d, str) else d for d in depends_on]
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Invalid depends_on format: {depends_on}, using empty list")
                        depends_on = []
                    
                    agents.append({
                        "role": role_name_lower,  # Use normalized name
                        "task": task,
                        "can_delegate": can_delegate,
                        "depends_on": depends_on  # Now guaranteed to be list of ints
                    })
            
            # If invalid roles were found, log error
            if invalid_roles:
                logger.error(f"✗ Invalid roles rejected: {invalid_roles}")
                logger.error(f"✗ Valid roles are: {self.role_library.list_roles()}")
                logger.error(f"📄 LLM Response that caused rejection: {content[:1000]}")
            
            # If no valid agents, use fallback
            if not agents:
                logger.error("✗ No valid agents in plan, using fallback")
                logger.error(f"📋 Plan data received: {plan_data}")
                logger.error(f"📄 Full LLM response: {content}")
                return self._create_fallback_plan(query)
            
            logger.info(f"📝 Created {len(agents)} agents from LLM plan")
            logger.info("📊 Dependencies BEFORE auto-fix:")
            for i, agent in enumerate(agents):
                deps = agent.get('depends_on', [])
                logger.info(f"   Agent {i} ({agent['role']}): {deps}")
            
            # CRITICAL: Auto-fix synthesizer dependencies
            agents = self._fix_synthesizer_dependencies(agents)
            
            logger.info("📊 Dependencies AFTER auto-fix:")
            for i, agent in enumerate(agents):
                deps = agent.get('depends_on', [])
                logger.info(f"   Agent {i} ({agent['role']}): {deps}")
            
            # Validate logical flow
            if not self._validate_plan_logic(agents):
                logger.warning("⚠️ Plan has logical issues, attempting to fix...")
                agents = self._fix_plan_logic(agents)
            
            plan = ExecutionPlan(
                description=plan_data.get("description", "Dynamic execution plan"),
                agents=agents,
                depth=depth
            )
            
            logger.info(f"✓ Created plan: {plan.description}")
            logger.info(f"✓ Agents: {[a['role'] for a in plan.agents]}")
            
            # DEBUG: Log dependencies for each agent
            logger.info("📊 Agent Dependencies:")
            for i, agent in enumerate(plan.agents):
                deps = agent.get('depends_on', [])
                if deps:
                    dep_roles = [plan.agents[d]['role'] for d in deps if d < len(plan.agents)]
                    logger.info(f"   Agent {i} ({agent['role']}): depends on {deps} → {dep_roles}")
                else:
                    logger.info(f"   Agent {i} ({agent['role']}): NO DEPENDENCIES (runs immediately)")
            
            return plan
            
        except Exception as e:
            logger.error(f"✗ Failed to create plan: {e}")
            logger.error(f"✗ Response: {content[:500]}")
            
            # Fallback plan
            return self._create_fallback_plan(query)
    
    def _fix_synthesizer_dependencies(self, agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure synthesizers depend on all content-producing agents.
        
        Args:
            agents: List of agent specifications.
            
        Returns:
            Fixed agent list.
        """
        for i, agent in enumerate(agents):
            if agent['role'] in ['synthesizer', 'writer'] and i > 0:
                # Synthesizer/writer should depend on all previous content producers
                depends_on = agent.get('depends_on', [])
                
                # Convert to list if single value
                if isinstance(depends_on, (int, str)):
                    depends_on = [depends_on]
                
                # Convert string deps to ints
                try:
                    depends_on = [int(d) if isinstance(d, str) else d for d in depends_on]
                except (ValueError, TypeError):
                    depends_on = []
                
                # If synthesizer has no dependencies, make it depend on all previous agents
                if not depends_on:
                    # Depend on all agents except other synthesizers/writers
                    content_producers = [
                        j for j in range(i) 
                        if agents[j]['role'] not in ['synthesizer', 'writer', 'critic']
                    ]
                    if content_producers:
                        agent['depends_on'] = content_producers
                        logger.info(f"🔧 Auto-fixed {agent['role']} {i}: now depends on {content_producers}")
                else:
                    # Update with int-converted deps
                    agent['depends_on'] = depends_on
        
        return agents
    
    def _validate_plan_logic(self, agents: List[Dict[str, Any]]) -> bool:
        """Validate that the plan has logical dependencies.
        
        Args:
            agents: List of agent specifications.
            
        Returns:
            True if plan is logically valid.
        """
        # Check 1: Synthesizers should not be in first layer
        for i, agent in enumerate(agents):
            if agent['role'] == 'synthesizer' and i < len(agents) - 1:
                depends_on = agent.get('depends_on', [])
                if not depends_on:
                    logger.warning(f"⚠️ Synthesizer at position {i} has no dependencies")
                    return False
        
        # Check 2: Detect potential redundancy (warning only, don't block)
        seen_roles = {}
        for i, agent in enumerate(agents):
            role = agent.get('role', '')
            if role in seen_roles:
                seen_roles[role].append(i)
            else:
                seen_roles[role] = [i]
        
        # Warn about multiple agents with same role
        for role, indices in seen_roles.items():
            if len(indices) > 2 and role != 'researcher':  # Multiple researchers can be valid
                logger.info(f"ℹ️  Multiple {role} agents: {indices} - ensure tasks are distinct")
        
        # Check 3: Basic dependency validation
        for i, agent in enumerate(agents):
            depends_on = agent.get('depends_on', [])
            # Convert to list if single value
            if isinstance(depends_on, (int, str)):
                depends_on = [depends_on]
            
            for dep in depends_on:
                # Convert to int if string
                try:
                    dep_int = int(dep) if isinstance(dep, str) else dep
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Invalid dependency value: {dep}")
                    continue
                    
                if dep_int == i:
                    logger.warning(f"⚠️ Agent {i} depends on itself - fixing")
                    return False
                # Check for forward dependencies
                if dep_int >= i:
                    logger.warning(f"⚠️ Agent {i} depends on future agent {dep_int} - fixing")
                    return False
        
        return True
    
    def _fix_plan_logic(self, agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix logical issues in the plan.
        
        Args:
            agents: List of agent specifications.
            
        Returns:
            Fixed agent list.
        """
        # Move synthesizers to the end if they're in the middle with no deps
        fixed_agents = []
        synthesizers = []
        
        for agent in agents:
            if agent['role'] == 'synthesizer' and not agent.get('depends_on'):
                synthesizers.append(agent)
            else:
                fixed_agents.append(agent)
        
        # Add synthesizers at the end
        for synth in synthesizers:
            # Make them depend on all previous agents
            synth['depends_on'] = list(range(len(fixed_agents)))
            fixed_agents.append(synth)
        
        return fixed_agents
    
    def _create_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create a simple fallback plan.
        
        Args:
            query: User's query.
            
        Returns:
            Simple execution plan.
        """
        logger.warning("Using fallback plan")
        
        # Check if needs web search
        needs_web = any(word in query.lower() for word in [
            'current', 'latest', 'today', 'news', 'weather', '2024', '2025', 'now'
        ])
        
        if needs_web:
            agents = [
                {"role": "researcher", "task": "Search for current information", "can_delegate": False},
                {"role": "synthesizer", "task": "Create final answer", "can_delegate": False}
            ]
        else:
            agents = [
                {"role": "analyzer", "task": "Answer the question", "can_delegate": False}
            ]
        
        return ExecutionPlan(
            description="Fallback plan",
            agents=agents,
            depth=0
        )
