"""Role library - defines available agent roles without hardcoding agents."""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentRole:
    """Definition of an agent role."""
    name: str
    description: str
    capabilities: List[str]
    system_prompt: str
    needs_tools: bool = False
    can_delegate: bool = False  # Can this role create sub-agents?


class RoleLibrary:
    """Library of available agent roles."""
    
    def __init__(self):
        """Initialize role library."""
        self.roles = self._load_roles()
        logger.info(f"Loaded {len(self.roles)} roles")
    
    def _load_roles(self) -> dict:
        """Load available roles."""
        return {
            "researcher": AgentRole(
                name="researcher",
                description="Conducts research and gathers information from the web",
                capabilities=["web_search", "fact_finding", "information_gathering"],
                system_prompt="You are a research specialist. Use web search to find accurate, current information. Cite your sources.",
                needs_tools=True
            ),
            "analyzer": AgentRole(
                name="analyzer",
                description="Analyzes data, compares options, provides insights",
                capabilities=["analysis", "comparison", "reasoning"],
                system_prompt="You are an analysis specialist. Break down complex information, identify patterns, and provide clear insights.",
                needs_tools=False
            ),
            "planner": AgentRole(
                name="planner",
                description="Creates plans, strategies, and step-by-step solutions",
                capabilities=["planning", "strategy", "organization"],
                system_prompt="You are a planning specialist. Create detailed, actionable plans with clear steps and considerations.",
                needs_tools=False,
                can_delegate=True
            ),
            "writer": AgentRole(
                name="writer",
                description="Writes content, summaries, and documentation",
                capabilities=["writing", "summarization", "documentation"],
                system_prompt="You are a writing specialist. Create clear, well-structured content tailored to the audience.",
                needs_tools=False
            ),
            "coder": AgentRole(
                name="coder",
                description="Writes code and solves programming problems (ONLY for coding tasks)",
                capabilities=["coding", "debugging", "code_review"],
                system_prompt="You are a coding specialist. Write clean, well-documented code and solve programming problems.",
                needs_tools=False
            ),
            "critic": AgentRole(
                name="critic",
                description="Reviews work, finds issues, suggests improvements",
                capabilities=["review", "quality_check", "validation"],
                system_prompt="You are a quality reviewer. Identify issues, gaps, and areas for improvement. Be constructive.",
                needs_tools=False
            ),
            "synthesizer": AgentRole(
                name="synthesizer",
                description="Combines multiple inputs into coherent final output",
                capabilities=["synthesis", "integration", "finalization"],
                system_prompt="You are a synthesis specialist. Combine all inputs into a comprehensive, well-structured final answer.",
                needs_tools=False
            ),
            "coordinator": AgentRole(
                name="coordinator",
                description="Manages complex multi-step workflows by delegating to specialized agents",
                capabilities=["task_decomposition", "delegation", "workflow_management"],
                system_prompt="You are a workflow coordinator. For complex tasks, break them into sub-tasks and delegate to specialized agents.",
                needs_tools=False,
                can_delegate=True
            )
        }
    
    def get_role(self, role_name: str) -> Optional[AgentRole]:
        """Get a role by name."""
        return self.roles.get(role_name)
    
    def list_roles(self) -> List[str]:
        """List all available role names."""
        return list(self.roles.keys())
    
    def describe_roles(self) -> str:
        """Get a description of all roles for the coordinator."""
        lines = ["Available Agent Roles:"]
        for name, role in self.roles.items():
            tools = " [CAN USE WEB SEARCH]" if role.needs_tools else ""
            delegate = " [CAN DELEGATE TO SUB-AGENTS]" if role.can_delegate else ""
            lines.append(f"- {name}: {role.description}{tools}{delegate}")
        return "\n".join(lines)
