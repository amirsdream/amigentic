"""
Magentic - Magnetic Agent Networks

A dynamic meta-agent system for intelligent task orchestration using LangGraph.
"""

__version__ = "2.0.0"
__author__ = "Magentic Team"
__description__ = "Dynamic meta-agent system with LangGraph infrastructure"

# Core configuration
from .config import Config
from .role_library import RoleLibrary, AgentRole as Role
from .observability import ObservabilityManager

# Agent system (new modular structure)
from .agents import MetaAgentSystem, create_llm

# Coordinator
from .coordinator import MetaCoordinator, ExecutionPlan

# Execution engine
from .execution import MagenticGraphBuilder, MagenticState

# Tools
from .tools import ToolManager

# Services
from .services import MCPClient, RAGService

# Pricing
from .pricing import calculate_cost, get_model_pricing, get_pricing_table_summary

# UI
from .ui import ExecutionVisualizer

# LangGraph runner
from .langgraph_runner import LangGraphExecutor

__all__ = [
    # Config
    "Config",
    "RoleLibrary",
    "Role",
    "ObservabilityManager",
    # Agents
    "MetaAgentSystem",
    "create_llm",
    # Coordinator
    "MetaCoordinator",
    "ExecutionPlan",
    # Execution
    "MagenticGraphBuilder",
    "MagenticState",
    "LangGraphExecutor",
    # Tools
    "ToolManager",
    # Services
    "MCPClient",
    "RAGService",
    # Pricing
    "calculate_cost",
    "get_model_pricing",
    "get_pricing_table_summary",
    # UI
    "ExecutionVisualizer",
]
