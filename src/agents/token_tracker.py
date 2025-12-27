"""Token usage tracking for agent executions."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Import metrics (optional - won't fail if not available)
try:
    from ..metrics import record_tokens, record_cost, PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_AVAILABLE = False
    record_tokens = None
    record_cost = None

# Import pricing
try:
    from ..pricing import calculate_cost, format_cost
except ImportError:
    calculate_cost = None
    format_cost = None

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "cost_formatted": format_cost(self.total_cost) if format_cost else f"${self.total_cost:.4f}",
        }


@dataclass
class AgentTokenUsage:
    """Token usage for a single agent execution."""

    agent_id: str
    role: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    llm_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "llm_calls": self.llm_calls,
            **self.usage.to_dict(),
        }


class TokenTracker:
    """Tracks token usage across an execution session."""

    def __init__(self, provider: str = "unknown", model: str = "unknown"):
        self.agents: Dict[str, AgentTokenUsage] = {}
        self.total: TokenUsage = TokenUsage()
        self.planning_tokens: TokenUsage = TokenUsage()
        self._provider = provider
        self._model = model
    
    def set_llm_info(self, provider: str, model: str):
        """Set the LLM provider and model for metrics."""
        self._provider = provider
        self._model = model

    def extract_usage_from_response(self, response: Any) -> TokenUsage:
        """Extract token usage from an LLM response.

        Works with various LangChain response formats.

        Args:
            response: LLM response object

        Returns:
            TokenUsage with extracted values
        """
        usage = TokenUsage()

        try:
            # Try response_metadata (newer LangChain format)
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata

                # OpenAI format
                if "token_usage" in metadata:
                    token_info = metadata["token_usage"]
                    usage.prompt_tokens = token_info.get("prompt_tokens", 0)
                    usage.completion_tokens = token_info.get("completion_tokens", 0)
                    usage.total_tokens = token_info.get("total_tokens", 0)
                    return usage

                # Anthropic format
                if "usage" in metadata:
                    token_info = metadata["usage"]
                    usage.prompt_tokens = token_info.get("input_tokens", 0)
                    usage.completion_tokens = token_info.get("output_tokens", 0)
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                    return usage

            # Try usage_metadata (alternative format)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage.prompt_tokens = getattr(response.usage_metadata, "input_tokens", 0)
                usage.completion_tokens = getattr(response.usage_metadata, "output_tokens", 0)
                usage.total_tokens = getattr(
                    response.usage_metadata,
                    "total_tokens",
                    usage.prompt_tokens + usage.completion_tokens,
                )
                return usage

            # Try direct attributes
            if hasattr(response, "llm_output") and response.llm_output:
                if "token_usage" in response.llm_output:
                    token_info = response.llm_output["token_usage"]
                    usage.prompt_tokens = token_info.get("prompt_tokens", 0)
                    usage.completion_tokens = token_info.get("completion_tokens", 0)
                    usage.total_tokens = token_info.get("total_tokens", 0)
                    return usage

        except Exception as e:
            logger.debug(f"Could not extract token usage: {e}")

        return usage

    def add_planning_usage(self, response: Any) -> TokenUsage:
        """Track token usage from planning phase.

        Args:
            response: LLM response from coordinator

        Returns:
            TokenUsage extracted
        """
        usage = self.extract_usage_from_response(response)
        
        # Calculate cost
        if calculate_cost:
            input_cost, output_cost, total_cost = calculate_cost(
                self._provider, self._model,
                usage.prompt_tokens, usage.completion_tokens
            )
            usage.input_cost = input_cost
            usage.output_cost = output_cost
            usage.total_cost = total_cost
        
        self.planning_tokens = self.planning_tokens + usage
        self.total = self.total + usage
        
        # Record to Prometheus metrics
        if PROMETHEUS_AVAILABLE and record_tokens:
            record_tokens(
                provider=self._provider or "unknown",
                model=self._model or "planning",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens
            )
        if PROMETHEUS_AVAILABLE and record_cost and usage.total_cost > 0:
            record_cost(
                provider=self._provider or "unknown",
                model=self._model or "planning",
                cost=usage.total_cost
            )
        
        logger.debug(f"Planning tokens: {usage.to_dict()}")
        return usage

    def add_agent_usage(self, agent_id: str, role: str, response: Any) -> TokenUsage:
        """Track token usage from an agent execution.

        Args:
            agent_id: Agent identifier
            role: Agent role name
            response: LLM response

        Returns:
            TokenUsage extracted
        """
        usage = self.extract_usage_from_response(response)

        # Calculate cost
        if calculate_cost:
            input_cost, output_cost, total_cost = calculate_cost(
                self._provider, self._model,
                usage.prompt_tokens, usage.completion_tokens
            )
            usage.input_cost = input_cost
            usage.output_cost = output_cost
            usage.total_cost = total_cost

        if agent_id not in self.agents:
            self.agents[agent_id] = AgentTokenUsage(agent_id=agent_id, role=role)

        self.agents[agent_id].usage = self.agents[agent_id].usage + usage
        self.agents[agent_id].llm_calls += 1
        self.total = self.total + usage

        # Record to Prometheus metrics
        if PROMETHEUS_AVAILABLE and record_tokens:
            record_tokens(
                provider=self._provider or "unknown",
                model=self._model or role,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens
            )
        if PROMETHEUS_AVAILABLE and record_cost and usage.total_cost > 0:
            record_cost(
                provider=self._provider or "unknown",
                model=self._model or role,
                cost=usage.total_cost
            )

        logger.debug(f"Agent {agent_id} tokens: {usage.to_dict()}")
        return usage

    def get_agent_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get token summary for a specific agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].to_dict()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get complete token usage summary."""
        return {
            "total": self.total.to_dict(),
            "planning": self.planning_tokens.to_dict(),
            "agents": {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            "agent_count": len(self.agents),
        }

    def reset(self):
        """Reset all tracking."""
        self.agents = {}
        self.total = TokenUsage()
        self.planning_tokens = TokenUsage()


# Global tracker instance for current execution
_current_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """Get or create the current token tracker."""
    global _current_tracker
    if _current_tracker is None:
        _current_tracker = TokenTracker()
    return _current_tracker


def reset_tracker(provider: str = "unknown", model: str = "unknown") -> TokenTracker:
    """Reset and return a fresh token tracker."""
    global _current_tracker
    _current_tracker = TokenTracker(provider=provider, model=model)
    return _current_tracker
