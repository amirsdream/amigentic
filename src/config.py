"""Application configuration module."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # LLM Provider selection
        self.llm_provider: str = os.getenv(
            "LLM_PROVIDER", "ollama"
        ).lower()  # ollama, openai, or claude

        # Ollama settings
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # OpenAI settings
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Anthropic (Claude) settings
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        # Common LLM settings
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

        # Legacy support for OLLAMA_TEMPERATURE
        if os.getenv("OLLAMA_TEMPERATURE"):
            self.llm_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

        # Application settings
        self.phoenix_port: int = int(os.getenv("PHOENIX_PORT", "6006"))
        self.enable_observability: bool = os.getenv("ENABLE_OBSERVABILITY", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT", "You are a helpful assistant that can search the web for information."
        )
        self.log_file: str = os.getenv("LOG_FILE", "agent.log")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.max_parallel_agents: int = int(os.getenv("MAX_PARALLEL_AGENTS", "3"))

        # UI Display settings
        self.ui_display_limit: int = int(os.getenv("UI_DISPLAY_LIMIT", "200"))

        # Agent context limits (for inter-agent communication)
        self.agent_context_limit: int = int(
            os.getenv("AGENT_CONTEXT_LIMIT", "4000")
        )  # Per-agent output limit when passed to next layer
        self.agent_history_limit: int = int(
            os.getenv("AGENT_HISTORY_LIMIT", "500")
        )  # Preview limit for conversation history

        # Delegation depth limits (for hierarchical agent execution)
        self.max_delegation_depth: int = int(
            os.getenv("MAX_DELEGATION_DEPTH", "3")
        )  # Default max depth for agent delegation
        self.absolute_max_depth: int = int(
            os.getenv("ABSOLUTE_MAX_DEPTH", "5")
        )  # Hard safety limit to prevent infinite recursion
        self.max_subtasks_per_delegation: int = int(
            os.getenv("MAX_SUBTASKS_PER_DELEGATION", "5")
        )  # Max subtasks an agent can delegate to at once
        self.max_total_delegations: int = int(
            os.getenv("MAX_TOTAL_DELEGATIONS", "20")
        )  # Max total delegations per query (across all depths)

        # RAG settings
        self.enable_rag: bool = os.getenv("ENABLE_RAG", "false").lower() in ("true", "1", "yes")
        self.rag_vector_store: str = os.getenv("RAG_VECTOR_STORE", "qdrant")
        self.rag_qdrant_mode: str = os.getenv("RAG_QDRANT_MODE", "memory")
        self.rag_qdrant_url: Optional[str] = os.getenv("RAG_QDRANT_URL")
        self.rag_qdrant_collection: str = os.getenv("RAG_QDRANT_COLLECTION", "knowledge_base")
        self.rag_persist_directory: str = os.getenv("RAG_PERSIST_DIRECTORY", "./rag_data")
        self.rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
        # Use same provider as LLM by default, or override
        self.rag_embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", self.llm_provider)
        # Auto-select model based on provider if not specified
        self.rag_embedding_model: Optional[str] = os.getenv("RAG_EMBEDDING_MODEL")

        # MCP settings
        self.enable_mcp: bool = os.getenv("ENABLE_MCP", "false").lower() in ("true", "1", "yes")
        self.mcp_gateway_url: str = os.getenv("MCP_GATEWAY_URL", "http://localhost:9000")

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration values.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate LLM provider
        if self.llm_provider not in ["ollama", "openai", "claude"]:
            return (
                False,
                f"LLM_PROVIDER must be 'ollama', 'openai', or 'claude', got '{self.llm_provider}'",
            )

        # Validate API keys for cloud providers
        if self.llm_provider == "openai" and not self.openai_api_key:
            return False, "OPENAI_API_KEY is required when LLM_PROVIDER=openai"

        if self.llm_provider == "claude" and not self.anthropic_api_key:
            return False, "ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude"

        if self.llm_temperature < 0 or self.llm_temperature > 2:
            return False, "LLM_TEMPERATURE must be between 0 and 2"

        if self.phoenix_port < 1024 or self.phoenix_port > 65535:
            return False, "PHOENIX_PORT must be between 1024 and 65535"

        if self.max_input_length < 1:
            return False, "MAX_INPUT_LENGTH must be positive"

        if self.max_parallel_agents < 1:
            return False, "MAX_PARALLEL_AGENTS must be positive"

        if self.ui_display_limit < 50:
            return False, "UI_DISPLAY_LIMIT must be at least 50 characters"

        if self.max_delegation_depth < 1 or self.max_delegation_depth > 10:
            return False, "MAX_DELEGATION_DEPTH must be between 1 and 10"

        if self.absolute_max_depth < self.max_delegation_depth:
            return False, "ABSOLUTE_MAX_DEPTH must be >= MAX_DELEGATION_DEPTH"

        return True, None

    def get_model_name(self) -> str:
        """Get the model name for the current provider."""
        return {
            "ollama": self.ollama_model,
            "openai": self.openai_model,
            "claude": self.anthropic_model,
        }.get(self.llm_provider, "unknown")

    def __repr__(self) -> str:
        """String representation of config."""
        model = self.get_model_name()

        return (
            f"Config(provider={self.llm_provider}, "
            f"model={model}, "
            f"temperature={self.llm_temperature}, "
            f"port={self.phoenix_port})"
        )
