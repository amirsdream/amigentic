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
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama, openai, or claude
        
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
        self.enable_observability: bool = os.getenv("ENABLE_OBSERVABILITY", "true").lower() in ("true", "1", "yes")
        self.max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful assistant that can search the web for information."
        )
        self.log_file: str = os.getenv("LOG_FILE", "agent.log")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.max_parallel_agents: int = int(os.getenv("MAX_PARALLEL_AGENTS", "3"))

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration values.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate LLM provider
        if self.llm_provider not in ["ollama", "openai", "claude"]:
            return False, f"LLM_PROVIDER must be 'ollama', 'openai', or 'claude', got '{self.llm_provider}'"
        
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
        
        return True, None

    def __repr__(self) -> str:
        """String representation of config."""
        model = {
            "ollama": self.ollama_model,
            "openai": self.openai_model,
            "claude": self.anthropic_model
        }.get(self.llm_provider, "unknown")
        
        return (
            f"Config(provider={self.llm_provider}, "
            f"model={model}, "
            f"temperature={self.llm_temperature}, "
            f"port={self.phoenix_port})"
        )
