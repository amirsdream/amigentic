"""Application configuration module."""

import os
from typing import Optional


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
        self.phoenix_port: int = int(os.getenv("PHOENIX_PORT", "6006"))
        self.max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful assistant that can search the web for information."
        )
        self.log_file: str = os.getenv("LOG_FILE", "agent.log")
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration values.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.ollama_temperature < 0 or self.ollama_temperature > 2:
            return False, "OLLAMA_TEMPERATURE must be between 0 and 2"
        
        if self.phoenix_port < 1024 or self.phoenix_port > 65535:
            return False, "PHOENIX_PORT must be between 1024 and 65535"
        
        if self.max_input_length < 1:
            return False, "MAX_INPUT_LENGTH must be positive"
        
        return True, None

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"Config(model={self.ollama_model}, "
            f"temperature={self.ollama_temperature}, "
            f"port={self.phoenix_port})"
        )
