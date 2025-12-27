"""LLM pricing configuration for cost calculation.

Prices are per 1 million tokens.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing for a specific model."""
    input_cost: float  # $ per 1M tokens
    output_cost: float  # $ per 1M tokens
    

# Pricing table (costs per 1 million tokens)
PRICING_TABLE: Dict[str, Dict[str, ModelPricing]] = {
    "google": {
        # Gemini 2.5 series
        "gemini-2.5-flash-lite": ModelPricing(0.0375, 0.15),
        "gemini-2.5-flash": ModelPricing(0.075, 0.30),
        "gemini-2.5-pro": ModelPricing(1.25, 5.00),
        "gemini-2.5-pro-deep-think": ModelPricing(2.50, 10.00),
        # Gemini 2.0 series
        "gemini-2.0-flash-lite": ModelPricing(0.075, 0.30),
        "gemini-2.0-flash": ModelPricing(0.10, 0.40),
        "gemini-2.0-flash-exp": ModelPricing(0.10, 0.40),
        # Legacy
        "gemini-1.5-flash": ModelPricing(0.075, 0.30),
        "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    },
    "openai": {
        # GPT-4.1 series
        "gpt-4.1-nano": ModelPricing(0.10, 0.40),
        "gpt-4.1-mini": ModelPricing(0.40, 1.60),
        "gpt-4.1": ModelPricing(2.00, 8.00),
        # GPT-4o series
        "gpt-4o-mini": ModelPricing(0.15, 0.60),
        "gpt-4o": ModelPricing(2.50, 10.00),
        # GPT-4.5
        "gpt-4.5": ModelPricing(75.00, 150.00),
        "gpt-4.5-preview": ModelPricing(75.00, 150.00),
        # o-series reasoning models
        "o1-mini": ModelPricing(1.10, 4.40),
        "o1": ModelPricing(15.00, 60.00),
        "o1-preview": ModelPricing(15.00, 60.00),
        "o1-pro": ModelPricing(150.00, 150.00),  # Output cost not specified, using input
        "o3-mini": ModelPricing(1.10, 4.40),
        "o3": ModelPricing(10.00, 40.00),
        "o3-pro": ModelPricing(30.00, 120.00),
        "o4-mini": ModelPricing(1.10, 4.40),
        # Legacy
        "gpt-4-turbo": ModelPricing(10.00, 30.00),
        "gpt-4": ModelPricing(30.00, 60.00),
        "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    },
    "anthropic": {
        # Claude 4 series
        "claude-sonnet-4-5-20250929": ModelPricing(3.00, 15.00),
        "claude-4-opus": ModelPricing(15.00, 75.00),
        "claude-4-sonnet": ModelPricing(3.00, 15.00),
        # Claude 3.7 series
        "claude-3.7-sonnet": ModelPricing(3.00, 15.00),
        # Claude 3.5 series
        "claude-3.5-sonnet": ModelPricing(3.00, 15.00),
        "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
        "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00),
        "claude-3.5-haiku": ModelPricing(0.80, 4.00),
        "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
        # Claude 3 series (legacy)
        "claude-3-opus": ModelPricing(15.00, 75.00),
        "claude-3-sonnet": ModelPricing(3.00, 15.00),
        "claude-3-haiku": ModelPricing(0.25, 1.25),
    },
    # Ollama/local models are free
    "ollama": {
        "default": ModelPricing(0.0, 0.0),
    },
}

# Provider name aliases
PROVIDER_ALIASES = {
    "google_genai": "google",
    "google-genai": "google",
    "gemini": "google",
    "openai": "openai",
    "azure": "openai",  # Azure uses OpenAI models
    "anthropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
}


def normalize_provider(provider: str) -> str:
    """Normalize provider name to standard format."""
    provider_lower = provider.lower().strip()
    return PROVIDER_ALIASES.get(provider_lower, provider_lower)


def normalize_model(model: str) -> str:
    """Normalize model name for lookup."""
    if not model:
        return "default"
    # Convert to lowercase and remove version suffixes like -20241022
    model_lower = model.lower().strip()
    return model_lower


def get_model_pricing(provider: str, model: str) -> Optional[ModelPricing]:
    """Get pricing for a specific provider/model combination.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'google')
        model: Model name (e.g., 'gpt-4o', 'claude-3.5-sonnet')
        
    Returns:
        ModelPricing if found, None otherwise
    """
    norm_provider = normalize_provider(provider)
    norm_model = normalize_model(model)
    
    provider_pricing = PRICING_TABLE.get(norm_provider)
    if not provider_pricing:
        return None
    
    # Try exact match first
    if norm_model in provider_pricing:
        return provider_pricing[norm_model]
    
    # Try partial match (for versioned models)
    for model_key, pricing in provider_pricing.items():
        if model_key in norm_model or norm_model in model_key:
            return pricing
    
    # Return default if available
    return provider_pricing.get("default")


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> Tuple[float, float, float]:
    """Calculate cost for token usage.
    
    Args:
        provider: Provider name
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Tuple of (input_cost, output_cost, total_cost) in dollars
    """
    pricing = get_model_pricing(provider, model)
    
    if not pricing:
        return (0.0, 0.0, 0.0)
    
    # Prices are per 1M tokens
    input_cost = (input_tokens / 1_000_000) * pricing.input_cost
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost
    total_cost = input_cost + output_cost
    
    return (input_cost, output_cost, total_cost)


def format_cost(cost: float) -> str:
    """Format cost for display.
    
    Args:
        cost: Cost in dollars
        
    Returns:
        Formatted string (e.g., '$0.0012' or '$1.23')
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def get_pricing_table_summary() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Get a summary of all pricing for display.
    
    Returns:
        Dict with provider -> model -> {input_cost, output_cost}
    """
    summary = {}
    for provider, models in PRICING_TABLE.items():
        summary[provider] = {}
        for model, pricing in models.items():
            if model != "default":
                summary[provider][model] = {
                    "input_cost": pricing.input_cost,
                    "output_cost": pricing.output_cost,
                }
    return summary
