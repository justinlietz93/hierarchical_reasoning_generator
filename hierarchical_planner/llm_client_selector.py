import logging
from typing import Dict, Any, Optional, Tuple, Callable

# Import all client functions
from . import gemini_client
from . import anthropic_client
from . import openai_client
from .exceptions import ConfigError

logger = logging.getLogger(__name__)

# A map to hold the client functions for each provider
PROVIDER_MAP = {
    "gemini": {
        "structured": gemini_client.generate_structured_content,
        "content": gemini_client.generate_content,
        "retry_call": gemini_client.call_gemini_with_retry,
    },
    "anthropic": {
        "structured": anthropic_client.generate_structured_content,
        "content": anthropic_client.generate_content,
        "retry_call": anthropic_client.call_anthropic_with_retry,
    },
    "openai": {
        "structured": openai_client.generate_structured_content,
        "content": openai_client.generate_content,
        "retry_call": openai_client.call_openai_with_retry,
    },
}

async def select_llm_client(config: Dict[str, Any], agent_name: str) -> Tuple[Callable, Callable, Callable]:
    """
    Selects the appropriate LLM client based on the agent's configuration.

    This function is now explicit and does not fall back to auto-selection.
    It requires that each agent has a provider defined in the config.

    Args:
        config: The application configuration dictionary.
        agent_name: The name of the agent for which to select the client 
                    (e.g., 'planner', 'qa_validator').

    Returns:
        A tuple containing the appropriate client functions:
        (generate_structured_content, generate_content, call_with_retry)

    Raises:
        ConfigError: If the specified agent or its provider is not configured.
    """
    try:
        # Get the provider for the specified agent
        provider = config['agents'][agent_name]['provider']
        logger.info(f"Using {provider.capitalize()} client for agent '{agent_name}' as per configuration.")
        
        # Look up the provider in our map
        client_functions = PROVIDER_MAP.get(provider.lower())
        
        if not client_functions:
            raise ConfigError(f"Invalid provider '{provider}' specified for agent '{agent_name}'.")
            
        return (
            client_functions["structured"],
            client_functions["content"],
            client_functions["retry_call"],
        )
        
    except KeyError as e:
        logger.error(f"Configuration error: Missing '{e.args[0]}' for agent '{agent_name}'.")
        raise ConfigError(f"Configuration for agent '{agent_name}' is missing required key: {e}") from e
