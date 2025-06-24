"""
LLM selector module for the Persona Builder.

Provides a function to select the appropriate LLM client based on the configuration.
"""
import logging
from typing import Dict, Any, Tuple, Callable

# Import LLM clients
from hierarchical_planner import gemini_client
from hierarchical_planner import anthropic_client

# Configure logger for this module
logger = logging.getLogger(__name__)

async def select_llm_client(config: Dict[str, Any]):
    """
    Selects the appropriate LLM client based on the configuration.
    
    Args:
        config: The application configuration dictionary.
        
    Returns:
        A tuple containing the appropriate client functions:
        (generate_structured_content, generate_content, call_with_retry)
    """
    # Check if Anthropic is configured
    if 'anthropic' in config and config.get('anthropic', {}).get('api_key'):
        logger.info("Using Anthropic client with Claude model for persona parsing.")
        return (
            anthropic_client.generate_structured_content,
            anthropic_client.generate_content,
            anthropic_client.call_anthropic_with_retry
        )
    # Default to Gemini
    logger.info("Using Gemini client for persona parsing.")
    return (
        gemini_client.generate_structured_content,
        gemini_client.generate_content,
        gemini_client.call_gemini_with_retry
    )
