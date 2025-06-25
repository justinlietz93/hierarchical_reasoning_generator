import logging
from typing import Dict, Any, Optional

# Import client functions
from gemini_client import generate_structured_content as gemini_generate_structured_content
from gemini_client import generate_content as gemini_generate_content
from gemini_client import call_gemini_with_retry
from anthropic_client import generate_structured_content as anthropic_generate_structured_content
from anthropic_client import generate_content as anthropic_generate_content
from anthropic_client import call_anthropic_with_retry

logger = logging.getLogger(__name__)

async def select_llm_client(config: Dict[str, Any], provider: Optional[str] = None):
    """
    Selects the appropriate LLM client based on the configuration and optional provider preference.
    
    Args:
        config: The application configuration dictionary.
        provider: Optional provider preference ('gemini', 'anthropic', 'deepseek')
        
    Returns:
        A tuple containing the appropriate client functions:
        (generate_structured_content, generate_content, call_with_retry)
    """
    # If a specific provider is requested, try to use it
    if provider:
        if provider.lower() == 'anthropic':
            if 'anthropic' in config and config.get('anthropic', {}).get('api_key'):
                logger.info("Using Anthropic client with Claude model (user specified).")
                return (
                    anthropic_generate_structured_content,
                    anthropic_generate_content,
                    call_anthropic_with_retry
                )
            else:
                logger.warning("Anthropic provider requested but API key not configured. Falling back to auto-selection.")
        elif provider.lower() == 'gemini':
            if 'api' in config and config.get('api', {}).get('resolved_key'):
                logger.info("Using Gemini client (user specified).")
                return (
                    gemini_generate_structured_content,
                    gemini_generate_content,
                    call_gemini_with_retry
                )
            else:
                logger.warning("Gemini provider requested but API key not configured. Falling back to auto-selection.")
        elif provider.lower() == 'deepseek':
            logger.warning("DeepSeek provider requested but not yet implemented. Falling back to auto-selection.")
    
    # Auto-selection logic (original behavior)
    # Check if Anthropic is configured
    if 'anthropic' in config and config.get('anthropic', {}).get('api_key'):
        logger.info("Using Anthropic client with Claude model (auto-selected).")
        return (
            anthropic_generate_structured_content,
            anthropic_generate_content,
            call_anthropic_with_retry
        )
    # Default to Gemini
    logger.info("Using Gemini client (auto-selected).")
    return (
        gemini_generate_structured_content,
        gemini_generate_content,
        call_gemini_with_retry
    )
