"""
Client module for interacting with the Anthropic API (Claude models).

Provides functions for configuring the client, initializing the model,
generating text content, generating structured JSON content, and handling retries.
Supports Claude 3.7 Sonnet with extended thinking capabilities.
"""
import anthropic
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List

# Local imports for exceptions
from .exceptions import ApiKeyError, ApiCallError, ApiResponseError, ApiBlockedError, JsonParsingError, JsonProcessingError

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global variables to store client configuration
_anthropic_client = None
_client_configured = False

def configure_client(api_key: str):
    """Configures the Anthropic client with the API key."""
    global _anthropic_client, _client_configured
    
    # Skip if already configured
    if _client_configured:
        return
        
    if not api_key:
        logger.error("Attempted to configure Anthropic client without an API key.")
        raise ApiKeyError("API key is required to configure the Anthropic client.")
    
    try:
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client configured successfully.")
        _client_configured = True
    except Exception as e:
        logger.error(f"Failed to configure Anthropic client: {e}", exc_info=True)
        raise ApiKeyError(f"Failed to configure Anthropic client: {e}") from e

def get_anthropic_client(config: Dict[str, Any]):
    """
    Initializes and returns the Anthropic client based on config.
    
    Args:
        config: The application configuration dictionary.
        
    Returns:
        The initialized Anthropic client.
        
    Raises:
        ApiKeyError: If the API key is missing or invalid.
    """
    global _anthropic_client, _client_configured
    
    api_config = config.get('anthropic', {})
    api_key = api_config.get('api_key')
    
    # If client is already initialized, reuse it
    if _client_configured and _anthropic_client is not None:
        return _anthropic_client
    
    # Configure client if not yet initialized
    configure_client(api_key)
    
    return _anthropic_client

def is_rate_limit_error(exception: Exception) -> bool:
    """Check if the exception is due to rate limiting."""
    error_msg = str(exception).lower()
    return (
        "429" in error_msg or 
        "quota" in error_msg or 
        "rate limit" in error_msg or
        "too many requests" in error_msg or
        "resource exhausted" in error_msg
    )

async def generate_content(prompt: str, config: Dict[str, Any]) -> str:
    """
    Sends a prompt to the configured Anthropic model and returns the text response.

    Args:
        prompt: The prompt string to send to the model.
        config: The application configuration dictionary.

    Returns:
        The generated text content.

    Raises:
        ApiCallError: If the API call fails.
        ApiResponseError: If the response is invalid or empty.
        ApiBlockedError: If the request is blocked.
    """
    try:
        client = get_anthropic_client(config)
        
        # Extract model settings from config
        api_config = config.get('anthropic', {})
        model_name = api_config.get('model_name', 'claude-3-7-sonnet-20250219')
        temperature = api_config.get('temperature', 0.7)
        max_tokens = api_config.get('max_tokens', 4096)
        
        # Check if extended thinking is enabled
        extended_thinking = api_config.get('extended_thinking', False)
        
        logger.debug(f"Sending prompt to Anthropic model {model_name}:\n{prompt[:200]}...") # Log truncated prompt
        
        # Prepare messages
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Prepare API call parameters
        api_params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add system prompt if provided
        system_prompt = api_config.get('system_prompt')
        if system_prompt:
            api_params["system"] = system_prompt
            
        # Add extended thinking if enabled
        if extended_thinking:
            api_params["thinking"] = {
                "enabled": True,
                "min_tokens": api_config.get('thinking_min_tokens', 1024),
                "max_tokens": api_config.get('thinking_max_tokens', 8192)
            }
            
        # Add stop sequences if provided
        stop_sequences = api_config.get('stop_sequences')
        if stop_sequences:
            api_params["stop_sequences"] = stop_sequences
            
        # Make the API call
        response = await asyncio.to_thread(
            client.messages.create,
            **api_params
        )
        
        # Extract content from response
        if not response.content:
            logger.error("Anthropic API response has no content.")
            raise ApiResponseError("Anthropic API response has no content.")
        
        # Extract text from content blocks
        result_text = ""
        for block in response.content:
            if block.type == "text":
                result_text += block.text
                
        if not result_text:
            logger.error("Anthropic API response has no text content.")
            raise ApiResponseError("Anthropic API response has no text content.")
            
        logger.debug("Anthropic response received successfully.")
        return result_text
        
    except anthropic.APIError as e:
        # Handle specific API errors
        if "blocked" in str(e).lower() or "content policy" in str(e).lower():
            logger.error(f"Anthropic API request was blocked: {e}", exc_info=True)
            raise ApiBlockedError(f"Anthropic API request was blocked: {e}", reason=str(e))
        else:
            logger.error(f"Error during Anthropic API call: {e}", exc_info=True)
            raise ApiCallError(f"Error during Anthropic API call: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during Anthropic API call: {e}", exc_info=True)
        raise ApiCallError(f"Unexpected error during Anthropic API call: {e}") from e

async def generate_structured_content(prompt: str, config: Dict[str, Any], structure_hint: str = "Return only JSON.") -> dict:
    """
    Sends a prompt expecting a structured (JSON) response from Anthropic.

    Args:
        prompt: The prompt string, instructing the model to return JSON.
        config: The application configuration dictionary.
        structure_hint: A hint added to the prompt to reinforce JSON output.

    Returns:
        A dictionary parsed from the JSON response.

    Raises:
        ApiCallError: If the underlying text generation fails.
        ApiResponseError: If the response is empty after cleaning.
        JsonParsingError: If the response is not valid JSON.
    """
    full_prompt = f"{prompt}\n\n{structure_hint}"
    try:
        # Pass config down to generate_content
        raw_response = await generate_content(full_prompt, config)

        # Attempt to clean and parse the JSON response
        # Basic cleaning: remove potential markdown code fences and leading/trailing whitespace
        cleaned_response = raw_response.strip()
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:] # Remove ```json prefix
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:] # Remove ``` prefix
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3] # Remove ``` suffix
        cleaned_response = cleaned_response.strip()

        # Handle potential empty responses after cleaning
        if not cleaned_response:
            logger.error("Anthropic returned an empty response after cleaning for structured content.")
            raise ApiResponseError("Anthropic returned an empty response after cleaning.")

        parsed_json = json.loads(cleaned_response)
        logger.debug("Successfully parsed JSON response from Anthropic.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}. Raw response was:\n---\n{raw_response}\n---", exc_info=True)
        raise JsonParsingError(f"Anthropic response was not valid JSON: {e}") from e
    except (ApiCallError, ApiResponseError) as e:
        # Re-raise errors from generate_content
        raise e
    except Exception as e:
        # Catch other potential errors during processing
        logger.error(f"Error processing Anthropic structured response: {e}", exc_info=True)
        raise JsonProcessingError(f"Error processing Anthropic structured response: {e}") from e

async def call_anthropic_with_retry(prompt_template: str, context: dict, config: Dict[str, Any], is_structured: bool = True):
    """
    Calls the appropriate Anthropic client function with retries, using config.
    
    Args:
        prompt_template: The prompt template string with placeholders.
        context: Dictionary of values to format the prompt template.
        config: The application configuration dictionary.
        is_structured: Whether to expect a structured (JSON) response.
        
    Returns:
        The generated content or parsed JSON.
        
    Raises:
        ApiCallError: If all retries fail.
    """
    prompt = prompt_template.format(**context)
    last_exception = None
    max_retries = config.get('anthropic', {}).get('retries', 3) # Get retries from config

    for attempt in range(max_retries):
        try:
            if is_structured:
                # Pass config to generate_structured_content
                response = await generate_structured_content(prompt, config)
            else:
                # Pass config to generate_content
                response = await generate_content(prompt, config)
            logger.debug(f"Anthropic call successful after {attempt + 1} attempt(s).")
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"Anthropic call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time) # Exponential backoff
            else:
                logger.error(f"Max retries ({max_retries}) reached for prompt. Last error: {last_exception}", exc_info=True)
                # Raise ApiCallError after all retries fail
                raise ApiCallError(f"Anthropic API call failed after {max_retries} retries.") from last_exception
