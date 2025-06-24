"""
Client module for interacting with the DeepSeek-V3-0324 model using the OpenAI API format.

Provides functions for configuring the client, initializing the model,
generating text content, and generating structured JSON content.
"""
import openai
import json
import logging
import asyncio
from typing import Dict, Any, Optional

# Local imports for exceptions
from exceptions import ApiKeyError, ApiCallError, ApiResponseError, JsonParsingError, JsonProcessingError

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global variables to store client configuration
_client_initialized = False
_client_base_url = None
_client_api_key = None

def configure_client(api_key: str, base_url: str = "https://api.deepseek.com/v1"):
    """
    Configures the DeepSeek client with the API key and base URL.
    
    Args:
        api_key: The API key for DeepSeek API access
        base_url: The base URL for DeepSeek API (default: https://api.deepseek.com/v1)
    """
    global _client_initialized, _client_base_url, _client_api_key
    
    if not api_key:
        logger.error("Attempted to configure DeepSeek client without an API key.")
        raise ApiKeyError("API key is required to configure the DeepSeek client.")
    
    try:
        # Store configuration globally
        _client_base_url = base_url
        _client_api_key = api_key
        _client_initialized = True
        
        # Initialize the OpenAI client
        openai.base_url = base_url
        openai.api_key = api_key
        
        logger.info("DeepSeek client configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure DeepSeek client: {e}", exc_info=True)
        raise ApiKeyError(f"Failed to configure DeepSeek client: {e}") from e

def get_client_config():
    """Returns the current client configuration."""
    global _client_initialized, _client_base_url, _client_api_key
    
    if not _client_initialized:
        logger.error("Attempted to use DeepSeek client before configuration.")
        raise ApiKeyError("DeepSeek client is not configured.")
    
    return {
        "base_url": _client_base_url,
        "api_key": _client_api_key
    }

async def generate_content(prompt: str, config: Dict[str, Any]) -> str:
    """
    Sends a prompt to the configured DeepSeek model and returns the text response.

    Args:
        prompt: The prompt string to send to the model.
        config: The application configuration dictionary.

    Returns:
        The generated text content.

    Raises:
        ApiCallError: If the API call fails.
        ApiResponseError: If the response is invalid or empty.
    """
    # Make sure client is configured
    get_client_config()
    
    try:
        # Extract model settings from config
        model_name = config.get('deepseek', {}).get('model_name', 'DeepSeek-V3-0324')
        temperature = config.get('deepseek', {}).get('temperature', 0.6)
        max_tokens = config.get('deepseek', {}).get('max_tokens', 8192)
        top_p = config.get('deepseek', {}).get('top_p', 1.0)
        
        logger.debug(f"Sending prompt to DeepSeek model {model_name}:\n{prompt[:200]}...") # Log truncated prompt
        
        response = await openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Extract content from response
        if not response.choices or len(response.choices) == 0:
            logger.error("DeepSeek API returned no choices.")
            raise ApiResponseError("DeepSeek API returned no choices.")
        
        result_text = response.choices[0].message.content
        if not result_text:
            logger.error("DeepSeek API returned empty content.")
            raise ApiResponseError("DeepSeek API returned empty content.")
        
        logger.debug("DeepSeek response received successfully.")
        return result_text

    except openai.APIError as e:
        logger.error(f"Error during DeepSeek API call: {e}", exc_info=True)
        raise ApiCallError(f"Error during DeepSeek API call: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during DeepSeek API call: {e}", exc_info=True)
        raise ApiCallError(f"Unexpected error during DeepSeek API call: {e}") from e

async def generate_structured_content(prompt: str, config: Dict[str, Any], structure_hint: str = "Return only JSON.") -> dict:
    """
    Sends a prompt expecting a structured (JSON) response from DeepSeek.

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
            logger.error("DeepSeek returned an empty response after cleaning for structured content.")
            raise ApiResponseError("DeepSeek returned an empty response after cleaning.")

        parsed_json = json.loads(cleaned_response)
        logger.debug("Successfully parsed JSON response from DeepSeek.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}. Raw response was:\n---\n{raw_response}\n---", exc_info=True)
        raise JsonParsingError(f"DeepSeek response was not valid JSON: {e}") from e
    except (ApiCallError, ApiResponseError) as e:
        # Re-raise errors from generate_content
        raise e
    except Exception as e:
        # Catch other potential errors during processing
        logger.error(f"Error processing DeepSeek structured response: {e}", exc_info=True)
        raise JsonProcessingError(f"Error processing DeepSeek structured response: {e}") from e
