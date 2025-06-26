"""
Client module for interacting with the Google Generative AI (Gemini) API.

Provides functions for configuring the client, initializing the model,
generating text content, generating structured JSON content, and handling retries.
"""
import google.generativeai as genai
import json
import logging
import asyncio
import re
from typing import Dict, Any
# Remove the specific generation_types import
# from google.generativeai.types import generation_types 

# Local imports for exceptions
from .exceptions import ApiKeyError, ApiCallError, ApiResponseError, ApiBlockedError, JsonParsingError, JsonProcessingError
# Import DeepSeek client for fallback
from . import deepseek_v3_client

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global variables to store initialized model
_gemini_model = None
_gemini_model_name = None
_client_configured = False
_deepseek_fallback_enabled = False

def configure_client(api_key: str):
    """Configures the Gemini client with the API key."""
    global _client_configured
    
    # Skip if already configured
    if _client_configured:
        return
        
    if not api_key:
        # This check might be redundant if config_loader ensures key exists
        # but kept for safety.
        logger.error("Attempted to configure Gemini client without an API key.")
        raise ApiKeyError("API key is required to configure the Gemini client.") # Use custom exception
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini client configured successfully.")
        _client_configured = True
    except Exception as e:
        # Catch potential configuration errors from the library
        logger.error(f"Failed to configure Gemini client: {e}", exc_info=True)
        raise ApiKeyError(f"Failed to configure Gemini client: {e}") from e # Use custom exception

def configure_deepseek_fallback(config: Dict[str, Any]):
    """Configure the DeepSeek client for fallback if API key is available."""
    global _deepseek_fallback_enabled
    
    deepseek_config = config.get('deepseek', {})
    api_key = deepseek_config.get('api_key')
    base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
    
    if not api_key:
        logger.warning("DeepSeek fallback is disabled because no API key was provided")
        _deepseek_fallback_enabled = False
        return False
        
    try:
        deepseek_v3_client.configure_client(api_key, base_url)
        _deepseek_fallback_enabled = True
        logger.info("DeepSeek fallback is enabled and configured successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure DeepSeek fallback: {e}")
        _deepseek_fallback_enabled = False
        return False

def get_gemini_model(config: Dict[str, Any]):
    """Initializes and returns the Gemini generative model based on config."""
    global _gemini_model, _gemini_model_name
    
    api_config = config.get('api', {})
    api_key = api_config.get('resolved_key')
    model_name = api_config.get('model_name', 'gemini-2.5-pro-exp-03-25') # Default if missing
    
    # If model is already initialized with the same model name, reuse it
    if _gemini_model is not None and _gemini_model_name == model_name:
        return _gemini_model

    # Configure client if model is not yet initialized or model name has changed
    configure_client(api_key)

    # Configuration for safety settings and generation from config
    generation_config = {
        "temperature": api_config.get('temperature', 0.7),
        "top_p": api_config.get('top_p', 1.0), # Added default
        "top_k": api_config.get('top_k', 32),   # Added default
        "max_output_tokens": api_config.get('max_output_tokens', 8192),
    }
    # Default safety settings (can be made configurable later if needed)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    try:
        _gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings # Use the defined settings
        )
        _gemini_model_name = model_name
        logger.info(f"Gemini model '{model_name}' initialized.")
        return _gemini_model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model '{model_name}': {e}", exc_info=True)
        # Catch potential model initialization errors
        raise ApiCallError(f"Failed to initialize Gemini model '{model_name}': {e}") from e # Use custom exception

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
    Sends a prompt to the configured Gemini model and returns the text response.

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
        model = get_gemini_model(config)
        request_options = {"timeout": 600}
        logger.debug(f"Sending prompt to Gemini model {model.model_name}:\n{prompt[:200]}...") # Log truncated prompt
        response = await model.generate_content_async(prompt, request_options=request_options)

        # More robust checking based on library structure
        if not response.candidates:
            blockage_reason = "Unknown"
            safety_ratings = None
            if response.prompt_feedback:
                try:
                    # Handle both string and enum types for block_reason
                    if hasattr(response.prompt_feedback.block_reason, 'name'):
                        blockage_reason = response.prompt_feedback.block_reason.name
                    else:
                        blockage_reason = str(response.prompt_feedback.block_reason)
                    safety_ratings = response.prompt_feedback.safety_ratings
                except AttributeError:
                    pass # Ignore if attributes don't exist
            msg = f"Gemini API call failed or blocked. No candidates returned."
            logger.error(f"{msg} Reason: {blockage_reason}. Safety Ratings: {safety_ratings}")
            raise ApiBlockedError(msg, reason=blockage_reason, ratings=safety_ratings)

        # Check the content of the first candidate
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            # Check finish reason if content is missing
            finish_reason_value = getattr(candidate, 'finish_reason', "UNKNOWN")
            # Handle both string and enum types for finish_reason
            if hasattr(finish_reason_value, 'name'):
                finish_reason_str = finish_reason_value.name
            else:
                finish_reason_str = str(finish_reason_value)
                
            safety_ratings = getattr(candidate, 'safety_ratings', None)
            msg = f"Gemini API response candidate has no content parts. Finish Reason: {finish_reason_str}."
            logger.error(f"{msg} Safety Ratings: {safety_ratings}")
            # Treat safety block as ApiBlockedError, others as ApiResponseError
            if finish_reason_str == "SAFETY":
                raise ApiBlockedError(msg, reason=finish_reason_str, ratings=safety_ratings)
            else:
                raise ApiResponseError(msg)

        logger.debug("Gemini response received successfully.")
        # Extract the text from the response
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'text'):
                text_parts.append(part.text)
        
        # Join all text parts into a single string
        result_text = ''.join(text_parts)
        return result_text

    except (ApiBlockedError, ApiResponseError) as e:
        # Re-raise specific errors already handled
        raise e
    except Exception as e:
        # Handle StopCandidateException by name instead of by type
        if e.__class__.__name__ == 'StopCandidateException':
            # Specific exception for stopped candidates (often safety)
            logger.error(f"Gemini API call stopped: {e}", exc_info=True)
            raise ApiBlockedError(f"Gemini API call stopped: {e}", reason="STOPPED") from e
        # Catch other potential API errors (network, auth, etc.)
        logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        raise ApiCallError(f"Error during Gemini API call: {e}") from e


async def generate_structured_content(prompt: str, config: Dict[str, Any], structure_hint: str = "Return only JSON.") -> dict:
    """
    Sends a prompt expecting a structured (JSON) response from Gemini.

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
        # Remove markdown code blocks if present (compatible with all Python versions)
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:] # Remove ```json prefix
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:] # Remove ``` prefix
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3] # Remove ``` suffix
        cleaned_response = cleaned_response.strip()

        # Handle potential empty responses after cleaning
        if not cleaned_response:
            logger.error("Gemini returned an empty response after cleaning for structured content.")
            raise ApiResponseError("Gemini returned an empty response after cleaning.")

        # Try to fix common JSON issues before parsing
        sanitized_response = _sanitize_json_response(cleaned_response)

        try:
            parsed_json = json.loads(sanitized_response)
            
            # Normalize steps format if the response contains steps
            if isinstance(parsed_json, dict) and "steps" in parsed_json:
                parsed_json["steps"] = _normalize_steps_format(parsed_json["steps"])
                
            logger.debug("Successfully parsed JSON response from Gemini.")
            return parsed_json
        except json.JSONDecodeError as first_error:
            # Try additional fallback strategies
            logger.warning(f"First JSON parse attempt failed: {first_error}. Trying fallback strategies...")
            
            # Strategy 1: Extract just the JSON part if there's extra text
            json_match = re.search(r'\{.*\}', sanitized_response, re.DOTALL)
            if json_match:
                try:
                    fallback_json = json_match.group(0)
                    fallback_sanitized = _sanitize_json_response(fallback_json)
                    parsed_json = json.loads(fallback_sanitized)
                    
                    # Apply normalization to fallback result as well
                    if isinstance(parsed_json, dict) and "steps" in parsed_json:
                        parsed_json["steps"] = _normalize_steps_format(parsed_json["steps"])
                        
                    logger.info("Successfully parsed JSON using fallback extraction strategy 1.")
                    return parsed_json
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Try to manually fix common JSON issues
            try:
                # More aggressive fixing - try to reconstruct the JSON structure
                manual_fix = _manual_json_fix(cleaned_response)
                parsed_json = json.loads(manual_fix)
                
                if isinstance(parsed_json, dict) and "steps" in parsed_json:
                    parsed_json["steps"] = _normalize_steps_format(parsed_json["steps"])
                
                logger.info("Successfully parsed JSON using manual fix strategy.")
                return parsed_json
            except (json.JSONDecodeError, Exception):
                pass
            
            # Strategy 3: Try to parse line by line and reconstruct
            try:
                reconstructed = _reconstruct_json_from_lines(cleaned_response)
                if reconstructed:
                    parsed_json = json.loads(reconstructed)
                    
                    if isinstance(parsed_json, dict) and "steps" in parsed_json:
                        parsed_json["steps"] = _normalize_steps_format(parsed_json["steps"])
                    
                    logger.info("Successfully parsed JSON using line-by-line reconstruction.")
                    return parsed_json
            except (json.JSONDecodeError, Exception):
                pass
            
            # Strategy 4: Last resort - create a minimal valid response
            logger.warning("All JSON parsing strategies failed. Creating minimal valid response.")
            
            # Try to extract meaningful content even if we can't parse the JSON properly
            if "steps" in raw_response.lower():
                return {"steps": [{"id": "error_step", "description": "Failed to parse LLM response - please retry"}]}
            elif "tasks" in raw_response.lower():
                return {"tasks": ["Failed to parse LLM response - please retry"]}
            elif "phases" in raw_response.lower():
                return {"phases": ["Failed to parse LLM response - please retry"]}
            
            # If all parsing fails, raise the original error
            logger.error(f"All JSON parsing strategies failed. Original error: {first_error}")
            raise first_error
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}. Raw response was:\n---\n{raw_response}\n---", exc_info=True)
        raise JsonParsingError(f"Gemini response was not valid JSON: {e}") from e
    except (ApiCallError, ApiResponseError) as e:
        # Re-raise errors from generate_content
        raise e
    except Exception as e:
        # Catch other potential errors during processing
        logger.error(f"Error processing Gemini structured response: {e}", exc_info=True)
        raise JsonProcessingError(f"Error processing Gemini structured response: {e}") from e


# --- Helper Function (Moved from main.py) ---

async def call_gemini_with_retry(prompt_template: str, context: dict, config: Dict[str, Any], is_structured: bool = True):
    """Calls the appropriate Gemini client function with retries, using config."""
    global _deepseek_fallback_enabled
    
    # Make sure DeepSeek fallback is configured if available in config
    if not _deepseek_fallback_enabled and 'deepseek' in config:
        configure_deepseek_fallback(config)
        
    prompt = prompt_template.format(**context)
    last_exception = None
    max_retries = config.get('api', {}).get('retries', 3) # Get retries from config

    for attempt in range(max_retries):
        try:
            if is_structured:
                # Pass config to generate_structured_content
                response = await generate_structured_content(prompt, config)
            else:
                 # Pass config to generate_content
                 # This branch might not be used if all prompts request JSON
                response = await generate_content(prompt, config)
            logger.debug(f"Gemini call successful after {attempt + 1} attempt(s).")
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"Gemini call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time) # Exponential backoff
            else:
                logger.error(f"Max retries ({max_retries}) reached for prompt. Last error: {last_exception}", exc_info=True)
                
                # Check if this was a rate limit error and if DeepSeek fallback is available
                if _deepseek_fallback_enabled and is_rate_limit_error(last_exception):
                    logger.info("Falling back to DeepSeek model after Gemini rate limit...")
                    try:
                        if is_structured:
                            return await deepseek_v3_client.generate_structured_content(prompt, config)
                        else:
                            return await deepseek_v3_client.generate_content(prompt, config)
                    except Exception as fallback_error:
                        logger.error(f"DeepSeek fallback also failed: {fallback_error}", exc_info=True)
                        # If fallback also fails, raise the original error
                
                # Raise ApiCallError after all retries and fallbacks fail
                raise ApiCallError(f"Gemini API call failed after {max_retries} retries.") from last_exception

def _sanitize_json_response(json_str: str) -> str:
    """
    Sanitize and fix common JSON formatting issues from LLM responses.
    
    Args:
        json_str: Raw JSON string from LLM
        
    Returns:
        Cleaned JSON string
    """
    import json as json_module
    
    # Start with the original string
    sanitized = json_str.strip()
    
    # Remove any non-JSON prefix/suffix text
    # Find the first { and last }
    start_idx = sanitized.find('{')
    end_idx = sanitized.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        sanitized = sanitized[start_idx:end_idx + 1]
    
    # Basic escaping for backslashes (but be careful not to double-escape)
    sanitized = re.sub(r'(?<!\\)\\(?![\\"/bfnrt])', r'\\\\', sanitized)
    
    # Fix common issues with trailing commas in objects and arrays
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    
    # Fix missing commas between array elements (strings)
    # Look for pattern: "text" followed by "text" without comma
    sanitized = re.sub(r'("(?:[^"\\]|\\.)*")(\s+)("(?:[^"\\]|\\.)*")', r'\1,\2\3', sanitized)
    
    # Fix missing commas between object properties
    # Look for pattern: } followed by { without comma
    sanitized = re.sub(r'}(\s*){', r'},\1{', sanitized)
    
    # Fix missing commas between object properties (key-value pairs)
    # Look for pattern: "value" followed by "key": without comma
    sanitized = re.sub(r'("(?:[^"\\]|\\.)*")(\s+)("(?:[^"\\]|\\.)*"\s*:)', r'\1,\2\3', sanitized)
    
    # Fix quotes around object keys that might be malformed
    # Match unquoted keys followed by colon (be more careful)
    sanitized = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', sanitized)
    
    # Fix step keys with spaces like "step 1" -> "step_1" for consistency
    sanitized = re.sub(r'"step\s+(\d+)"', r'"step_\1"', sanitized)
    
    # Try to fix malformed strings by ensuring they're properly quoted
    # This is a more aggressive fix for cases where quotes are missing
    lines = sanitized.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Look for array elements that might be missing quotes
        # Pattern: spaces followed by text that should be quoted
        if re.search(r'^\s*[^"\s{}\[\],:]+\s*[,\]\}]?$', line.strip()):
            # This looks like an unquoted string in an array
            stripped = line.strip()
            if stripped.endswith(','):
                line = line.replace(stripped, f'"{stripped[:-1]}",')
            elif stripped.endswith(']') or stripped.endswith('}'):
                line = line.replace(stripped[:-1], f'"{stripped[:-1]}"') + stripped[-1]
            else:
                line = line.replace(stripped, f'"{stripped}"')
        
        fixed_lines.append(line)
    
    sanitized = '\n'.join(fixed_lines)
    
    # Final attempt: try to parse and fix incrementally
    try:
        json_module.loads(sanitized)
        logger.debug(f"JSON sanitization successful. Original length: {len(json_str)}, Sanitized length: {len(sanitized)}")
    except json_module.JSONDecodeError as e:
        logger.debug(f"JSON sanitization still has issues after fixes: {e}")
    
    return sanitized

def _normalize_steps_format(steps: list) -> list:
    """
    Normalize steps from old format to new format.
    
    Old format: [{"step 1": "description"}, {"step 2": "description"}]
    New format: [{"id": "step_1", "description": "description"}, {"id": "step_2", "description": "description"}]
    
    Args:
        steps: List of step dictionaries
        
    Returns:
        List of normalized step dictionaries
    """
    normalized_steps = []
    
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
            
        # Skip if already in new format
        if "description" in step:
            normalized_steps.append(step)
            continue
            
        # Convert from old format
        normalized_step = {}
        qa_info = step.get("qa_info", {})
        
        # Find the step key (not qa_info)
        step_keys = [k for k in step.keys() if k != "qa_info"]
        if step_keys:
            step_key = step_keys[0]
            step_description = step[step_key]
            
            # Generate new id from old key or use index
            if step_key.lower().startswith("step"):
                step_id = step_key.replace(" ", "_").lower()
            else:
                step_id = f"step_{i+1}"
                
            normalized_step = {
                "id": step_id,
                "description": step_description
            }
            
            if qa_info:
                normalized_step["qa_info"] = qa_info
                
        normalized_steps.append(normalized_step)
    
    return normalized_steps

def _manual_json_fix(json_str: str) -> str:
    """
    Manually attempt to fix JSON by reconstructing it from scratch.
    
    Args:
        json_str: Raw JSON string that failed to parse
        
    Returns:
        Reconstructed JSON string
    """
    lines = json_str.split('\n')
    result_lines = []
    in_array = False
    in_object = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#') or stripped.startswith('//'):
            continue
            
        # Handle opening braces
        if '{' in stripped:
            in_object = True
            result_lines.append(stripped)
            continue
            
        # Handle opening brackets
        if '[' in stripped:
            in_array = True
            result_lines.append(stripped)
            continue
            
        # Handle closing braces/brackets
        if '}' in stripped or ']' in stripped:
            # Make sure the previous line doesn't end with a comma
            if result_lines and result_lines[-1].rstrip().endswith(','):
                result_lines[-1] = result_lines[-1].rstrip()[:-1]
            result_lines.append(stripped)
            if '}' in stripped:
                in_object = False
            if ']' in stripped:
                in_array = False
            continue
            
        # Handle array/object content
        if in_array or in_object:
            # Ensure strings are properly quoted
            if stripped and not stripped.startswith('"') and not stripped.endswith('"'):
                if not stripped.endswith(','):
                    stripped = f'"{stripped}",'
                else:
                    stripped = f'"{stripped[:-1]}",'
            
            # Ensure proper comma placement
            if not stripped.endswith(',') and not stripped.endswith('}') and not stripped.endswith(']'):
                stripped += ','
                
            result_lines.append('  ' + stripped)
    
    return '\n'.join(result_lines)


def _reconstruct_json_from_lines(json_str: str) -> str:
    """
    Try to reconstruct JSON by parsing it line by line and fixing issues.
    
    Args:
        json_str: Raw JSON string that failed to parse
        
    Returns:
        Reconstructed JSON string or None if failed
    """
    try:
        # Look for array patterns like ["item1", "item2", ...]
        if '"steps":' in json_str or '"tasks":' in json_str:
            # Extract the array content
            array_match = re.search(r'["\'](steps|tasks)["\']\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if array_match:
                key_name = array_match.group(1)
                array_content = array_match.group(2)
                
                # Split by lines and clean each item
                lines = array_content.split('\n')
                items = []
                
                for line in lines:
                    line = line.strip()
                    if not line or line in [',', '[', ']']:
                        continue
                    
                    # Remove trailing comma
                    if line.endswith(','):
                        line = line[:-1]
                    
                    # Ensure proper quoting
                    if not line.startswith('"') and not line.startswith("'"):
                        # Find the actual content (skip leading quotes/spaces)
                        content = line.strip('\'"')
                        items.append(f'"{content}"')
                    else:
                        items.append(line)
                
                # Reconstruct the JSON
                reconstructed = '{\n  "' + key_name + '": [\n'
                for i, item in enumerate(items):
                    reconstructed += f'    {item}'
                    if i < len(items) - 1:
                        reconstructed += ','
                    reconstructed += '\n'
                reconstructed += '  ]\n}'
                
                return reconstructed
        
        return None
    except Exception:
        return None
