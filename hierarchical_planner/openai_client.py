import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional

import openai
from openai import AsyncOpenAI

from .exceptions import ApiKeyError, ApiCallError, ApiResponseError, JsonParsingError

logger = logging.getLogger(__name__)

async def generate_content(prompt: str, config: Dict[str, Any]) -> str:
    """
    Sends a prompt to the configured OpenAI model and returns the text response.
    """
    api_key = config.get("openai", {}).get("api_key")
    if not api_key:
        raise ApiKeyError("OpenAI API key is not configured.")

    client = AsyncOpenAI(api_key=api_key)
    model = config.get("openai", {}).get("model_name", "o4-mini")

    try:
        response = await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                "type": "text"
                }
            },
            reasoning={
                "effort": "medium",
                "summary": "auto"
            },
            tools=[],
            store=True
        )
        # Per the user's guidance, the response object should have a 'text' attribute.
        if not response.text:
            raise ApiResponseError("OpenAI API response has no text content.")
        
        return response.text

    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        raise ApiCallError(f"OpenAI API error: {e}") from e

async def generate_structured_content(prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends a prompt expecting a structured (JSON) response from OpenAI.
    """
    content = await generate_content(prompt, config)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from OpenAI response: {content}", exc_info=True)
        raise JsonParsingError(f"OpenAI response was not valid JSON: {e}") from e

async def call_openai_with_retry(prompt_template: str, context: dict, config: Dict[str, Any], is_structured: bool = True):
    """Calls the appropriate OpenAI client function with retries."""
    prompt = prompt_template.format(**context)
    max_retries = config.get('api', {}).get('retries', 3)
    last_exception = None

    for attempt in range(max_retries):
        try:
            if is_structured:
                response = await generate_structured_content(prompt, config)
            else:
                response = await generate_content(prompt, config)
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"OpenAI call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) reached for OpenAI prompt. Last error: {last_exception}", exc_info=True)
                raise ApiCallError(f"OpenAI API call failed after {max_retries} retries.") from last_exception
