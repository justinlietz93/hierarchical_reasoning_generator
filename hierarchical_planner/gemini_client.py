import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro-exp-03-25"

def configure_client():
    """Configures the Gemini client with the API key."""
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=API_KEY)

def get_gemini_model():
    """Initializes and returns the Gemini generative model."""
    configure_client()
    # Configuration for safety settings and generation - adjust as needed
    generation_config = {
        "temperature": 0.7, # Controls randomness
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 8192, # Max output tokens for the model
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

async def generate_content(prompt: str) -> str:
    """
    Sends a prompt to the configured Gemini model and returns the text response.

    Args:
        prompt: The prompt string to send to the model.

    Returns:
        The generated text content.

    Raises:
        Exception: If the API call fails or returns an unexpected response.
    """
    try:
        model = get_gemini_model()
        response = await model.generate_content_async(prompt) # Use async for potentially long calls
        # Basic error/safety check (can be expanded)
        if not response.candidates or not response.candidates[0].content.parts:
             # Attempt to get blockage reason if available
            blockage_reason = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                blockage_reason = response.prompt_feedback.block_reason.name
            raise Exception(f"API call failed or blocked. Reason: {blockage_reason}. Full response: {response}")

        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        # Re-raise the exception to be handled by the caller
        raise

async def generate_structured_content(prompt: str, structure_hint: str = "Return only JSON.") -> dict:
    """
    Sends a prompt expecting a structured (JSON) response from Gemini.

    Args:
        prompt: The prompt string, instructing the model to return JSON.
        structure_hint: A hint added to the prompt to reinforce JSON output.

    Returns:
        A dictionary parsed from the JSON response.

    Raises:
        Exception: If the API call fails or the response is not valid JSON.
    """
    full_prompt = f"{prompt}\n\n{structure_hint}"
    raw_response = await generate_content(full_prompt)

    # Attempt to clean and parse the JSON response
    try:
        # Basic cleaning: remove potential markdown code fences
        cleaned_response = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
        # More robust cleaning might be needed depending on model behavior
        parsed_json = json.loads(cleaned_response)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Raw response was:\n---\n{raw_response}\n---")
        raise Exception("Gemini response was not valid JSON.") from e
    except Exception as e:
        # Catch other potential errors during parsing/cleaning
        print(f"Error processing Gemini response: {e}")
        raise

# Example usage (for testing purposes, can be removed)
# if __name__ == "__main__":
#     import asyncio
#     async def test():
#         try:
#             # Ensure you have GEMINI_API_KEY set in your environment or .env file
#             print("Testing basic generation...")
#             basic_response = await generate_content("Explain the concept of recursion in one sentence.")
#             print(f"Basic Response: {basic_response}")

#             print("\nTesting structured generation...")
#             structured_prompt = "List three common Python data types as a JSON array."
#             structured_response = await generate_structured_content(structured_prompt)
#             print(f"Structured Response: {json.dumps(structured_response, indent=2)}")

#         except Exception as e:
#             print(f"Test failed: {e}")

#     asyncio.run(test())
