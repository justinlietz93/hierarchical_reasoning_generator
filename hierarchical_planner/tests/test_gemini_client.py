import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import google.generativeai as genai
from google.generativeai.types import generation_types

# Module to test
from .. import gemini_client
from ..exceptions import ApiKeyError, ApiCallError, ApiResponseError, ApiBlockedError, JsonParsingError, JsonProcessingError

# --- Test Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a basic mock config dictionary for tests."""
    return {
        'api': {
            'resolved_key': 'test_api_key_123',
            'model_name': 'mock-model-test',
            'temperature': 0.5,
            'max_output_tokens': 100,
            'retries': 2,
            'delay_between_qa_calls_sec': 0.1 # For faster tests
        },
        # Add other sections if needed by client functions
    }

@pytest.fixture
def mock_generative_model(mocker):
    """Mocks the genai.GenerativeModel instance and its methods."""
    mock_model = AsyncMock(spec=genai.GenerativeModel)
    mock_model.model_name = 'mock-model-test' # Set the model name attribute

    # Mock the response structure
    mock_response = AsyncMock()
    mock_candidate = MagicMock()
    mock_content = MagicMock()
    mock_part = MagicMock()

    # Default successful response setup
    mock_part.text = "Default successful response text."
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    # Use simple string for finish_reason
    mock_candidate.finish_reason = "STOP"
    mock_candidate.safety_ratings = [] # Assume safe by default
    mock_response.candidates = [mock_candidate]
    mock_response.prompt_feedback = MagicMock() # Add prompt_feedback mock
    mock_response.prompt_feedback.block_reason = None
    mock_response.prompt_feedback.safety_ratings = []

    # Mock the response.text property to return text from parts
    mock_response.text = mock_part.text

    mock_model.generate_content_async.return_value = mock_response

    # Patch the class instantiation
    mocker.patch('google.generativeai.GenerativeModel', return_value=mock_model)
    # Also patch configure if needed, though we test API key errors separately
    mocker.patch('google.generativeai.configure')

    return mock_model, mock_response, mock_candidate, mock_content, mock_part


# --- Test Cases ---

@pytest.mark.asyncio
async def test_generate_content_success(mock_config, mock_generative_model):
    """Test successful content generation."""
    mock_model, mock_response, _, _, mock_part = mock_generative_model
    mock_part.text = "Test response content."
    mock_response.text = "Test response content."

    result = await gemini_client.generate_content("Test prompt", mock_config)

    assert result == "Test response content."
    mock_model.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_generate_content_api_error(mock_config, mock_generative_model):
    """Test handling of generic API errors during generation."""
    mock_model, _, _, _, _ = mock_generative_model
    mock_model.generate_content_async.side_effect = Exception("Simulated API network error")

    with pytest.raises(ApiCallError, match="Simulated API network error"):
        await gemini_client.generate_content("Test prompt", mock_config)

@pytest.mark.asyncio
async def test_generate_content_blocked(mock_config, mock_generative_model):
    """Test handling of blocked responses (no candidates)."""
    mock_model, mock_response, _, _, _ = mock_generative_model
    mock_response.candidates = [] # Simulate no candidates
    
    # Use string instead of enum
    mock_response.prompt_feedback.block_reason = "SAFETY" 

    with pytest.raises(ApiBlockedError, match="No candidates returned"):
        await gemini_client.generate_content("Test prompt", mock_config)

@pytest.mark.asyncio
async def test_generate_content_empty_parts(mock_config, mock_generative_model):
    """Test handling of responses with candidates but empty content parts."""
    mock_model, mock_response, mock_candidate, mock_content, _ = mock_generative_model
    mock_content.parts = [] # Simulate empty parts
    
    # Use string instead of enum
    mock_candidate.finish_reason = "MAX_TOKENS"  # Example reason

    # Remove the problematic patch and just test that ApiResponseError is raised
    with pytest.raises(ApiResponseError, match="has no content parts"):
        await gemini_client.generate_content("Test prompt", mock_config)

@pytest.mark.asyncio
async def test_generate_content_stopped_exception(mock_config, mock_generative_model):
    """Test handling of StopCandidateException."""
    mock_model, _, _, _, _ = mock_generative_model
    
    # Create a custom exception to simulate StopCandidateException
    class MockStopCandidateException(Exception):
        pass
    
    # Set the class name to match what we're checking for in generate_content
    MockStopCandidateException.__name__ = 'StopCandidateException'
    mock_model.generate_content_async.side_effect = MockStopCandidateException("Stopped due to safety")

    with pytest.raises(ApiBlockedError, match="Stopped due to safety"):
        await gemini_client.generate_content("Test prompt", mock_config)


@pytest.mark.asyncio
async def test_generate_structured_content_success(mock_config, mock_generative_model):
    """Test successful structured content generation and JSON parsing."""
    mock_model, mock_response, _, _, mock_part = mock_generative_model
    expected_dict = {"key": "value", "list": [1, 2]}
    mock_part.text = f"```json\n{json.dumps(expected_dict)}\n```" # Simulate markdown fences
    mock_response.text = mock_part.text

    result = await gemini_client.generate_structured_content("Test prompt for JSON", mock_config)

    assert result == expected_dict
    mock_model.generate_content_async.assert_called_once()
    # Check if the structure hint was added to the prompt
    call_args, _ = mock_model.generate_content_async.call_args
    assert "Return only JSON." in call_args[0]

@pytest.mark.asyncio
async def test_generate_structured_content_invalid_json(mock_config, mock_generative_model):
    """Test handling of invalid JSON responses."""
    mock_model, mock_response, _, _, mock_part = mock_generative_model
    mock_part.text = "This is not JSON { definitely not"
    mock_response.text = mock_part.text

    with pytest.raises(JsonParsingError, match="not valid JSON"):
        await gemini_client.generate_structured_content("Test prompt for JSON", mock_config)

@pytest.mark.asyncio
async def test_generate_structured_content_empty_response(mock_config, mock_generative_model):
    """Test handling of empty responses after cleaning."""
    mock_model, mock_response, _, _, mock_part = mock_generative_model
    mock_part.text = "```json\n```" # Empty after cleaning
    mock_response.text = mock_part.text

    with pytest.raises(ApiResponseError, match="empty response after cleaning"):
        await gemini_client.generate_structured_content("Test prompt for JSON", mock_config)

@pytest.mark.asyncio
async def test_generate_structured_content_api_error_propagates(mock_config, mock_generative_model):
    """Test that API errors from generate_content propagate."""
    mock_model, _, _, _, _ = mock_generative_model
    mock_model.generate_content_async.side_effect = ApiCallError("Underlying API error")

    with pytest.raises(ApiCallError, match="Underlying API error"):
        await gemini_client.generate_structured_content("Test prompt for JSON", mock_config)


@pytest.mark.asyncio
async def test_call_gemini_with_retry_success_first_try(mock_config, mock_generative_model):
    """Test retry wrapper succeeding on the first attempt."""
    mock_model, mock_response, _, _, mock_part = mock_generative_model
    expected_dict = {"data": "ok"}
    json_str = json.dumps(expected_dict)
    mock_part.text = json_str
    mock_response.text = json_str

    result = await gemini_client.call_gemini_with_retry("Template: {placeholder}", {"placeholder": "value"}, mock_config, is_structured=True)

    assert result == expected_dict
    mock_model.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_call_gemini_with_retry_success_after_failure(mock_config, mock_generative_model, mocker):
    """Test retry wrapper succeeding after initial failures."""
    mock_model, _, _, _, _ = mock_generative_model
    expected_dict = {"data": "finally ok"}
    final_response_text = json.dumps(expected_dict)

    # Create a custom response for the successful second attempt
    success_response = AsyncMock()
    success_candidate = MagicMock()
    success_content = MagicMock()
    success_part = MagicMock(text=final_response_text)
    success_content.parts = [success_part]
    success_candidate.content = success_content
    success_response.candidates = [success_candidate]
    success_response.text = final_response_text

    # Simulate failure, then success
    mock_model.generate_content_async.side_effect = [
        Exception("Attempt 1 failed"),
        success_response
    ]
    
    mocker.patch('asyncio.sleep', return_value=None) # Patch sleep to avoid delays

    result = await gemini_client.call_gemini_with_retry("Template: {p}", {"p": "v"}, mock_config, is_structured=True)

    assert result == expected_dict
    assert mock_model.generate_content_async.call_count == 2
    asyncio.sleep.assert_called_once()

@pytest.mark.asyncio
async def test_call_gemini_with_retry_max_retries_exceeded(mock_config, mock_generative_model, mocker):
    """Test retry wrapper failing after max retries."""
    mock_model, _, _, _, _ = mock_generative_model
    mock_config['api']['retries'] = 2 # Set retries for test

    # Simulate consistent failure
    mock_model.generate_content_async.side_effect = Exception("Persistent failure")
    mocker.patch('asyncio.sleep', return_value=None) # Patch sleep

    with pytest.raises(ApiCallError, match="failed after 2 retries"):
        await gemini_client.call_gemini_with_retry("Template: {p}", {"p": "v"}, mock_config, is_structured=True)

    assert mock_model.generate_content_async.call_count == 2
    assert asyncio.sleep.call_count == 1 # Called after first failure

def test_configure_client_no_key(mock_config):
    """Test configure_client raises error if key is missing."""
    mock_config['api']['resolved_key'] = None
    with pytest.raises(ApiKeyError):
        gemini_client.configure_client(None) # Pass None explicitly

@patch('google.generativeai.configure', side_effect=Exception("Config lib error"))
def test_configure_client_library_error(mock_genai_configure, mock_config):
    """Test configure_client handles errors from the genai library."""
    with pytest.raises(ApiKeyError, match="Config lib error"):
        gemini_client.configure_client(mock_config['api']['resolved_key'])

# Add more tests for get_gemini_model if needed, e.g., different configs
