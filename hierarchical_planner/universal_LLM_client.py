import os
import warnings
import json # Added for config file loading
from typing import List, Dict, Optional, Any, Literal, Tuple

# --- Dependencies ---
# Install required libraries:
# pip install openai anthropic google-generativeai python-dotenv

try:
    import openai
except ImportError:
    warnings.warn("openai library not found. OpenAI models will not be available. "
                  "Install with: pip install openai")
    openai = None

try:
    import anthropic
except ImportError:
    warnings.warn("anthropic library not found. Anthropic models will not be available. "
                  "Install with: pip install anthropic")
    anthropic = None

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
except ImportError:
    warnings.warn("google-generativeai library not found. Google models will not be available. "
                  "Install with: pip install google-generativeai")
    genai = None

# --- Custom Exceptions (Unchanged) ---
class LLMClientError(Exception):
    """Custom exception for UniversalLLMClient errors."""
    pass

class MissingAPIKeyError(LLMClientError):
    """Raised when an API key for a required provider is missing."""
    pass

class InvalidModelError(LLMClientError):
    """Raised when the model string or config nickname is invalid."""
    pass

class APIRequestError(LLMClientError):
    """Raised when an underlying API call fails."""
    pass

class ConfigError(LLMClientError): # New exception for config issues
    """Raised for errors related to loading or parsing the config file."""
    pass


# --- The Universal Client (Updated) ---
class UniversalLLMClient:
    """
    A universal client wrapper for OpenAI, Anthropic, and Google LLMs.

    Acts as an orchestrator, providing a single interface to generate text
    using different models from various providers. Can use predefined model
    configurations from a JSON file or direct provider/model strings.

    Relies on environment variables for API keys by default:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY

    Alternatively, keys can be passed during initialization.
    """

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "google"]
    # Default config file path
    DEFAULT_CONFIG_PATH = "llm_config.json"
    # Default parameters used if not specified in config or generate() call
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1024


    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config_filepath: Optional[str] = DEFAULT_CONFIG_PATH,
        google_safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
    ):
        """
        Initializes the client, loads configurations, and configures API access.

        Args:
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            anthropic_api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            google_api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            config_filepath: Path to the JSON configuration file for model shortcuts.
                             Set to None to disable config loading.
            google_safety_settings: Optional safety settings for Google Gemini models.
        """
        self._openai_client = None
        self._anthropic_client = None
        self._google_clients: Dict[str, Any] = {}
        self._google_configured = False
        self._google_safety_settings = google_safety_settings # Store safety settings

        # --- Load Model Configurations ---
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        if config_filepath:
            try:
                # Use utf-8 encoding for broader compatibility
                with open(config_filepath, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    if "models" in loaded_config and isinstance(loaded_config["models"], dict):
                         self._model_configs = loaded_config["models"]
                         print(f"Loaded model configurations from {config_filepath}")
                    else:
                         warnings.warn(f"Config file {config_filepath} loaded but missing "
                                       f"valid 'models' dictionary at the top level.")

            except FileNotFoundError:
                # Only warn if the default path wasn't found, raise if a specific path was given
                if config_filepath == self.DEFAULT_CONFIG_PATH:
                     warnings.warn(f"Default config file not found at {config_filepath}. "
                                   f"Only direct 'provider/model' strings can be used.")
                else:
                     raise ConfigError(f"Specified config file not found: {config_filepath}")
            except json.JSONDecodeError as e:
                raise ConfigError(f"Error decoding JSON from config file {config_filepath}: {e}")
            except Exception as e:
                 raise ConfigError(f"An unexpected error occurred loading config file {config_filepath}: {e}")


        # --- Configure Provider Clients (Same as before) ---
        if openai:
            _openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if _openai_key:
                try:
                    self._openai_client = openai.OpenAI(api_key=_openai_key)
                except Exception as e:
                    warnings.warn(f"Failed to initialize OpenAI client: {e}")

        if anthropic:
            _anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if _anthropic_key:
                try:
                    self._anthropic_client = anthropic.Anthropic(api_key=_anthropic_key)
                except Exception as e:
                    warnings.warn(f"Failed to initialize Anthropic client: {e}")

        if genai:
            _google_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
            if _google_key:
                try:
                    genai.configure(api_key=_google_key)
                    self._google_configured = True
                    # Apply default safety settings if not provided
                    if self._google_safety_settings is None:
                        self._google_safety_settings = {
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        }
                except Exception as e:
                    warnings.warn(f"Failed to configure Google Generative AI: {e}")
                    self._google_configured = False
            else:
                self._google_configured = False


    def _parse_model_string(self, model_string: str) -> Tuple[str, str]:
        """Splits 'provider/model_name' into (provider, model_name)."""
        try:
            provider, model_name = model_string.lower().split('/', 1)
            if provider not in self.SUPPORTED_PROVIDERS:
                raise ValueError(f"Unsupported provider: {provider}")
            return provider, model_name
        except ValueError:
            raise InvalidModelError(
                f"Invalid model format: '{model_string}'. Expected format: 'provider/model_name' "
                f"(e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus-20240229', 'google/gemini-pro')."
            )

    def _get_google_model(self, model_name: str) -> Any:
        """Initializes and caches a Google GenerativeModel."""
        if not self._google_configured:
            raise MissingAPIKeyError(
                "Google API key not configured. Provide GOOGLE_API_KEY env var or google_api_key."
            )
        # Use models/ prefix convention for Gemini API
        api_model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"

        if model_name not in self._google_clients: # Cache using original name for consistency
            try:
                self._google_clients[model_name] = genai.GenerativeModel(api_model_name)
            except Exception as e:
                # Catch potential errors during model initialization (e.g., invalid name)
                raise APIRequestError(f"Failed to initialize Google model '{model_name}' (API name: '{api_model_name}'): {e}")
        return self._google_clients[model_name]

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        # Make defaults None to easily check if they were explicitly passed
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        **kwargs # For other provider-specific arguments
    ) -> str:
        """
        Generates text using the specified model and parameters.

        The `model` parameter can be a configuration nickname defined in the
        JSON config file (e.g., 'default_gpt') or a direct string specifying
        the provider and model (e.g., 'openai/gpt-4o').

        Parameters explicitly passed to this method (like `temperature`, `max_tokens`)
        will override values defined in the configuration file for that specific call.

        Args:
            model: The model identifier (config nickname or 'provider/model_name').
            messages: A list of message dictionaries.
            system_prompt: An optional system prompt.
            max_tokens: Max tokens to generate (overrides config if provided).
            temperature: Sampling temperature (overrides config if provided).
            stop_sequences: Optional list of strings to stop generation.
            top_p: Optional nucleus sampling parameter.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The generated text content as a string.

        Raises:
            InvalidModelError: If the model nickname/string is invalid or provider unsupported.
            ConfigError: If a nickname is used but the config is invalid.
            MissingAPIKeyError: If the API key for the required provider is missing.
            APIRequestError: If the API call to the provider fails.
            LLMClientError: For other client-related issues.
        """
        provider: Optional[str] = None
        model_name: Optional[str] = None
        config_params: Dict[str, Any] = {}

        # 1. Check if 'model' is a nickname in the loaded config
        if model in self._model_configs:
            config = self._model_configs[model]
            provider = config.get("provider")
            model_name = config.get("model_name")

            if not provider or not model_name:
                raise ConfigError(
                    f"Configuration nickname '{model}' is missing required 'provider' or 'model_name' "
                    f"in file '{self.config_filepath}'."
                )
            if provider not in self.SUPPORTED_PROVIDERS:
                 raise ConfigError(f"Unsupported provider '{provider}' found in config for nickname '{model}'.")

            # Load parameters from config, using class defaults as fallback
            config_params["temperature"] = config.get("temperature", self.DEFAULT_TEMPERATURE)
            config_params["max_tokens"] = config.get("max_tokens", self.DEFAULT_MAX_TOKENS)
            # Add other potential config params here if needed in the future
            print(f"Using config '{model}': provider={provider}, model={model_name}, params={config_params}")

        else:
            # 2. Treat 'model' as a 'provider/model_name' string
            try:
                provider, model_name = self._parse_model_string(model)
                 # Use class defaults when no config is used
                config_params["temperature"] = self.DEFAULT_TEMPERATURE
                config_params["max_tokens"] = self.DEFAULT_MAX_TOKENS
                print(f"Using direct model string: provider={provider}, model={model_name}")
            except InvalidModelError:
                 # If it wasn't found in config and isn't a valid string format
                 raise InvalidModelError(
                     f"Model identifier '{model}' is not a valid configuration nickname "
                     f"found in '{getattr(self, 'config_filepath', 'N/A')}' and is not a valid 'provider/model_name' string."
                 )

        # 3. Determine final parameters, applying overrides from method arguments
        final_temperature = temperature if temperature is not None else config_params["temperature"]
        final_max_tokens = max_tokens if max_tokens is not None else config_params["max_tokens"]
        final_stop_sequences = stop_sequences # Pass through directly
        final_top_p = top_p # Pass through directly

        # Filter out known params from kwargs to avoid duplication if passed via kwargs too
        provider_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'stop_sequences', 'top_p']}


        # 4. Call the appropriate provider method
        try:
            if provider == "openai":
                if not self._openai_client:
                    raise MissingAPIKeyError("OpenAI client not initialized. Check API key or installation.")
                return self._generate_openai(
                    model_name, messages, system_prompt,
                    final_max_tokens, final_temperature, final_stop_sequences, final_top_p,
                    **provider_kwargs
                )
            elif provider == "anthropic":
                if not self._anthropic_client:
                    raise MissingAPIKeyError("Anthropic client not initialized. Check API key or installation.")
                # Anthropic requires max_tokens, ensure it has a value
                if final_max_tokens is None:
                    final_max_tokens = self.DEFAULT_MAX_TOKENS # Fallback if still None somehow
                    warnings.warn(f"max_tokens was not specified for Anthropic model '{model_name}', using default {final_max_tokens}.")

                return self._generate_anthropic(
                    model_name, messages, system_prompt,
                    final_max_tokens, final_temperature, final_stop_sequences,
                    **provider_kwargs
                )
            elif provider == "google":
                 # Google client/model is fetched/initialized on demand
                 return self._generate_google(
                     model_name, messages, system_prompt,
                     final_max_tokens, final_temperature, final_stop_sequences, final_top_p,
                     **provider_kwargs
                 )
            else:
                # Should be caught earlier, but defensive check
                raise InvalidModelError(f"Provider '{provider}' selection failed unexpectedly.")

        # --- Error Handling for API Calls (largely unchanged, added context) ---
        except (openai.APIError, anthropic.APIError, Exception) as e:
             # Catch specific API errors and general exceptions during the call
             err_context = f"provider={provider}, model={model_name}"
             if isinstance(e, (openai.AuthenticationError, anthropic.AuthenticationError)):
                 raise MissingAPIKeyError(f"Authentication failed for {err_context}: {e}")
             elif isinstance(e, (openai.RateLimitError, anthropic.RateLimitError)):
                 raise APIRequestError(f"Rate limit exceeded for {err_context}: {e}")
             elif isinstance(e, (openai.NotFoundError, anthropic.NotFoundError)):
                 raise InvalidModelError(f"Model '{model_name}' not found or invalid for {provider}: {e}")
             # Add more specific error handling as needed (e.g., Google API errors if distinct types exist)
             else:
                 # General API or unexpected error during generation
                 raise APIRequestError(f"Error during API call for {err_context}: {e}")
        except LLMClientError:
             # Re-raise our custom errors directly
             raise
        except Exception as e:
            # Catch any other unexpected errors
            raise LLMClientError(f"An unexpected error occurred: {e}")


    # --- Provider Specific Methods (_generate_openai, _generate_anthropic, _generate_google) ---
    # (These methods remain largely the same as in the previous version, ensuring they accept
    # the necessary parameters like max_tokens, temperature, stop_sequences, top_p, **kwargs)

    def _generate_openai(self, model_name: str, messages: List[Dict[str, str]], system_prompt: Optional[str], max_tokens: Optional[int], temperature: Optional[float], stop_sequences: Optional[List[str]], top_p: Optional[float], **kwargs) -> str:
        """Handles OpenAI API call."""
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user").lower()
            if role not in ["user", "assistant", "system", "tool"]:
                 warnings.warn(f"Mapping unknown role '{role}' to 'user' for OpenAI.")
                 role = "user"
            openai_messages.append({"role": role, "content": msg.get("content", "")})

        # Prepare arguments, filtering out None values unless the API accepts them
        api_args = {
            "model": model_name,
            "messages": openai_messages,
        }
        if max_tokens is not None: api_args["max_tokens"] = max_tokens
        if temperature is not None: api_args["temperature"] = temperature
        if stop_sequences is not None: api_args["stop"] = stop_sequences
        if top_p is not None: api_args["top_p"] = top_p
        api_args.update(kwargs) # Add remaining specific kwargs

        response = self._openai_client.chat.completions.create(**api_args)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _generate_anthropic(self, model_name: str, messages: List[Dict[str, str]], system_prompt: Optional[str], max_tokens: int, temperature: Optional[float], stop_sequences: Optional[List[str]], **kwargs) -> str:
        """Handles Anthropic API call. max_tokens is required."""
        anthropic_messages = []
        valid_roles = {"user", "assistant"}
        for msg in messages:
            role = msg.get("role", "user").lower()
            if role not in valid_roles:
                warnings.warn(f"Mapping unknown role '{role}' to 'user' for Anthropic.")
                role = "user"
            anthropic_messages.append({"role": role, "content": msg.get("content", "")})

        # Prepare arguments, filtering out None values unless the API accepts them
        api_args = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens, # Required by Anthropic
        }
        if system_prompt is not None: api_args["system"] = system_prompt
        if temperature is not None: api_args["temperature"] = temperature
        if stop_sequences is not None: api_args["stop_sequences"] = stop_sequences
        # Add other Anthropic specific params like top_p, top_k if needed from kwargs
        api_args.update(kwargs)

        response = self._anthropic_client.messages.create(**api_args)

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                    # break # Assuming first text block is sufficient? Or concat all? Let's concat.

        return content.strip()


    def _generate_google(self, model_name: str, messages: List[Dict[str, str]], system_prompt: Optional[str], max_tokens: Optional[int], temperature: Optional[float], stop_sequences: Optional[List[str]], top_p: Optional[float], **kwargs) -> str:
        """Handles Google Gemini API call."""
        google_model = self._get_google_model(model_name)

        google_contents = []
        processed_system_prompt = False # Track if system prompt was prepended

        # Process messages, prepend system prompt if needed
        for i, msg in enumerate(messages):
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            google_role = "user" # Default

            if role == "assistant":
                google_role = "model"
            elif role == "user":
                google_role = "user"
                if system_prompt and not processed_system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                    processed_system_prompt = True
            elif role == "system":
                 # Handle system message in list: prepend to next user msg or warn/ignore
                 warnings.warn(f"System role in messages list for Google Gemini is handled by prepending to the next user message. Use 'system_prompt' argument for clarity.", UserWarning)
                 if i + 1 < len(messages) and messages[i+1].get("role", "user").lower() == "user":
                     messages[i+1]["content"] = f"{content}\n\n{messages[i+1].get('content', '')}"
                 elif not processed_system_prompt: # If no user message follows, treat this as the system prompt
                      system_prompt = content # Override/set system prompt from this message
                      processed_system_prompt = True # Mark as handled
                 continue # Skip adding this system message directly to contents
            else:
                warnings.warn(f"Mapping unknown role '{role}' to 'user' for Google Gemini.")
                google_role = "user"
                # Also prepend system prompt if needed for mapped roles
                if system_prompt and not processed_system_prompt and google_role == "user":
                    content = f"{system_prompt}\n\n{content}"
                    processed_system_prompt = True

            google_contents.append({"role": google_role, "parts": [{"text": content}]})

        # Handle case where only a system_prompt is given with no user messages
        if not google_contents and system_prompt:
             google_contents.append({"role": "user", "parts": [{"text": system_prompt}]})
             processed_system_prompt = True


        # Configure generation parameters, filtering Nones
        generation_config_args = {}
        if max_tokens is not None: generation_config_args["max_output_tokens"] = max_tokens
        if temperature is not None: generation_config_args["temperature"] = temperature
        if stop_sequences is not None: generation_config_args["stop_sequences"] = stop_sequences
        if top_p is not None: generation_config_args["top_p"] = top_p
        # top_k could also be added here from kwargs if needed
        generation_config_args.update(kwargs) # Add other specific kwargs

        generation_config = GenerationConfig(**generation_config_args)

        # Call the API
        try:
            response = google_model.generate_content(
                contents=google_contents,
                generation_config=generation_config,
                safety_settings=self._google_safety_settings,
                # **kwargs # kwargs are now merged into generation_config
            )
        except Exception as e:
             # Catch potential google-specific API call errors here
             raise APIRequestError(f"Error during Google Gemini API call for model '{model_name}': {e}")


        # Extract text (same logic as before)
        try:
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif response.candidates and response.candidates[0].content.parts:
                 return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            else:
                 finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'NO_CANDIDATES'
                 safety_ratings = getattr(response.candidates[0], 'safety_ratings', []) if response.candidates else []
                 # Provide more context if blocked or empty
                 if finish_reason != "STOP":
                     warnings.warn(f"Google Gemini generation for '{model_name}' finished unexpectedly (Reason: {finish_reason}). Safety Ratings: {safety_ratings}")
                 elif not hasattr(response, 'text') or not response.text:
                      warnings.warn(f"Google Gemini generation for '{model_name}' resulted in no text content (Reason: {finish_reason}).")
                 return ""
        except (AttributeError, IndexError, StopIteration, TypeError) as e:
             warnings.warn(f"Could not extract text from Google Gemini response structure for model '{model_name}'. Error: {e}. Response: {response}")
             return ""
        except Exception as e:
             if "invalid start byte" in str(e) and hasattr(response, 'prompt_feedback'):
                 warnings.warn(f"Could not extract text from Google Gemini response for model '{model_name}' due to feedback processing error: {e}. Returning empty string.")
                 return ""
             else:
                 raise APIRequestError(f"Error processing Google Gemini response for model '{model_name}': {e}") from e


# --- Example Usage (Updated) ---
if __name__ == "__main__":
    # --- Setup: Ensure llm_config.json exists ---
    CONFIG_FILE = "llm_config.json"
    if not os.path.exists(CONFIG_FILE):
        print(f"Creating sample config file: {CONFIG_FILE}")
        sample_config = {
          "models": {
            "default_gpt": {
              "provider": "openai", "model_name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 512
            },
            "creative_claude": {
              "provider": "anthropic", "model_name": "claude-3-haiku-20240307", "temperature": 0.9, "max_tokens": 1024
            },
            "pro_gemini": {
                "provider": "google", "model_name": "gemini-1.5-pro-latest", "temperature": 0.6, "max_tokens": 2048
            },
             "fast_gemini": {
                "provider": "google", "model_name": "gemini-1.5-flash-latest" # Uses default temp/tokens
             },
            "concise_gpt4o": {
              "provider": "openai", "model_name": "gpt-4o", "temperature": 0.3, "max_tokens": 256
            }
          }
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2)
        except Exception as e:
            print(f"Error creating sample config file: {e}")
            # Continue without config if creation fails

    # Load .env file if available
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("Loaded environment variables from .env file.")
        else:
             print("No .env file found or it is empty, relying on system environment variables.")
    except ImportError:
        print("dotenv library not found, relying on system environment variables.")
        pass

    # --- Initialize the client ---
    try:
        # It will try to load keys from env vars and config from llm_config.json
        client = UniversalLLMClient(config_filepath=CONFIG_FILE)
        # Example: Initialize without config: client = UniversalLLMClient(config_filepath=None)
        print("UniversalLLMClient initialized.")
    except (LLMClientError, ConfigError) as e:
        print(f"Error initializing client: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit(1)


    # --- Test Cases ---

    # 1. Using a config nickname ('concise_gpt4o')
    print("\n--- Test 1: Using config nickname 'concise_gpt4o' ---")
    if client._openai_client: # Check if provider is available
        try:
            response = client.generate(
                model="concise_gpt4o", # Nickname from llm_config.json
                messages=[{"role": "user", "content": "Explain the difference between a list and a tuple in Python."}],
                system_prompt="Be extremely concise."
                # Uses temp=0.3, max_tokens=256 from config
            )
            print(f"Response:\n{response}")
        except (LLMClientError, ConfigError) as e: print(f"Error: {e}")
        except Exception as e: print(f"Unexpected Error: {e}")
    else: print("Skipping (OpenAI client not available)")

    # 2. Using a config nickname and overriding a parameter
    print("\n--- Test 2: Using config nickname 'default_gpt' & overriding temperature ---")
    if client._openai_client:
        try:
            response = client.generate(
                model="default_gpt", # Nickname from llm_config.json
                messages=[{"role": "user", "content": "Suggest three creative names for a new coffee shop."}],
                temperature=0.95 # Override config temp (0.7)
                # Uses max_tokens=512 from config
            )
            print(f"Response:\n{response}")
        except (LLMClientError, ConfigError) as e: print(f"Error: {e}")
        except Exception as e: print(f"Unexpected Error: {e}")
    else: print("Skipping (OpenAI client not available)")

    # 3. Using a direct provider/model string (should still work)
    print("\n--- Test 3: Using direct 'anthropic/claude-3-sonnet-20240229' string ---")
    if client._anthropic_client:
         try:
            response = client.generate(
                model="anthropic/claude-3-sonnet-20240229", # Direct string
                messages=[{"role": "user", "content": "Write a SQL query to find users who signed up in the last 7 days."}],
                system_prompt="Assume a table named 'users' with 'user_id' and 'signup_date' (timestamp).",
                max_tokens=150, # Explicitly provided
                temperature=0.5 # Explicitly provided
            )
            print(f"Response:\n{response}")
         except (LLMClientError, ConfigError) as e: print(f"Error: {e}")
         except Exception as e: print(f"Unexpected Error: {e}")
    else: print("Skipping (Anthropic client not available)")

    # 4. Using a config nickname for Google Gemini
    print("\n--- Test 4: Using config nickname 'fast_gemini' ---")
    if client._google_configured:
        try:
            response = client.generate(
                model="fast_gemini", # Nickname from llm_config.json
                messages=[{"role": "user", "content": "What is Oshkosh, Wisconsin known for?"}],
                # Uses model gemini-1.5-flash-latest from config
                # Uses temp=0.75 from config
                # Uses default max_tokens=1024 (since not in config)
            )
            print(f"Response:\n{response}")
        except (LLMClientError, ConfigError) as e: print(f"Error: {e}")
        except Exception as e: print(f"Unexpected Error: {e}")
    else: print("Skipping (Google client not available)")

    # 5. Test case where nickname doesn't exist
    print("\n--- Test 5: Using non-existent nickname 'bad_nickname' ---")
    try:
        response = client.generate(
            model="bad_nickname",
            messages=[{"role": "user", "content": "Test"}]
        )
        print(f"Response:\n{response}")
    except InvalidModelError as e:
        print(f"Caught expected error: {e}") # Expect InvalidModelError
    except (LLMClientError, ConfigError) as e: print(f"Error: {e}")
    except Exception as e: print(f"Unexpected Error: {e}")