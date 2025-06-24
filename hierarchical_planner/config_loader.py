"""
Configuration loading module for the Hierarchical Planner.

Handles loading settings from a YAML file, merging with defaults,
resolving the API key, and making file paths absolute.
"""
import yaml
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any, Optional

# Local imports for exceptions
from exceptions import ConfigError, ConfigNotFoundError, ConfigParsingError, ApiKeyError

# Load .env file if present (for API key primarily)
# Load from the same directory as this script
script_dir = os.path.dirname(__file__)
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path)

# Get logger for this module
logger = logging.getLogger(__name__)

#: Default configuration values.
DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    'api': {
        'key': 'GEMINI_API_KEY', # Default to checking this env var
        'model_name': 'gemini-2.5-pro-exp-03-25',
        'temperature': 0.7,
        'max_output_tokens': 8192,
        'retries': 3
    },
    'anthropic': {
        'api_key': 'ANTHROPIC_API_KEY', # Default to checking this env var
        'model_name': 'claude-3-7-sonnet-20250219',
        'temperature': 0.7,
        'max_tokens': 8192,
        'retries': 3,
        'extended_thinking': True,
        'thinking_min_tokens': 1024,
        'thinking_max_tokens': 8192
    },
    'deepseek': {
        'api_key': 'DEEPSEEK_API_KEY', # Default to checking this env var
        'base_url': 'https://api.deepseek.com/v1',
        'model_name': 'DeepSeek-V3-0324',
        'temperature': 0.6,
        'max_tokens': 8192,
        'top_p': 1.0
    },
    'files': {
        'default_task': 'task.txt',
        'default_output': 'reasoning_tree.json',
        'default_validated_output': 'reasoning_tree_validated.json'
    },
    'logging': {
        'level': 'INFO',
        'log_file': 'logs/planner.log',
        'log_to_console': True
    }
}

# Custom exceptions are now defined in exceptions.py

def _resolve_api_key(key_setting: Optional[str], default_env_var: str) -> Optional[str]:
    """
    Resolves an API key based on the config setting.

    Resolution logic:
    1. If key_setting is None or empty, check the default_env_var.
    2. If key_setting looks like a direct key (heuristic: >20 chars, no spaces),
       check if an env var with that name exists (prefer env var), otherwise use the setting directly.
    3. Assume key_setting is an env var name and check it.
    4. As a final fallback, check default_env_var again.

    Args:
        key_setting: The value of the api key from the configuration.
        default_env_var: The default environment variable name to check.

    Returns:
        The resolved API key string, or None if resolution fails.
    """
    if not key_setting:
        # If key setting is null or empty, rely solely on direct env var check
        return os.getenv(default_env_var)

    # Check if the setting itself is the key (basic check: contains no spaces and is long enough)
    # This is a heuristic and might need refinement. Assume keys don't look like typical env var names.
    if ' ' not in key_setting and len(key_setting) > 20: # Arbitrary length check
         # Check if it's *also* an env var name, prefer env var if it exists
        env_value = os.getenv(key_setting)
        if env_value:
            return env_value
        # Otherwise, assume the setting *is* the key
        return key_setting

    # Assume the setting is an environment variable name
    api_key = os.getenv(key_setting)
    if not api_key:
        # As a fallback, also check the default env var
        api_key = os.getenv(default_env_var)

    return api_key

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, merges with defaults, resolves the API key,
    and makes relevant file paths absolute based on the location of this module.

    Args:
        config_path: Path to the YAML configuration file, expected to be relative
                     to the directory containing this `config_loader.py` module.
                     Defaults to '../config/config.yaml'.

    Returns:
        A dictionary containing the loaded and processed configuration.

    Raises:
        ConfigNotFoundError: If the config file is specified but not found.
        ConfigParsingError: If the config file cannot be parsed.
        ApiKeyError: If the API key cannot be resolved.
        ConfigError: For other configuration-related issues.
    """
    config = DEFAULT_CONFIG.copy() # Start with defaults

    # Determine the absolute path to the config file relative to this script's location
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, config_path)

    try:
        if os.path.exists(abs_config_path):
            with open(abs_config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Deep merge user config into defaults (simple merge for now)
                    for section, settings in user_config.items():
                        if section in config and isinstance(config[section], dict):
                            config[section].update(settings)
                        else:
                            config[section] = settings
        else:
            # Config file is optional, proceed with defaults if not found
            logger.warning(f"Configuration file '{abs_config_path}' not found. Using default settings.")
            # No exception raised if the default path doesn't exist

    except yaml.YAMLError as e:
        raise ConfigParsingError(f"Error parsing configuration file '{abs_config_path}': {e}") from e
    except IOError as e:
        # This might indicate a permissions issue if the file exists but can't be read
        raise ConfigError(f"Error reading configuration file '{abs_config_path}': {e}") from e
    except Exception as e:
        # Catch unexpected errors during loading
        raise ConfigError(f"Unexpected error loading configuration file '{abs_config_path}': {e}") from e


    # Resolve the Gemini API key
    api_key_setting = config.get('api', {}).get('key')
    resolved_api_key = _resolve_api_key(api_key_setting, 'GEMINI_API_KEY')

    if not resolved_api_key:
        # If still no key, raise a specific ApiKeyError
        raise ApiKeyError(
            "GEMINI_API_KEY could not be resolved. "
            "Ensure it's set in the environment (e.g., via .env file), "
            "directly in config.yaml, or referenced by name via the 'api.key' setting."
        )

    # Store the resolved key back into the config dict for easy access
    config['api']['resolved_key'] = resolved_api_key
    # Avoid logging the key itself unless debugging
    logger.info("Gemini API key resolved successfully.")

    # Resolve the Anthropic API key if it exists in config
    if 'anthropic' in config:
        anthropic_key_setting = config.get('anthropic', {}).get('api_key')
        resolved_anthropic_key = _resolve_api_key(anthropic_key_setting, 'ANTHROPIC_API_KEY')
        
        # Store the resolved key back into the config dict
        if resolved_anthropic_key:
            config['anthropic']['api_key'] = resolved_anthropic_key
            logger.info("Anthropic API key resolved successfully.")
        else:
            logger.warning("Anthropic API key could not be resolved. Anthropic client will not be available.")
    
    # Resolve the DeepSeek API key if it exists in config
    if 'deepseek' in config:
        deepseek_key_setting = config.get('deepseek', {}).get('api_key')
        resolved_deepseek_key = _resolve_api_key(deepseek_key_setting, 'DEEPSEEK_API_KEY')
        
        # Store the resolved key back into the config dict
        if resolved_deepseek_key:
            config['deepseek']['resolved_key'] = resolved_deepseek_key
            logger.info("DeepSeek API key resolved successfully.")
        else:
            logger.warning("DeepSeek API key could not be resolved. Fallback to DeepSeek will not be available.")

    # Resolve relative file paths (task, output, log) based on the script directory
    # This assumes the paths in config.yaml are relative to the `hierarchical_planner` dir
    for key in ['default_task', 'default_output', 'default_validated_output']:
        if key in config['files']:
            config['files'][key] = os.path.join(script_dir, config['files'][key])

    if 'log_file' in config['logging']:
        config['logging']['log_file'] = os.path.join(script_dir, config['logging']['log_file'])


    return config

# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Assumes config.yaml is in ../config relative to this file
        loaded_config = load_config('../config/config.yaml')
        import json
        print("Configuration loaded successfully:")
        print(json.dumps(loaded_config, indent=2))

        # Test API key resolution specifically
        print("\nTesting API Key Resolution:")
        print(f"Resolved Gemini Key: {loaded_config.get('api', {}).get('resolved_key')}")
        print(f"Resolved Anthropic Key: {loaded_config.get('anthropic', {}).get('api_key')}")
        print(f"Resolved DeepSeek Key: {loaded_config.get('deepseek', {}).get('resolved_key')}")

    except ConfigError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        # Use logger if available, otherwise print
        logger.error(f"An unexpected error occurred during config load test: {e}", exc_info=True)
