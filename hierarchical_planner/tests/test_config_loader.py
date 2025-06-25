import pytest
import os
import yaml
from unittest.mock import patch, mock_open

# Module to test
from .. import config_loader
from ..exceptions import ConfigError, ConfigNotFoundError, ConfigParsingError, ApiKeyError

# --- Test Fixtures ---

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables."""
    monkeypatch.setenv("TEST_API_KEY_FROM_ENV", "env_key_123")
    monkeypatch.setenv("GEMINI_API_KEY", "default_env_key_456") # Default fallback
    # Ensure no real key interferes if set locally
    monkeypatch.delenv("REAL_API_KEY_VAR", raising=False)

@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture to create temporary config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    def _create_file(content, filename="config.yaml"):
        file_path = config_dir / filename
        file_path.write_text(yaml.dump(content), encoding='utf-8')
        # Return path relative to tmp_path for use in tests
        return os.path.join("config", filename)
    return _create_file

# --- Test Cases ---

def test_load_config_defaults_no_file(mock_env_vars, tmp_path):
    """Test loading defaults when no config file exists, using default env var."""
    # Change CWD to tmp_path to simulate running from project root
    os.chdir(tmp_path)
    # Create the hierarchical_planner directory structure expected by the loader
    (tmp_path / "hierarchical_planner").mkdir()

    # Update to expect the correct default model name
    config = config_loader.load_config('config/non_existent.yaml') # Non-existent path

    assert config['api']['model_name'] == 'gemini-2.5-pro-exp-03-25'
    assert config['files']['default_task'].endswith('task.txt') # Check relative part
    assert config['logging']['level'] == 'INFO'
    assert config['api']['resolved_key'] == "default_env_key_456" # From mock_env_vars

def test_load_config_valid_file(mock_env_vars, temp_config_file, tmp_path):
    """Test loading a valid config file, overriding defaults."""
    os.chdir(tmp_path)
    planner_dir = tmp_path / "hierarchical_planner"
    planner_dir.mkdir()

    config_content = {
        'api': {'model_name': 'gemini-ultra', 'key': 'TEST_API_KEY_FROM_ENV', 'retries': 5},
        'logging': {'level': 'DEBUG'}
    }
    relative_config_path = temp_config_file(config_content) # Creates config/config.yaml
    
    # Mock the config loader to use our config file path
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content), \
         patch('os.environ.get', side_effect=lambda key, default=None: 
               "env_key_123" if key == "TEST_API_KEY_FROM_ENV" else 
               "default_env_key_456" if key == "GEMINI_API_KEY" else default):
        
        # Pass path relative to hierarchical_planner dir
        config = config_loader.load_config(f"../{relative_config_path}")

        assert config['api']['model_name'] == 'gemini-ultra'
        assert config['logging']['level'] == 'DEBUG'
        assert config['api']['retries'] == 5
        assert config['api']['resolved_key'] == "env_key_123" # Resolved from env var name in config
        assert config['files']['default_output'].endswith('reasoning_tree.json') # Default preserved


def test_load_config_direct_key(mock_env_vars, temp_config_file, tmp_path):
    """Test loading config with API key directly in the file."""
    os.chdir(tmp_path)
    planner_dir = tmp_path / "hierarchical_planner"
    planner_dir.mkdir()
    
    direct_key = "abc123xyz789this_is_a_direct_key_longer_than_20_chars"
    config_content = {'api': {'key': direct_key}}
    relative_config_path = temp_config_file(config_content)

    # Mock the config loader to use our config file path
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content):
        
        config = config_loader.load_config(f"../{relative_config_path}")
        assert config['api']['resolved_key'] == direct_key


def test_load_config_direct_key_also_env_var(mock_env_vars, temp_config_file, tmp_path):
    """Test direct key in config when an env var with the same name exists (prefer env var)."""
    os.chdir(tmp_path)
    planner_dir = tmp_path / "hierarchical_planner"
    planner_dir.mkdir()
    
    # Key looks like an env var name, and the env var exists
    config_content = {'api': {'key': 'TEST_API_KEY_FROM_ENV'}}
    relative_config_path = temp_config_file(config_content)

    # Mock the config loader to use our config file path
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content), \
         patch('os.environ.get', side_effect=lambda key, default=None: 
               "env_key_123" if key == "TEST_API_KEY_FROM_ENV" else 
               "default_env_key_456" if key == "GEMINI_API_KEY" else default):
        
        config = config_loader.load_config(f"../{relative_config_path}")
        assert config['api']['resolved_key'] == "env_key_123" # Prefers env var value


def test_load_config_invalid_yaml(mock_env_vars, temp_config_file, tmp_path):
    """Test loading a file with invalid YAML content."""
    os.chdir(tmp_path)
    planner_dir = tmp_path / "hierarchical_planner"
    planner_dir.mkdir()
    
    # Create an invalid YAML file
    invalid_yaml = "api: { model: 'test', key: "
    
    # Mock the config loader to use our invalid YAML
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=invalid_yaml)), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError("YAML syntax error")):
        
        with pytest.raises(ConfigParsingError):
            config_loader.load_config('../config/invalid.yaml')

@patch('builtins.open', side_effect=IOError("Permission denied"))
@patch('os.path.exists', return_value=True)
def test_load_config_read_error(mock_exists, mock_open, mock_env_vars, tmp_path):
    """Test handling of IOErrors during file reading."""
    os.chdir(tmp_path)
    (tmp_path / "hierarchical_planner").mkdir()

    with pytest.raises(ConfigError, match="Error reading configuration file"):
        config_loader.load_config('../config/config.yaml')

def test_load_config_no_api_key(monkeypatch, temp_config_file, tmp_path):
    """Test error when API key cannot be resolved."""
    os.chdir(tmp_path)
    (tmp_path / "hierarchical_planner").mkdir()
    # Remove all potential env vars
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TEST_API_KEY_FROM_ENV", raising=False)

    # Config file specifies an env var that doesn't exist
    config_content = {'api': {'key': 'NON_EXISTENT_KEY_VAR'}}
    relative_config_path = temp_config_file(config_content)

    # Use appropriate mocks to ensure the ApiKeyError is raised
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content), \
         patch('os.environ.get', return_value=None):  # Ensure all env vars return None
        
        with pytest.raises(ApiKeyError):
            config_loader.load_config(f"../{relative_config_path}")

def test_load_config_empty_key_setting(mock_env_vars, temp_config_file, tmp_path):
    """Test empty 'key' setting relies on default GEMINI_API_KEY env var."""
    os.chdir(tmp_path)
    (tmp_path / "hierarchical_planner").mkdir()
    config_content = {'api': {'key': ''}} # Empty key setting
    relative_config_path = temp_config_file(config_content)

    # Mock to ensure we get the expected behavior
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content), \
         patch('os.environ.get', side_effect=lambda key, default=None: 
               "env_key_123" if key == "TEST_API_KEY_FROM_ENV" else 
               "default_env_key_456" if key == "GEMINI_API_KEY" else default):
        
        config = config_loader.load_config(f"../{relative_config_path}")
        assert config['api']['resolved_key'] == "default_env_key_456" # Falls back to default env var

def test_load_config_null_key_setting(mock_env_vars, temp_config_file, tmp_path):
    """Test null 'key' setting relies on default GEMINI_API_KEY env var."""
    os.chdir(tmp_path)
    (tmp_path / "hierarchical_planner").mkdir()
    config_content = {'api': {'key': None}} # Null key setting
    relative_config_path = temp_config_file(config_content)

    # Mock to ensure we get the expected behavior
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content), \
         patch('os.environ.get', side_effect=lambda key, default=None: 
               "env_key_123" if key == "TEST_API_KEY_FROM_ENV" else 
               "default_env_key_456" if key == "GEMINI_API_KEY" else default):
        
        config = config_loader.load_config(f"../{relative_config_path}")
        assert config['api']['resolved_key'] == "default_env_key_456" # Falls back to default env var

def test_file_path_resolution(mock_env_vars, temp_config_file, tmp_path):
    """Test that file paths are resolved correctly relative to the module."""
    os.chdir(tmp_path) # Simulate running from project root
    planner_dir = tmp_path / "hierarchical_planner"
    planner_dir.mkdir()

    config_content = {
        'files': {'default_task': 'custom_task.txt'},
        'logging': {'log_file': 'logs/custom.log'}
    }
    relative_config_path = temp_config_file(config_content)

    # Mock the script_dir to return our test directory path
    with patch('os.path.dirname', return_value=str(planner_dir)), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml.dump(config_content))), \
         patch('yaml.safe_load', return_value=config_content):
        
        config = config_loader.load_config(f"../{relative_config_path}")

        # Check if the paths are absolute and end correctly
        assert os.path.isabs(config['files']['default_task'])
        
        # Compare paths in a platform-independent way
        expected_task_path = os.path.join('hierarchical_planner', 'custom_task.txt')
        assert os.path.normpath(config['files']['default_task']).endswith(os.path.normpath(expected_task_path))
        
        assert os.path.isabs(config['files']['default_output'])
        expected_output_path = os.path.join('hierarchical_planner', 'reasoning_tree.json') 
        assert os.path.normpath(config['files']['default_output']).endswith(os.path.normpath(expected_output_path))
        
        assert os.path.isabs(config['logging']['log_file'])
        expected_log_path = os.path.join('hierarchical_planner', 'logs', 'custom.log')
        assert os.path.normpath(config['logging']['log_file']).endswith(os.path.normpath(expected_log_path))
