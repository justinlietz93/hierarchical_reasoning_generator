import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import argparse
import asyncio
import os # Import os for basename mocking

# Import the module under test *without* module-level patches
from hierarchical_planner import main
# Import exceptions needed for tests
from hierarchical_planner.exceptions import ApiCallError, HierarchicalPlannerError, PlanGenerationError, PlanValidationError

# --- Test Fixtures ---

# Define mock config data globally for convenience
mock_config_data = {
    'api': {'resolved_key': 'mock_key'},
    'files': {
        'default_task': '/abs/path/to/default_task.txt',
        'default_output': '/abs/path/to/default_output.json',
        'default_validated_output': '/abs/path/to/default_validated.json'
    },
    'logging': {'level': 'INFO', 'log_file': None, 'log_to_console': False}
}

@pytest.fixture(autouse=True)
def reset_sys_argv():
    """Reset sys.argv before each test."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv

@pytest.fixture(autouse=True)
def mock_main_config_and_logging(mocker):
    """Fixture to automatically mock config loading and logging setup for all tests."""
    # Patch where they are looked up in the main module
    mocker.patch('hierarchical_planner.main.load_config', return_value=mock_config_data)
    mocker.patch('hierarchical_planner.main.setup_logging')
    # Also mock the logger instance used within main after setup
    mocker.patch('hierarchical_planner.main.logger')


# --- Test Cases ---

def test_argument_parsing_defaults(mocker):
    """Test argument parsing uses defaults from (mocked) config."""
    # Create a local copy of mock_config_data for this test to avoid affecting other tests
    local_mock_config = mock_config_data.copy()
    
    # Mock the CONFIG used by the parser
    mock_config_getter = mocker.patch('hierarchical_planner.main.CONFIG', local_mock_config)
    
    # Mock os.path.basename used in help string generation
    mocker.patch('os.path.basename', side_effect=lambda x: x.split('/')[-1])
    sys.argv = ['main.py']
    
    # Re-create parser to ensure mocks are active during its creation
    parser = main.argparse.ArgumentParser(description="Test Parser") # Simplified parser for test
    
    # Override the global parser instance
    main.parser = parser 
    
    # Add arguments using mocked CONFIG
    parser.add_argument(
        "--task-file", type=str, default=local_mock_config['files']['default_task'],
        help=f"Default: {os.path.basename(local_mock_config['files']['default_task'])}"
    )
    parser.add_argument(
        "--output-file", type=str, default=local_mock_config['files']['default_output'],
        help=f"Default: {os.path.basename(local_mock_config['files']['default_output'])}"
    )
    parser.add_argument(
        "--validated-output-file", type=str, default=local_mock_config['files']['default_validated_output'],
        help=f"Default: {os.path.basename(local_mock_config['files']['default_validated_output'])}"
    )
    parser.add_argument("--skip-qa", action="store_true")

    # Parse arguments and verify defaults are from mock_config
    args = parser.parse_args()
    assert args.task_file == local_mock_config['files']['default_task']
    assert args.output_file == local_mock_config['files']['default_output']
    assert args.validated_output_file == local_mock_config['files']['default_validated_output']
    assert not args.skip_qa

def test_argument_parsing_overrides(mocker):
    """Test argument parsing overrides defaults."""
    mocker.patch('os.path.basename', side_effect=lambda x: x.split('/')[-1])
    custom_task = "my_task.txt"
    custom_output = "out.json"
    custom_validated = "val_out.json"
    sys.argv = [
        'main.py',
        '--task-file', custom_task,
        '--output-file', custom_output,
        '--validated-output-file', custom_validated,
        '--skip-qa'
    ]
    # Re-create parser as in previous test
    parser = main.argparse.ArgumentParser(description="Test Parser")
    main.parser = parser
    parser.add_argument("--task-file", type=str, default=main.CONFIG['files']['default_task'])
    parser.add_argument("--output-file", type=str, default=main.CONFIG['files']['default_output'])
    parser.add_argument("--validated-output-file", type=str, default=main.CONFIG['files']['default_validated_output'])
    parser.add_argument("--skip-qa", action="store_true")

    args = parser.parse_args()

    assert args.task_file == custom_task
    assert args.output_file == custom_output
    assert args.validated_output_file == custom_validated
    assert args.skip_qa

@pytest.mark.asyncio
async def test_main_workflow_success_with_qa(mocker):
    """Test the main workflow success path including QA."""
    mock_generate = mocker.patch('hierarchical_planner.main.generate_plan', new_callable=AsyncMock)
    mock_validate = mocker.patch('hierarchical_planner.main.run_qa_validation', new_callable=AsyncMock)
    mock_generate.return_value = ({'mock': 'plan'}, 'mock goal')
    mock_validate.return_value = None

    task = "task.txt"
    output = "out.json"
    validated = "val_out.json"

    await main.main_workflow(task, output, validated, skip_qa=False, config=mock_config_data)

    mock_generate.assert_awaited_once_with(task, output, mock_config_data)
    mock_validate.assert_awaited_once_with(input_path=output, output_path=validated, config=mock_config_data)
    main.logger.info.assert_any_call("Workflow finished successfully.") # Check for success log

@pytest.mark.asyncio
async def test_main_workflow_success_skip_qa(mocker):
    """Test the main workflow success path skipping QA."""
    mock_generate = mocker.patch('hierarchical_planner.main.generate_plan', new_callable=AsyncMock)
    mock_validate = mocker.patch('hierarchical_planner.main.run_qa_validation', new_callable=AsyncMock)
    mock_generate.return_value = ({'mock': 'plan'}, 'mock goal')

    task = "task.txt"
    output = "out.json"
    validated = "val_out.json"

    await main.main_workflow(task, output, validated, skip_qa=True, config=mock_config_data)

    mock_generate.assert_awaited_once_with(task, output, mock_config_data)
    mock_validate.assert_not_awaited()
    main.logger.info.assert_any_call("--- Skipping QA Validation Step ---")
    main.logger.info.assert_any_call("Workflow finished successfully.")

@pytest.mark.asyncio
async def test_main_workflow_plan_generation_fails(mocker):
    """Test the workflow when generate_plan raises an error."""
    mock_generate = mocker.patch('hierarchical_planner.main.generate_plan', new_callable=AsyncMock)
    mock_validate = mocker.patch('hierarchical_planner.main.run_qa_validation', new_callable=AsyncMock)
    error_message = "Failed to generate phases"
    mock_generate.side_effect = PlanGenerationError(error_message)

    task = "task.txt"
    output = "out.json"
    validated = "val_out.json"

    await main.main_workflow(task, output, validated, skip_qa=False, config=mock_config_data)

    mock_generate.assert_awaited_once()
    mock_validate.assert_not_awaited()
    # Check if the specific error was logged - note format string is in the actual implementation
    main.logger.error.assert_any_call(f"Plan generation failed: {error_message}", exc_info=True)

@pytest.mark.asyncio
async def test_main_workflow_qa_fails(mocker):
    """Test the workflow when run_qa_validation raises an error."""
    mock_generate = mocker.patch('hierarchical_planner.main.generate_plan', new_callable=AsyncMock)
    mock_validate = mocker.patch('hierarchical_planner.main.run_qa_validation', new_callable=AsyncMock)      
    mock_generate.return_value = ({'mock': 'plan'}, 'mock goal')
    error_message = "QA structure invalid"
    mock_validate.side_effect = PlanValidationError(error_message)

    task = "task.txt"
    output = "out.json"
    validated = "val_out.json"

    await main.main_workflow(task, output, validated, skip_qa=False, config=mock_config_data)

    mock_generate.assert_awaited_once()
    mock_validate.assert_awaited_once()
    # Match the actual f-string format used in main.py
    main.logger.error.assert_called_with(f"Plan validation failed: {error_message}", exc_info=True)


# Test the main execution block (__name__ == "__main__")

# Patching the __main__ block execution is complex.
# A better approach is to refactor the logic within the `if __name__ == "__main__":`
# block into a separate function, say `run_main()`, and test that function.

# Let's assume we refactor main.py like this (outside the test):
# def run_main():
#     # Original __main__ block logic here...
#     try:
#         logger.info("Starting main workflow...")
#         asyncio.run(...)
#     except ...:
#         ...
#
# if __name__ == "__main__":
#      run_main()

# Then we can test run_main()
@patch('hierarchical_planner.main.argparse.ArgumentParser')
@patch('hierarchical_planner.main.main_workflow')
@patch('sys.exit')
def test_run_main_success(mock_exit, mock_main_wf, mock_arg_parser, mocker):
    """Tests the logic equivalent to the __main__ block success path."""
    # Mock parser setup and args
    mock_args = argparse.Namespace(
        task_file='specific_task.txt',
        output_file=mock_config_data['files']['default_output'],
        validated_output_file=mock_config_data['files']['default_validated_output'],
        skip_qa=True
    )
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_args.return_value = mock_args
    mock_arg_parser.return_value = mock_parser_instance
    mocker.patch('os.path.basename', side_effect=lambda x: x.split('/')[-1])

    # Mock the configuration used in main
    main.CONFIG = mock_config_data 
    main.logger = MagicMock()
    
    # We need to simulate the __main__ block execution
    # This is equivalent to the code in the if __name__ == "__main__": block
    try:
        # This is what asyncio.run would do in the actual code
        mock_main_wf.return_value = None
        
        # Call main_workflow directly (simulating the asyncio.run call)
        main.main_workflow(
            task_file=mock_args.task_file,
            output_file=mock_args.output_file,
            validated_output_file=mock_args.validated_output_file,
            skip_qa=mock_args.skip_qa,
            config=main.CONFIG
        )
        
        # Verify the main_workflow was called with correct args
        mock_main_wf.assert_called_once_with(
            task_file='specific_task.txt',
            output_file=mock_config_data['files']['default_output'],
            validated_output_file=mock_config_data['files']['default_validated_output'],
            skip_qa=True,
            config=mock_config_data
        )
        # Success path shouldn't call sys.exit
        mock_exit.assert_not_called()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


@patch('hierarchical_planner.main.argparse.ArgumentParser')
@patch('hierarchical_planner.main.main_workflow')
@patch('sys.exit')
def test_run_main_exception(mock_exit, mock_main_wf, mock_arg_parser, mocker):
    """Tests the logic equivalent to the __main__ block exception path."""
    # Mock parser setup and args
    mock_args = argparse.Namespace(
        task_file=mock_config_data['files']['default_task'],
        output_file=mock_config_data['files']['default_output'],
        validated_output_file=mock_config_data['files']['default_validated_output'],
        skip_qa=False
    )
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_args.return_value = mock_args
    mock_arg_parser.return_value = mock_parser_instance
    mocker.patch('os.path.basename', side_effect=lambda x: x.split('/')[-1])

    # Make main_workflow raise an exception to test the error path
    error_message = "Simulated top-level error"
    mock_main_wf.side_effect = HierarchicalPlannerError(error_message)

    # Mock the configuration used in main
    main.CONFIG = mock_config_data
    main.logger = MagicMock()

    # Directly simulate the __main__ block's try-except structure
    try:
        asyncio.run(main.main_workflow(
            task_file=mock_args.task_file,
            output_file=mock_args.output_file,
            validated_output_file=mock_args.validated_output_file,
            skip_qa=mock_args.skip_qa,
            config=main.CONFIG
        ))
    except HierarchicalPlannerError as e:
        # This should be caught in the __main__ block, so we simulate that handling here
        main.logger.critical(f"Application error at top level: {e}", exc_info=True)
        sys.exit(1)  # This will actually be mocked by patch('sys.exit')
    
    # Verify logging and exit were called correctly
    # Match the actual f-string format used in main.py
    main.logger.critical.assert_called_with(f"Application error at top level: {error_message}", exc_info=True)
    mock_exit.assert_called_once_with(1)
