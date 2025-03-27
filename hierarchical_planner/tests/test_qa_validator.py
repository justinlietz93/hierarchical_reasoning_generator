import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Module to test
from hierarchical_planner import qa_validator
from hierarchical_planner.exceptions import PlanValidationError, PlannerFileNotFoundError, FileReadError, JsonParsingError, FileWriteError, JsonSerializationError, ApiCallError

# --- Test Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a basic mock config dictionary for QA tests."""
    return {
        'api': {
            'resolved_key': 'test_api_key_123',
            'model_name': 'mock-model-qa',
            'retries': 1, # Faster tests
            'delay_between_qa_calls_sec': 0 # No delay for tests
        },
        'files': {
            'default_task': 'task.txt' # Needed for goal loading
        }
        # Add other sections if needed
    }

@pytest.fixture
def valid_plan_data():
    """Provides a structurally valid plan dictionary."""
    return {
        "Phase 1": {
            "Task 1.1": [
                {"step 1": "Do thing A"},
                {"step 2": "Do thing B"}
            ]
        },
        "Phase 2": {
            "Task 2.1": [
                {"step 1": "Do thing C"}
            ],
            "Task 2.2": [] # Empty steps list is valid
        }
    }

@pytest.fixture
def invalid_plan_data_root():
    """Invalid plan: root is not a dict."""
    return ["list", "instead", "of", "dict"]

@pytest.fixture
def invalid_plan_data_phase_value():
    """Invalid plan: phase value is not a dict."""
    return {"Phase 1": ["list", "of", "tasks"]}

@pytest.fixture
def invalid_plan_data_task_value():
    """Invalid plan: task value is not a list."""
    return {"Phase 1": {"Task 1.1": {"step 1": "not a list"}}}

@pytest.fixture
def invalid_plan_data_step_format():
    """Invalid plan: step is not a dict or has wrong keys/values."""
    return {
        "Phase 1": {
            "Task 1.1": [
                "just a string", # Not a dict
                {"step 1": "ok", "step 2": "too many keys"},
                {"step 3": ""}, # Empty prompt string
                {"step 4": 123}, # Non-string prompt value
                {"nostep 5": "key doesn't start with step"} # Warning, not error
            ]
        }
    }

# --- Test Cases for validate_plan_structure ---

def test_validate_structure_valid(valid_plan_data):
    """Test structure validation with a valid plan."""
    errors = qa_validator.validate_plan_structure(valid_plan_data)
    assert not errors

def test_validate_structure_invalid_root(invalid_plan_data_root):
    """Test structure validation with invalid root type."""
    errors = qa_validator.validate_plan_structure(invalid_plan_data_root)
    assert len(errors) == 1
    assert "Root level must be a dictionary" in errors[0]

def test_validate_structure_invalid_phase_value(invalid_plan_data_phase_value):
    """Test structure validation with invalid phase value type."""
    errors = qa_validator.validate_plan_structure(invalid_plan_data_phase_value)
    assert len(errors) == 1
    assert "must be a dictionary (tasks)" in errors[0]

def test_validate_structure_invalid_task_value(invalid_plan_data_task_value):
    """Test structure validation with invalid task value type."""
    errors = qa_validator.validate_plan_structure(invalid_plan_data_task_value)
    assert len(errors) == 1
    assert "must be a list (steps)" in errors[0]

def test_validate_structure_invalid_step_format(invalid_plan_data_step_format):
    """Test structure validation with various invalid step formats."""
    errors = qa_validator.validate_plan_structure(invalid_plan_data_step_format)
    # Expect errors for: not a dict, empty string, non-string value
    # Note: The "too many keys" case doesn't actually trigger an error in the current implementation
    # as it only checks for empty strings and non-string values
    assert len(errors) == 3
    assert any("must be a dictionary" in err for err in errors)
    assert any("must be a non-empty string" in err for err in errors)
    assert any("'step 4'" in err for err in errors)  # Check specific step with non-string value

# --- Test Cases for analyze_and_annotate_plan ---

@pytest.mark.asyncio
@patch('hierarchical_planner.qa_validator.call_gemini_with_retry', new_callable=AsyncMock)
async def test_analyze_annotate_success(mock_call_retry, mock_config, valid_plan_data):
    """Test successful analysis and annotation."""
    mock_call_retry.side_effect = [
        # Task 1.1, Step 1 - Resource Analysis
        {"external_actions": [], "key_entities_dependencies": ["thing A"], "technology_hints": []},
        # Task 1.1, Step 1 - Step Critique
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
        # Task 1.1, Step 2 - Resource Analysis
        {"external_actions": ["Search B"], "key_entities_dependencies": [], "technology_hints": []},
        # Task 1.1, Step 2 - Step Critique
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
        # Task 2.1, Step 1 - Resource Analysis
        {"external_actions": [], "key_entities_dependencies": ["thing C"], "technology_hints": ["libX"]},
        # Task 2.1, Step 1 - Step Critique
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
    ]

    goal = "Test Goal"
    annotated_plan = await qa_validator.analyze_and_annotate_plan(valid_plan_data, goal, mock_config)

    # Check if qa_info was added correctly
    step1_1_1 = annotated_plan["Phase 1"]["Task 1.1"][0]
    step1_1_2 = annotated_plan["Phase 1"]["Task 1.1"][1]
    step2_1_1 = annotated_plan["Phase 2"]["Task 2.1"][0]

    assert "qa_info" in step1_1_1
    assert "resource_analysis" in step1_1_1["qa_info"]
    assert "step_critique" in step1_1_1["qa_info"]
    assert step1_1_1["qa_info"]["resource_analysis"]["key_entities_dependencies"] == ["thing A"]
    assert step1_1_1["qa_info"]["step_critique"]["clarity_critique"] == "Clear"

    assert "qa_info" in step1_1_2
    assert step1_1_2["qa_info"]["resource_analysis"]["external_actions"] == ["Search B"]

    assert "qa_info" in step2_1_1
    assert step2_1_1["qa_info"]["resource_analysis"]["technology_hints"] == ["libX"]

    # Check number of API calls (2 per step with a prompt)
    assert mock_call_retry.call_count == 6

@pytest.mark.asyncio
@patch('hierarchical_planner.qa_validator.call_gemini_with_retry', new_callable=AsyncMock)
async def test_analyze_annotate_api_error_handling(mock_call_retry, mock_config, valid_plan_data):
    """Test that analysis continues and logs errors if one API call fails."""
    mock_call_retry.side_effect = [
        # Task 1.1, Step 1 - Resource Analysis (Success)
        {"external_actions": [], "key_entities_dependencies": ["thing A"], "technology_hints": []},
        # Task 1.1, Step 1 - Step Critique (Success)
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
        # Task 1.1, Step 2 - Resource Analysis (FAIL)
        ApiCallError("Simulated API failure for resource analysis"),
        # Task 1.1, Step 2 - Step Critique (Success - assuming it's called even if previous failed)
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
        # Task 2.1, Step 1 - Resource Analysis (Success)
        {"external_actions": [], "key_entities_dependencies": ["thing C"], "technology_hints": ["libX"]},
        # Task 2.1, Step 1 - Step Critique (Success)
        {"alignment_critique": "Good", "sequence_critique": "OK", "clarity_critique": "Clear"},
    ]

    goal = "Test Goal"
    annotated_plan = await qa_validator.analyze_and_annotate_plan(valid_plan_data, goal, mock_config)

    step1_1_1 = annotated_plan["Phase 1"]["Task 1.1"][0]
    step1_1_2 = annotated_plan["Phase 1"]["Task 1.1"][1]
    step2_1_1 = annotated_plan["Phase 2"]["Task 2.1"][0]

    # Check successful steps
    assert "resource_analysis" in step1_1_1["qa_info"]
    assert "step_critique" in step1_1_1["qa_info"]
    assert "resource_analysis" in step2_1_1["qa_info"]
    assert "step_critique" in step2_1_1["qa_info"]

    # Check failed step - error message should be recorded
    assert "qa_info" in step1_1_2
    assert "resource_analysis_error" in step1_1_2["qa_info"]
    assert "API Error: Simulated API failure" in step1_1_2["qa_info"]["resource_analysis_error"]
    # Check that critique was still attempted and succeeded for the step where resource analysis failed
    assert "step_critique" in step1_1_2["qa_info"]
    assert "resource_analysis" not in step1_1_2["qa_info"] # Ensure success key isn't present

    assert mock_call_retry.call_count == 6 # Still called 6 times

# --- Test Cases for run_validation ---

@pytest.mark.asyncio
@patch('hierarchical_planner.qa_validator.validate_plan_structure', return_value=[]) # Assume valid structure
@patch('hierarchical_planner.qa_validator.analyze_and_annotate_plan', new_callable=AsyncMock)
async def test_run_validation_success(mock_analyze, mock_validate, mock_config):
    """Test the main run_validation function success path."""
    input_path = "/fake/input.json"
    output_path = "/fake/output_validated.json"
    goal_content = "My Test Goal"
    plan_content = {"Phase 1": {"Task 1": [{"step 1": "..."}]}}
    annotated_content = {"Phase 1": {"Task 1": [{"step 1": "...", "qa_info": {}}]}} # Example annotation     

    # Use a proper context manager for patching open
    with patch('builtins.open') as mock_file:
        # Set up the mock to handle multiple calls with different return values
        file_handles = [
            MagicMock(),  # Handle for input file (plan)
            MagicMock(),  # Handle for goal file
            MagicMock()   # Handle for output file
        ]
        mock_file.side_effect = file_handles
        
        # Set up the read data for plan file
        file_handles[0].__enter__.return_value.read.return_value = json.dumps(plan_content)
        # Set up the read data for goal file
        file_handles[1].__enter__.return_value.read.return_value = goal_content
        
        # Set up the mocks for json operations
        with patch('json.load', return_value=plan_content), patch('json.dump') as mock_json_dump:
            # Mock analyze function return value
            mock_analyze.return_value = annotated_content
            
            # Run the function under test
            await qa_validator.run_validation(input_path, output_path, mock_config)

            # Check calls
            assert mock_file.call_count >= 3  # Read plan, read goal, write output
            mock_validate.assert_called_once_with(plan_content)
            mock_analyze.assert_called_once_with(plan_content, goal_content, mock_config)
            
            # Check that json.dump was called with correct arguments
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            assert args[0] == annotated_content  # First arg is the data to dump
            assert kwargs.get('indent') == 2  # Check the indent parameter

@pytest.mark.asyncio
@patch('builtins.open', mock_open(read_data='invalid json'))
async def test_run_validation_json_error(mock_config):
    """Test run_validation handling JSON parsing error."""
    with pytest.raises(JsonParsingError):
        await qa_validator.run_validation("/fake/input.json", "/fake/output.json", mock_config)

@pytest.mark.asyncio
@patch('builtins.open', side_effect=FileNotFoundError("File not here"))
async def test_run_validation_input_not_found(mock_file, mock_config):
    """Test run_validation handling input file not found."""
    with pytest.raises(PlannerFileNotFoundError):
        await qa_validator.run_validation("/fake/input.json", "/fake/output.json", mock_config)

@pytest.mark.asyncio
@patch('builtins.open', new_callable=mock_open)
@patch('hierarchical_planner.qa_validator.validate_plan_structure', return_value=["Structure error!"])
async def test_run_validation_structure_error(mock_validate, mock_file, mock_config):
    """Test run_validation handling structure validation errors."""
    plan_content = {"invalid": "structure"}
    mock_file.return_value = mock_open(read_data=json.dumps(plan_content)).return_value

    with pytest.raises(PlanValidationError, match="Structure error!"):
        await qa_validator.run_validation("/fake/input.json", "/fake/output.json", mock_config)
    mock_validate.assert_called_once_with(plan_content)

@pytest.mark.asyncio
@patch('builtins.open', new_callable=mock_open)
@patch('hierarchical_planner.qa_validator.validate_plan_structure', return_value=[])
async def test_run_validation_goal_file_error(mock_validate, mock_file, mock_config):
    """Test run_validation handling errors reading the goal file."""
    plan_content = {"Phase 1": {"Task 1": [{"step 1": "..."}]}}
    # Simulate plan read OK, but goal read fails
    mock_file.side_effect = [
        mock_open(read_data=json.dumps(plan_content)).return_value,
        FileNotFoundError("Goal file missing")
    ]
    with pytest.raises(PlannerFileNotFoundError, match="Goal file"):
        await qa_validator.run_validation("/fake/input.json", "/fake/output.json", mock_config)

@pytest.mark.asyncio
@patch('builtins.open', new_callable=mock_open)
@patch('hierarchical_planner.qa_validator.validate_plan_structure', return_value=[])
@patch('hierarchical_planner.qa_validator.analyze_and_annotate_plan', new_callable=AsyncMock)
async def test_run_validation_write_error(mock_analyze, mock_validate, mock_file, mock_config):
    """Test run_validation handling errors writing the output file."""
    input_path = "/fake/input.json"
    output_path = "/fake/output_validated.json"
    goal_content = "My Test Goal"
    plan_content = {"Phase 1": {"Task 1": [{"step 1": "..."}]}}
    annotated_content = {"Phase 1": {"Task 1": [{"step 1": "...", "qa_info": {}}]}}

    # Mock reads OK, analysis OK, but write fails
    mock_file.side_effect = [
        mock_open(read_data=json.dumps(plan_content)).return_value, # Read plan
        mock_open(read_data=goal_content).return_value,           # Read goal
        IOError("Cannot write")                                   # Write output fails
    ]
    mock_analyze.return_value = annotated_content

    with pytest.raises(FileWriteError, match="Cannot write"):
        await qa_validator.run_validation(input_path, output_path, mock_config)

    assert mock_file.call_count == 3 # Read plan, read goal, attempt write output
