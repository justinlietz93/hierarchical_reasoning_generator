"""
Main script for the Hierarchical Planner application.

Orchestrates the process of:
1. Loading configuration.
2. Setting up logging.
3. Reading a high-level task from an input file.
4. Calling the Gemini API to break down the task into Phases -> Tasks -> Steps.
5. Saving the generated plan as a JSON file.
6. Optionally, running a QA validation and annotation process on the plan.
"""
import asyncio
import json
import os
import argparse
import logging
import sys
from typing import Dict, Any

# Local imports
# Note: gemini_client functions now require config passed in
# Moved call_gemini_with_retry to gemini_client
from .gemini_client import generate_structured_content, generate_content, call_gemini_with_retry
# TODO: Update qa_validator import/call signature when config is integrated there
from .qa_validator import run_validation as run_qa_validation
from .config_loader import load_config # ConfigError is now in exceptions
from .logger_setup import setup_logging # Added
# Import custom exceptions
from .exceptions import (
    HierarchicalPlannerError, ConfigError, FileProcessingError,
    FileNotFoundError as PlannerFileNotFoundError, # Alias to avoid conflict
    FileReadError, FileWriteError, PlanGenerationError, PlanValidationError,
    JsonSerializationError, ApiCallError, JsonProcessingError # Added JsonProcessingError
)

# --- Load Configuration ---
try:
    # Assumes config.yaml is in ../config relative to this file's location (hierarchical_planner/)
    CONFIG = load_config('../config/config.yaml')
    # --- Setup Logging ---
    setup_logging(CONFIG) # Call the setup function HERE
    logger = logging.getLogger(__name__) # Get logger AFTER setup
    logger.info("Configuration loaded and logging configured successfully.")
except ConfigError as e:
    # Catch specific ConfigError from loader
    print(f"CRITICAL: Configuration error: {e}", file=sys.stderr)
    sys.exit(1) # Exit if config fails
except Exception as e:
     # Catch any other unexpected error during setup
    print(f"CRITICAL: Unexpected error during application setup: {e}", file=sys.stderr)
    sys.exit(1)


# --- Configuration (Now loaded from CONFIG) ---
# Constants are no longer needed here

# --- Prompt Templates ---
# These are crucial and will likely need refinement based on Gemini's responses.

PHASE_GENERATION_PROMPT = """
Given the high-level user goal: "{goal}"

Break this goal down into the major, distinct phases required for completion.
Each phase should represent a significant stage of the project.
List the phases concisely.

Return the response as a JSON object with a single key "phases" containing a list of strings, where each string is a phase name/description.
Example:
{{
  "phases": [
    "Phase 1: Planning and Design",
    "Phase 2: Core Feature Implementation",
    "Phase 3: Testing and Refinement",
    "Phase 4: Deployment"
  ]
}}
"""

TASK_GENERATION_PROMPT = """
Given the high-level user goal: "{goal}"
And the current phase: "{phase}"

Break this phase down into specific, actionable tasks.
Each task should be a concrete unit of work needed to complete the phase.
List the tasks concisely.

Return the response as a JSON object with a single key "tasks" containing a list of strings, where each string is a task name/description.
Example:
{{
  "tasks": [
    "Task 1.1: Define data models",
    "Task 1.2: Design UI mockups",
    "Task 1.3: Set up project structure"
  ]
}}
"""

STEP_GENERATION_PROMPT = """
You are an expert planner assisting an autonomous AI coding agent.
Your goal is to generate a sequence of detailed, step-by-step prompts that will guide the AI agent to complete a specific task.

High-level user goal: "{goal}"
Current phase: "{phase}"
Current task: "{task}"

Generate a sequence of prompts for the AI coding agent to execute this task.
Each prompt should be:
1.  **Actionable:** Clearly state what the AI agent needs to do.
2.  **Specific:** Provide enough detail for the agent to understand the requirement.
3.  **Contextual:** Assume the agent has access to the project state from previous steps.
4.  **Include Hints:** Where necessary, suggest tools, libraries, techniques, or external actions (e.g., "Search the web for...", "Install library X", "Write unit tests for Y", "Refactor Z for clarity").
5.  **Sequential:** The prompts should follow a logical order of execution.

Return the response as a JSON object with a single key "steps" containing a list of dictionaries. Each dictionary must have exactly one key-value pair, where the key is the step identifier (e.g., "step 1", "step 2") and the value is the detailed prompt string for the AI agent.
Example:
{{
  "steps": [
    {{"step 1": "Create a new Python file named 'parser.py'."}},
    {{"step 2": "Define a function 'parse_input(data: str) -> dict' in 'parser.py'."}},
    {{"step 3": "Implement basic error handling for invalid input formats within the 'parse_input' function."}},
    {{"step 4": "Write three unit tests for the 'parse_input' function using the 'unittest' library, covering valid input, invalid input, and edge cases."}}
  ]
}}
"""

# --- Helper Function (Moved to gemini_client.py) ---
# async def call_gemini_with_retry(...): ... # Moved

# --- Main Logic ---

async def generate_plan(task_file: str, output_file: str, config: Dict[str, Any]) -> tuple[dict | None, str | None]:
    """
    Generates the hierarchical plan (Phases, Tasks, Steps) using Gemini.

    Reads the goal from the task_file, then iteratively calls the Gemini API
    (via `call_gemini_with_retry`) to generate phases, tasks for each phase,
    and steps for each task. Saves the resulting plan structure to output_file.

    Args:
        task_file: Absolute path to the input file containing the high-level goal.
        output_file: Absolute path to save the generated JSON plan.
        config: The application configuration dictionary.

    Returns:
        A tuple containing:
            - The generated reasoning tree (dict) or None if generation failed critically.
            - The goal string read from the task file, or None if reading failed.

    Raises:
        PlannerFileNotFoundError: If the task file is not found.
        FileReadError: If the task file cannot be read.
        PlanGenerationError: If a critical part of the plan (e.g., phases) fails generation.
        ApiCallError: If an API call fails after retries during generation.
        FileWriteError: If the output JSON file cannot be written.
        JsonSerializationError: If the generated plan cannot be serialized to JSON.
    """
    logger.info("Starting hierarchical planning process...")
    goal: str | None = None # Initialize goal

    # 1. Read Goal
    try:
        # Ensure task_file path is absolute or correctly relative (config_loader handles this)
        logger.info(f"Reading goal from: {task_file}")
        with open(task_file, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
            logger.error(f"Task file '{task_file}' is empty.")
            raise FileReadError(f"Task file '{task_file}' is empty.") # Raise custom exception
        logger.info(f"Goal read successfully: '{goal[:100]}...'") # Log truncated goal
    except FileNotFoundError: # Catch built-in error
        logger.error(f"Task file '{task_file}' not found.")
        raise PlannerFileNotFoundError(f"Task file '{task_file}' not found.") from None # Raise custom exception
    except IOError as e:
        logger.error(f"Error reading task file '{task_file}': {e}", exc_info=True)
        raise FileReadError(f"Error reading task file '{task_file}': {e}") from e # Raise custom exception


    reasoning_tree = {}
    # script_dir = os.path.dirname(__file__) or '.' # No longer needed for output path resolution

    try:
        # 2. Generate Phases
        logger.info("Generating phases...")
        phase_context = {"goal": goal}
        # Pass config to retry function (now imported from gemini_client)
        # Catch potential API errors from retry function
        phase_response = await call_gemini_with_retry(PHASE_GENERATION_PROMPT, phase_context, config)
        phases = phase_response.get("phases", [])
        if not phases:
            logger.error("Could not generate phases from Gemini response.")
            raise PlanGenerationError("Failed to generate phases from Gemini.") # Raise custom exception
        logger.info(f"Generated {len(phases)} phases.")

    except ApiCallError as e:
        # Handle API call failures specifically if needed, or let it propagate
        logger.error(f"API call failed during phase generation: {e}", exc_info=True)
        raise # Re-raise to be caught by main_workflow or main block

    # --- Generation Loop (Tasks and Steps) ---
    # Wrap the loop in a try block to catch errors during task/step generation
    try:
        # 3. Generate Tasks for each Phase
        for phase in phases:
            logger.info(f"Generating tasks for Phase: {phase}")
            task_context = {"goal": goal, "phase": phase}
            # Pass config to retry function
            task_response = await call_gemini_with_retry(TASK_GENERATION_PROMPT, task_context, config)
            tasks = task_response.get("tasks", [])
            if not tasks:
                logger.warning(f"Could not generate tasks for phase '{phase}'. Skipping.")
                reasoning_tree[phase] = {} # Add phase even if no tasks generated
                continue
            logger.info(f"Generated {len(tasks)} tasks for this phase.")
            reasoning_tree[phase] = {}

            # 4. Generate Steps for each Task
            for task in tasks:
                logger.info(f"  Generating steps for Task: {task}")
                step_context = {"goal": goal, "phase": phase, "task": task}
                # Pass config to retry function
                step_response = await call_gemini_with_retry(STEP_GENERATION_PROMPT, step_context, config)
                steps = step_response.get("steps", [])
                if not steps:
                     logger.warning(f"Could not generate steps for task '{task}'. Skipping.")
                     reasoning_tree[phase][task] = [] # Add task even if no steps generated
                     continue
                logger.info(f"  Generated {len(steps)} steps for this task.")
                reasoning_tree[phase][task] = steps

        # 5. Write Output JSON
        # Output path is now absolute, resolved by config_loader
        logger.info(f"Writing initial reasoning tree to {output_file}...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Correct indentation for json.dump
                json.dump(reasoning_tree, f, indent=2, ensure_ascii=False)
            logger.info("Initial planning process completed successfully.")
        except IOError as e:
            logger.error(f"Error writing output file '{output_file}': {e}", exc_info=True)
            raise FileWriteError(f"Error writing output file '{output_file}': {e}") from e # Raise custom exception
        except TypeError as e:
             logger.error(f"Error serializing reasoning tree to JSON: {e}", exc_info=True)
             raise JsonSerializationError(f"Error serializing reasoning tree to JSON: {e}") from e # Raise custom exception

        return reasoning_tree, goal # Return the generated tree and goal

    except ApiCallError as e:
         # Catch API errors during task/step generation
        logger.error(f"API call failed during task/step generation: {e}", exc_info=True)
        raise # Re-raise
    except Exception as e:
        # Catch other unexpected errors during generation loop
        logger.error(f"An unexpected error occurred during plan generation loop: {e}", exc_info=True)
        raise PlanGenerationError(f"An unexpected error occurred during plan generation loop: {e}") from e


async def main_workflow(task_file: str, output_file: str, validated_output_file: str, skip_qa: bool, config: Dict[str, Any]):
    """
    Orchestrates the full application workflow.

    Calls `generate_plan` to create the initial plan. If successful and QA is
    not skipped, calls `run_validation` to perform QA analysis and annotation.
    Handles and logs exceptions raised during the process.

    Args:
        task_file: Absolute path to the input task file.
        output_file: Absolute path for the initial output plan JSON.
        validated_output_file: Absolute path for the validated/annotated plan JSON.
        skip_qa: Boolean flag to skip the QA validation step.
        config: The application configuration dictionary.
    """
    reasoning_tree: dict | None = None
    goal: str | None = None
    try:
        # Step 1: Generate the initial plan
        reasoning_tree, goal = await generate_plan(task_file, output_file, config)
        # generate_plan now raises exceptions on failure, so no need to check for None

        # Step 2: Run QA Validation (if not skipped)
        if not skip_qa:
            logger.info("--- Starting QA Validation Step ---")
            if not goal:
                 # This case should ideally be caught by generate_plan raising an error
                 logger.error("Goal is missing, cannot run QA validation.")
                 raise PlanValidationError("Goal is missing, cannot run QA validation.")

            # Pass config down to QA validation function
            await run_qa_validation(input_path=output_file, output_path=validated_output_file, config=config)
            logger.info("--- QA Validation Step Completed ---")
        else:
            logger.info("--- Skipping QA Validation Step ---")

        logger.info("Workflow finished successfully.")

    except PlannerFileNotFoundError as e:
        logger.error(f"Input file error: {e}")
        # No further action needed, error logged
    except FileProcessingError as e:
        logger.error(f"File processing error: {e}", exc_info=True)
    except PlanGenerationError as e:
        logger.error(f"Plan generation failed: {e}", exc_info=True)
    except PlanValidationError as e: # Catch validation errors if run_qa_validation raises them
        logger.error(f"Plan validation failed: {e}", exc_info=True)
    except ApiCallError as e:
        logger.error(f"API call failed during workflow: {e}", exc_info=True)
    except JsonProcessingError as e:
        logger.error(f"JSON processing error: {e}", exc_info=True)
    except HierarchicalPlannerError as e:
        # Catch any other application-specific errors
        logger.error(f"An application error occurred: {e}", exc_info=True)
    except Exception as e:
        # Catch unexpected errors
        logger.critical(f"An unexpected error occurred in the main workflow: {e}", exc_info=True)

    # Removed duplicated main_workflow definition and redundant checks/logging


if __name__ == "__main__":
    # Config is loaded and logging is set up before this block

    # Use default file paths from loaded config
    # These paths are resolved to be absolute in config_loader
    default_task = CONFIG['files']['default_task']
    default_output = CONFIG['files']['default_output']
    default_validated = CONFIG['files']['default_validated_output']

    parser = argparse.ArgumentParser(description="Generate and optionally validate a hierarchical reasoning plan.")
    parser.add_argument(
        "--task-file",
        type=str,
        default=default_task,
        help=f"Path to the file containing the high-level task (default: {os.path.basename(default_task)})" # Show relative default in help
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=default_output,
        help=f"Path to save the initial reasoning tree JSON (default: {os.path.basename(default_output)})" # Show relative default in help
    )
    parser.add_argument(
        "--validated-output-file",
        type=str,
        default=default_validated,
        help=f"Path to save the validated reasoning tree JSON (default: {os.path.basename(default_validated)})" # Show relative default in help
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip the QA validation and annotation step."
    )
    # TODO: Add arguments to override config settings like model, temperature, log level?

    args = parser.parse_args()

    # API key check is now handled by config_loader, remove old check

    # Run the main workflow, passing resolved paths from args
    # Config is already loaded globally as CONFIG
    try:
        logger.info("Starting main workflow...")
        asyncio.run(main_workflow(
            task_file=args.task_file,
            output_file=args.output_file,
            validated_output_file=args.validated_output_file,
            skip_qa=args.skip_qa,
            config=CONFIG # Pass global config
        ))
    except HierarchicalPlannerError as e:
        # Catch known application errors that might have been missed in main_workflow
        logger.critical(f"Application error at top level: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Catch any truly unexpected errors
        logger.critical(f"Unhandled exception at top level: {e}", exc_info=True)
        sys.exit(1)
