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
from typing import Dict, Any, Optional

# Local imports
# Note: client functions now require config passed in
# Import Gemini client
from gemini_client import generate_structured_content as gemini_generate_structured_content
from gemini_client import generate_content as gemini_generate_content
from gemini_client import call_gemini_with_retry
# Import Anthropic client
from anthropic_client import generate_structured_content as anthropic_generate_structured_content
from anthropic_client import generate_content as anthropic_generate_content
from anthropic_client import call_anthropic_with_retry
# TODO: Update qa_validator import/call signature when config is integrated there
from qa_validator import run_validation as run_qa_validation
from config_loader import load_config # ConfigError is now in exceptions
from logger_setup import setup_logging # Added
from checkpoint_manager import CheckpointManager # Import the checkpoint manager
# Import custom exceptions
from exceptions import (
    HierarchicalPlannerError, ConfigError, FileProcessingError,
    FileNotFoundError as PlannerFileNotFoundError,
    FileReadError, FileWriteError, PlanGenerationError, PlanValidationError,
    JsonSerializationError, ApiCallError, JsonProcessingError, ProjectBuilderError
)
from project_builder import ProjectBuilder

# --- Load Configuration ---
try:
    # Assumes config.yaml is in config relative to this file's location (hierarchical_planner/)
    CONFIG = load_config('config/config.yaml')
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

CONSTITUTION_GENERATION_PROMPT = """
You are a Founding Architect agent. Your sole purpose is to analyze a high-level user goal and establish the immutable foundational rules for the project.
Based on the user's goal, you must make definitive, high-level project decisions to prevent ambiguity and context drift later in the development process.

User Goal: "{goal}"

Generate a "Project Constitution" by determining the foundational rules.
Your response MUST be a valid JSON object conforming to the following JSON Schema:

{schema}
"""

PHASE_GENERATION_PROMPT = """
Project Constitution:
{constitution}

Given the high-level user goal: "{goal}"
And the established Project Constitution, break this goal down into the major, distinct phases required for completion.
Each phase should represent a significant stage of the project.
List the phases concisely.

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
Project Constitution:
{constitution}

Given the high-level user goal: "{goal}"
And the current phase: "{phase}"

Based on the constitution, break this phase down into specific, actionable tasks.
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

Project Constitution:
{constitution}

High-level user goal: "{goal}"
Current phase: "{phase}"
Current task: "{task}"

Adhering strictly to the Project Constitution, generate a sequence of prompts for the AI coding agent to execute this task.
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

# --- Helper Functions ---
from llm_client_selector import select_llm_client

# --- Main Logic ---

async def generate_constitution(goal: str, config: Dict[str, Any], provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates the project constitution by reading the schema and calling the LLM.
    """
    logger.info("Generating Project Constitution...")
    try:
        # Load the constitution schema from the external file
        schema_path = os.path.join(os.path.dirname(__file__), 'config', 'project_constitution_schema.json')
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        _, _, call_with_retry = await select_llm_client(config, provider)
        constitution_context = {
            "goal": goal,
            "schema": json.dumps(schema, indent=2)
        }
        constitution_response = await call_with_retry(CONSTITUTION_GENERATION_PROMPT, constitution_context, config)
        
        # TODO: Add validation against the loaded schema here
        
        constitution_path = "project_constitution.json"
        with open(constitution_path, 'w', encoding='utf-8') as f:
            json.dump(constitution_response, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Project Constitution saved to {constitution_path}")
        return constitution_response
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse project constitution schema: {e}", exc_info=True)
        raise ConfigError("Could not load or parse project_constitution_schema.json.") from e
    except (ApiCallError, JsonSerializationError, FileWriteError) as e:
        logger.error(f"Failed to generate or save the Project Constitution: {e}", exc_info=True)
        raise PlanGenerationError("Could not establish Project Constitution.") from e


async def generate_plan(task_file: str, output_file: str, config: Dict[str, Any], resume: bool = True, provider: Optional[str] = None, constitution: Dict[str, Any] = None) -> tuple[dict | None, str | None]:
    """
    Generates the hierarchical plan (Phases, Tasks, Steps) using Gemini.

    Reads the goal from the task_file, then iteratively calls the Gemini API
    (via `call_gemini_with_retry`) to generate phases, tasks for each phase,
    and steps for each task. Saves the resulting plan structure to output_file.

    Args:
        task_file: Absolute path to the input file containing the high-level goal.
        output_file: Absolute path to save the generated JSON plan.
        config: The application configuration dictionary.
        resume: Whether to attempt to resume from a checkpoint if available.

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
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    checkpoint_path = ""
    
    goal: str | None = None
    reasoning_tree = {}
    last_processed_phase = None
    last_processed_task = None

    # 1. Read Goal
    try:
        logger.info(f"Reading goal from: {task_file}")
        with open(task_file, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
            logger.error(f"Task file '{task_file}' is empty.")
            raise FileReadError(f"Task file '{task_file}' is empty.")
        logger.info(f"Goal read successfully: '{goal[:100]}...'")
    except FileNotFoundError:
        logger.error(f"Task file '{task_file}' not found.")
        raise PlannerFileNotFoundError(f"Task file '{task_file}' not found.")
    except IOError as e:
        logger.error(f"Error reading task file '{task_file}': {e}", exc_info=True)
        raise FileReadError(f"Error reading task file '{task_file}': {e}")

    # 2. Check for an existing checkpoint if resume is enabled
    if resume:
        logger.info("Checking for existing checkpoints to resume from...")
        checkpoint_data, checkpoint_path = checkpoint_manager.find_latest_generation_checkpoint(goal)
        
        if checkpoint_data:
            # Resume from checkpoint
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            reasoning_tree = checkpoint_data.get("reasoning_tree", {})
            last_processed_phase = checkpoint_data.get("last_processed_phase")
            last_processed_task = checkpoint_data.get("last_processed_task")
            
            if reasoning_tree:
                logger.info(f"Restored checkpoint with {len(reasoning_tree)} phases")
                # If we have phases already and no last_processed_phase, we've completed phase generation
                if not last_processed_phase and reasoning_tree:
                    # Get the last phase we'll process next
                    phases = list(reasoning_tree.keys())
                    if phases:
                        last_processed_phase = phases[-1]  # Start with the first phase
                        logger.info(f"Will resume by processing tasks for phase: {last_processed_phase}")
    
    try:
        if not constitution:
            raise PlanGenerationError("Cannot generate plan without a Project Constitution.")
            
        constitution_str = json.dumps(constitution, indent=2)

        # Select the appropriate LLM client
        _, _, call_with_retry = await select_llm_client(config, provider)
        
        # 3. Generate Phases if we don't have them
        if not reasoning_tree:
            logger.info("Generating phases...")
            phase_context = {"goal": goal, "constitution": constitution_str}
            phase_response = await call_with_retry(PHASE_GENERATION_PROMPT, phase_context, config)
            phases = phase_response.get("phases", [])
            
            if not phases:
                logger.error("Could not generate phases from Gemini response.")
                raise PlanGenerationError("Failed to generate phases from Gemini.")
            
            logger.info(f"Generated {len(phases)} phases.")
            
            # Initialize reasoning tree with empty entries for each phase
            reasoning_tree = {phase: {} for phase in phases}
            
            # Save checkpoint after phase generation
            checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                goal=goal, 
                current_state=reasoning_tree,
                last_processed_phase=None,  # We haven't started processing tasks yet
                last_processed_task=None
            )

        # 4. Generate Tasks for each Phase and Steps for each Task
        phases = list(reasoning_tree.keys())
        
        # Determine where to start based on checkpoint
        start_phase_idx = 0
        if last_processed_phase:
            try:
                # Find the index of the last processed phase
                start_phase_idx = phases.index(last_processed_phase)
                
                # If we have a last_processed_task and it exists in the current phase's tasks,
                # we need to continue from the next task
                if last_processed_task and last_processed_task in reasoning_tree[last_processed_phase]:
                    # Continue from next phase since we completed all tasks in this phase
                    start_phase_idx += 1
            except ValueError:
                # Phase not found, start from beginning
                logger.warning(f"Last processed phase '{last_processed_phase}' not found in phases, starting from beginning")
                start_phase_idx = 0
        
        for phase_idx in range(start_phase_idx, len(phases)):
            phase = phases[phase_idx]
            logger.info(f"Processing Phase: {phase} [{phase_idx + 1}/{len(phases)}]")
            
            # Skip if we already have tasks for this phase
            if reasoning_tree[phase] and all(isinstance(reasoning_tree[phase][task], list) for task in reasoning_tree[phase]):
                logger.info(f"Skipping phase '{phase}' as it already has complete tasks with steps")
                continue
            
            # Generate tasks for this phase if needed
            if not reasoning_tree[phase]:
                logger.info(f"Generating tasks for Phase: {phase}")
                task_context = {"goal": goal, "phase": phase, "constitution": constitution_str}
                task_response = await call_with_retry(TASK_GENERATION_PROMPT, task_context, config)
                tasks = task_response.get("tasks", [])
                
                if not tasks:
                    logger.warning(f"No tasks generated for phase '{phase}'. Continuing to next phase.")
                    # Create an empty dict for the phase
                    reasoning_tree[phase] = {}
                    # Save checkpoint after attempting task generation
                    checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                        goal=goal, 
                        current_state=reasoning_tree,
                        last_processed_phase=phase,
                        last_processed_task=None
                    )
                    continue
                
                logger.info(f"Generated {len(tasks)} tasks for phase '{phase}'")
                # Initialize each task with an empty list to be filled with steps
                reasoning_tree[phase] = {task: [] for task in tasks}
                
                # Save checkpoint after task generation
                checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                    goal=goal, 
                    current_state=reasoning_tree,
                    last_processed_phase=phase,
                    last_processed_task=None
                )
            
            # Generate steps for each task in this phase
            tasks = list(reasoning_tree[phase].keys())
            
            # Determine where to start for tasks based on checkpoint
            start_task_idx = 0
            if phase == last_processed_phase and last_processed_task:
                try:
                    # Find the index of the last processed task
                    start_task_idx = tasks.index(last_processed_task) + 1  # Start from the next task
                    if start_task_idx >= len(tasks):
                        # If we've processed all tasks in this phase, continue to the next phase
                        logger.info(f"All tasks in phase '{phase}' already processed")
                        continue
                except ValueError:
                    # Task not found, start from beginning of this phase
                    logger.warning(f"Last processed task '{last_processed_task}' not found in phase '{phase}', starting from first task")
                    start_task_idx = 0
            
            for task_idx in range(start_task_idx, len(tasks)):
                task = tasks[task_idx]
                logger.info(f"  Processing Task: {task} [{task_idx + 1}/{len(tasks)}]")
                
                # Skip if we already have steps for this task
                if reasoning_tree[phase][task]:
                    logger.info(f"  Skipping task '{task}' as it already has steps")
                    continue
                
                # Generate steps for this task
                logger.info(f"  Generating steps for Task: {task}")
                step_context = {"goal": goal, "phase": phase, "task": task, "constitution": constitution_str}
                
                try:
                    step_response = await call_with_retry(STEP_GENERATION_PROMPT, step_context, config)
                    steps = step_response.get("steps", [])
                    
                    if not steps:
                        logger.warning(f"No steps generated for task '{task}' in phase '{phase}'. Continuing to next task.")
                        # Initialize with empty list to mark as processed
                        reasoning_tree[phase][task] = []
                    else:
                        logger.info(f"  Generated {len(steps)} steps for task '{task}'")
                        reasoning_tree[phase][task] = steps
                except Exception as e:
                    logger.error(f"Error generating steps for task '{task}': {e}", exc_info=True)
                    # Mark the task as having an error by storing a special error indicator
                    reasoning_tree[phase][task] = [{"error": f"Failed to generate steps: {str(e)}"}]
                
                # Save checkpoint after processing each task
                checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                    goal=goal, 
                    current_state=reasoning_tree,
                    last_processed_phase=phase,
                    last_processed_task=task
                )
        
        # 5. Write Output JSON
        logger.info(f"Writing reasoning tree to {output_file}...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reasoning_tree, f, indent=2, ensure_ascii=False)
            logger.info("Planning process completed successfully.")
            
            # Delete checkpoint since we completed successfully
            if checkpoint_path:
                checkpoint_manager.delete_checkpoint(checkpoint_path)
        except IOError as e:
            logger.error(f"Error writing output file '{output_file}': {e}", exc_info=True)
            raise FileWriteError(f"Error writing output file '{output_file}': {e}")
        except TypeError as e:
            logger.error(f"Error serializing reasoning tree to JSON: {e}", exc_info=True)
            raise JsonSerializationError(f"Error serializing reasoning tree to JSON: {e}")
        
        return reasoning_tree, goal

    except ApiCallError as e:
        logger.error(f"API call failed during generation: {e}", exc_info=True)
        
        # Save our progress so far
        if reasoning_tree and goal:
            checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                goal=goal, 
                current_state=reasoning_tree,
                last_processed_phase=last_processed_phase,
                last_processed_task=last_processed_task
            )
            logger.info(f"Progress saved to checkpoint: {checkpoint_path}")
        
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during plan generation: {e}", exc_info=True)
        
        # Save our progress so far
        if reasoning_tree and goal:
            checkpoint_path = checkpoint_manager.save_generation_checkpoint(
                goal=goal, 
                current_state=reasoning_tree,
                last_processed_phase=last_processed_phase,
                last_processed_task=last_processed_task
            )
            logger.info(f"Progress saved to checkpoint: {checkpoint_path}")
        
        raise PlanGenerationError(f"An unexpected error occurred during plan generation: {e}") from e


async def main_workflow(task_file: str, output_file: str, validated_output_file: str, skip_qa: bool, config: Dict[str, Any], skip_resume: bool = False, provider: Optional[str] = None, validate_only: bool = False):
    """
    Orchestrates the full application workflow.
    """
    reasoning_tree: dict | None = None
    goal: str | None = None
    constitution: dict | None = None
    
    try:
        logger.info(f"Reading goal from: {task_file}")
        with open(task_file, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
            raise FileReadError(f"Task file '{task_file}' is empty.")
    except FileNotFoundError as e:
        raise PlannerFileNotFoundError(f"Task file not found: {e.filename}") from e

    try:
        # Determine providers for each agent
        default_provider = config.get('default_provider', 'gemini')
        architect_provider = provider or config.get('agents', {}).get('founding_architect', {}).get('provider', default_provider)
        planner_provider = provider or config.get('agents', {}).get('planner', {}).get('provider', default_provider)
        validator_provider = provider or config.get('agents', {}).get('qa_validator', {}).get('provider', default_provider)

        # Step 0: Generate or Load Project Constitution
        constitution_path = "project_constitution.json"
        if not skip_resume and os.path.exists(constitution_path):
            logger.info(f"Loading existing Project Constitution from {constitution_path}")
            with open(constitution_path, 'r', encoding='utf-8') as f:
                constitution = json.load(f)
        else:
            constitution = await generate_constitution(goal, config, architect_provider)

        if validate_only:
            logger.info("--- Running in Validation-Only Mode ---")
            try:
                logger.info(f"Loading existing plan from: {output_file}")
                with open(output_file, 'r', encoding='utf-8') as f:
                    reasoning_tree = json.load(f)
                if not reasoning_tree:
                    raise PlanValidationError("Reasoning tree is empty, cannot validate.")
            except FileNotFoundError as e:
                raise PlannerFileNotFoundError(f"Required file not found for validation: {e.filename}") from e
            except json.JSONDecodeError as e:
                raise JsonParsingError(f"Error parsing JSON from {output_file}: {e}") from e
        else:
            # Step 1: Generate the initial plan
            reasoning_tree, _ = await generate_plan(
                task_file=task_file,
                output_file=output_file,
                config=config,
                resume=not skip_resume,
                provider=planner_provider,
                constitution=constitution
            )

        # Ensure we have a plan to validate
        if not reasoning_tree:
            logger.error("No reasoning tree available to validate. Halting.")
            return

        # Step 2: Run QA Validation (if not skipped)
        if not skip_qa:
            logger.info("--- Starting QA Validation Step ---")
            if not goal:
                 # This case should ideally be caught by generate_plan raising an error
                 logger.error("Goal is missing, cannot run QA validation.")
                 raise PlanValidationError("Goal is missing, cannot run QA validation.")

            # Pass config down to QA validation function, including resume flag
            await run_qa_validation(
                input_path=output_file,
                output_path=validated_output_file,
                config=config,
                resume=not skip_resume,
                provider=validator_provider,
                constitution=constitution
            )
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoints if they exist."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run the Project Builder using a generated reasoning tree."
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=CONFIG['files'].get('default_project_dir', 'generated_project'), # Get from config or default
        help="Directory where the project will be built (default: generated_project or from config)."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=['gemini', 'anthropic', 'deepseek'],
        help="Choose which LLM provider to use (gemini, anthropic, deepseek). If not specified, auto-selects based on available API keys."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run only the QA validation on an existing reasoning tree. Requires --output-file to be set."
    )

    args = parser.parse_args()

    # Decide whether to run the planning workflow or the build workflow
    if args.build:
        # --- Project Build Workflow ---
        logger.info("Starting Project Build workflow...")
        try:
            # Determine which reasoning tree file to use
            # Use validated if it exists and QA wasn't skipped, otherwise use the initial output
            reasoning_tree_input = args.output_file
            if not args.skip_qa and os.path.exists(args.validated_output_file):
                reasoning_tree_input = args.validated_output_file
                logger.info(f"Using validated reasoning tree: {reasoning_tree_input}")
            elif not os.path.exists(reasoning_tree_input):
                 logger.error(f"Reasoning tree file not found for building: {reasoning_tree_input}")
                 raise PlannerFileNotFoundError(f"Reasoning tree file not found: {reasoning_tree_input}")
            else:
                 logger.info(f"Using initial reasoning tree: {reasoning_tree_input}")

            # Ensure project_dir is absolute
            project_dir_abs = os.path.abspath(args.project_dir)

            # Instantiate and run the builder
            # NOTE: ProjectBuilder currently uses simulation logic.
            # Real execution requires integrating tool calls.
            builder = ProjectBuilder(
                reasoning_tree_path=reasoning_tree_input,
                config_path='config/config.yaml', # Assuming config path relative to main.py location
                project_dir=project_dir_abs
            )
            builder.build() # This currently runs the simulation

            logger.info("Project Build workflow finished.")

        except (ProjectBuilderError, PlannerFileNotFoundError, ConfigError) as e:
             logger.critical(f"Project build failed: {e}", exc_info=True)
             sys.exit(1)
        except Exception as e:
             logger.critical(f"Unhandled exception during project build: {e}", exc_info=True)
             sys.exit(1)

    else:
        # --- Plan Generation/Validation Workflow ---
        logger.info("Starting Plan Generation/Validation workflow...")
        try:
            asyncio.run(main_workflow(
                task_file=args.task_file,
                output_file=args.output_file,
                validated_output_file=args.validated_output_file,
                skip_qa=args.skip_qa,
                config=CONFIG,
                skip_resume=args.no_resume,
                provider=args.provider,
                validate_only=args.validate_only
            ))
        except HierarchicalPlannerError as e:
            logger.critical(f"Application error during plan generation/validation: {e}", exc_info=True)
            sys.exit(1) # Exit after logging the specific error
        except Exception as e:
            logger.critical(f"Unhandled exception during plan generation/validation: {e}", exc_info=True)
            sys.exit(1) # Exit after logging the general error
