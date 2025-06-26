"""
Quality Assurance (QA) and Validation module for the Hierarchical Planner.

Provides functions to:
1. Validate the structural integrity of the generated plan JSON.
2. Analyze the plan using Gemini for goal alignment, logical sequence,
   clarity, resource identification, etc.
3. Annotate the plan JSON with QA findings.
"""
import json
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Tuple

# Local imports
from .checkpoint_manager import CheckpointManager # Import the checkpoint manager
from .llm_client_selector import select_llm_client

from .exceptions import (
    FileProcessingError, PlannerFileNotFoundError, FileReadError, FileWriteError,
    JsonSerializationError, JsonProcessingError, JsonParsingError, PlanValidationError, ApiCallError,
    ConfigError
)


# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Configuration (Removed hardcoded defaults) ---

# --- Prompt Templates for QA --- (Keep as is)

ALIGNMENT_CHECK_PROMPT = """
You are a meticulous project plan reviewer.
Your review must be consistent with the established Project Constitution.

Project Constitution:
{constitution}

Analyze the following step(s) within the context of the overall goal, phase, and task.

Overall Goal: "{goal}"
Current Phase: "{phase}"
Current Task: "{task}"
Step(s) to Review:
{steps_json}

Critique the following aspects:
1.  **Goal Alignment:** Do these steps logically contribute to achieving the stated task and, ultimately, the overall goal? Are there any steps that seem irrelevant or counter-productive?
2.  **Logical Sequence:** Is the order of the steps logical? Are there any missing prerequisite steps or obvious ordering issues?
3.  **Clarity & Actionability:** Are the prompts clear and specific enough for an AI coding agent to understand and execute?

Return your critique as a JSON object with keys "alignment_critique", "sequence_critique", and "clarity_critique". Provide concise feedback for each. If no issues are found, state that explicitly (e.g., "Steps appear well-aligned and logical.").
Example:
{{
  "alignment_critique": "Steps logically contribute to the task.",
  "sequence_critique": "Sequence seems correct.",
  "clarity_critique": "Step 2 could be more specific about the expected output format."
}}
"""

RESOURCE_IDENTIFICATION_PROMPT = """
You are an assistant analyzing prompts intended for an AI coding agent.
Your analysis must be consistent with the established Project Constitution.

Project Constitution:
{constitution}

Analyze the following prompt in the context of the overall goal, phase, and task.

Overall Goal: "{goal}"
Current Phase: "{phase}"
Current Task: "{task}"
Prompt to Analyze: "{prompt_text}"

Identify the following based *only* on the provided prompt text:
1.  **External Actions:** List any explicit instructions requiring interaction outside the current codebase (e.g., "Search the web for...", "Install library X", "Consult API documentation for Y").
2.  **Key Entities/Dependencies:** List specific functions, classes, variables, or file names mentioned that are likely defined in previous steps or need to be used in subsequent steps (focus on concrete names).
3.  **Technology Hints:** List any specific technologies, libraries, frameworks, or versions mentioned.

Return the analysis as a JSON object with keys "external_actions", "key_entities_dependencies", and "technology_hints". Each key should map to a list of strings. If nothing is identified for a category, return an empty list.
Example:
{{
  "external_actions": ["Search the web for Python GUI libraries", "Install 'requests' library"],
  "key_entities_dependencies": ["parse_input function", "parser.py file", "data variable"],
  "technology_hints": ["Python", "unittest library"]
}}
"""

# --- Validation Functions ---

def validate_plan_structure(plan_data: dict) -> list[str]:
    """
    Programmatically validates the basic structure of the reasoning tree JSON.

    Checks for correct types (dict, list) at each level (phases, tasks, steps)
    and verifies the format of individual step entries.

    Args:
        plan_data: The loaded JSON data as a Python dictionary representing the plan.

    Returns:
        A list of error message strings. An empty list indicates a valid structure.
        This function does not raise exceptions itself, allowing the caller
        (e.g., `run_validation`) to decide how to handle errors.
    """
    errors: list[str] = []
    if not isinstance(plan_data, dict):
        errors.append("Root level must be a dictionary (phases).")
        return errors # Cannot proceed if root is not dict

    for phase_name, tasks in plan_data.items():
        if not isinstance(tasks, dict):
            errors.append(f"Phase '{phase_name}' value must be a dictionary (tasks).")
            continue # Skip tasks if phase value is wrong type

        for task_name, steps in tasks.items():
            if not isinstance(steps, list):
                errors.append(f"Task '{task_name}' in Phase '{phase_name}' value must be a list (steps).")
                continue # Skip steps if task value is wrong type

            for i, step_obj in enumerate(steps):
                step_index = i + 1
                if not isinstance(step_obj, dict):
                    errors.append(f"Step {step_index} in Task '{task_name}', Phase '{phase_name}' must be a dictionary.")
                    continue

                # Allow for qa_info key, find the prompt key
                step_keys = list(step_obj.keys())
                
                # Handle new format with "id" and "description" fields
                if "description" in step_obj:
                    prompt_key = "description"
                    step_value = step_obj.get("description")
                    
                    # Validate id field if present
                    if "id" in step_obj:
                        step_id = step_obj.get("id")
                        if not isinstance(step_id, str) or not step_id:
                            errors.append(f"Step {step_index} 'id' in Task '{task_name}', Phase '{phase_name}' must be a non-empty string.")
                else:
                    # Handle old format where step key is dynamic (e.g., "step 1", "step 2")
                    prompt_key = next((k for k in step_keys if k != 'qa_info'), None)

                    if not prompt_key:
                         # If only qa_info exists or it's empty, it's an error
                         if len(step_keys) > 0 and all(k == 'qa_info' for k in step_keys):
                             logger.warning(f"Step {step_index} in Task '{task_name}', Phase '{phase_name}' only contains 'qa_info'. Assuming valid structure but no prompt.")
                             continue # Allow steps that might *only* have QA info after processing? Or error? Let's allow for now.
                         else:
                             errors.append(f"Step {step_index} in Task '{task_name}', Phase '{phase_name}' has no prompt key (e.g., 'step N' or 'description'). Found keys: {step_keys}")
                             continue

                    # Check prompt key format (optional, warning only) for old format
                    if not prompt_key.lower().startswith("step"):
                         logger.warning(f"Step {step_index} key ('{prompt_key}') in Task '{task_name}', Phase '{phase_name}' does not start with 'step'.")
                    
                    step_value = step_obj.get(prompt_key) # Use .get for safety

                # Check prompt value
                if not isinstance(step_value, str) or not step_value:
                    errors.append(f"Step {step_index} value ('{prompt_key}') in Task '{task_name}', Phase '{phase_name}' must be a non-empty string (the prompt).")

                # Check qa_info structure if present (optional)
                qa_info = step_obj.get('qa_info')
                if qa_info is not None and not isinstance(qa_info, dict):
                     errors.append(f"Step {step_index} 'qa_info' in Task '{task_name}', Phase '{phase_name}' must be a dictionary if present.")

    return errors

async def validate_steps(steps: List[Dict[str, Any]], goal: str, phase: str, task: str, config: Dict[str, Any], constitution: Dict[str, Any], agent_name: str) -> List[Dict[str, Any]]:
    """
    Validates a list of steps for a given task.
    """
    logger.info(f"      Validating {len(steps)} steps for Task: {task}")
    _, _, call_with_retry = await select_llm_client(config, agent_name)
    constitution_str = json.dumps(constitution, indent=2)

    for step_idx, step_obj in enumerate(steps, 1):
        step_keys = list(step_obj.keys())
        
        # Handle new format with "id" and "description" fields
        if "description" in step_obj:
            prompt_key = "description"
            step_prompt = step_obj.get("description", "")
        else:
            # Handle old format where step key is dynamic (e.g., "step 1", "step 2")
            prompt_key = next((k for k in step_keys if k != 'qa_info'), None)
            if not prompt_key:
                logger.info(f"        [SKIP] Step {step_idx}/{len(steps)}: No prompt key found")
                continue
            step_prompt = step_obj.get(prompt_key, "")
        
        if not step_prompt:
            logger.info(f"        [SKIP] Step {step_idx}/{len(steps)}: Empty prompt")
            continue

        logger.info(f"        [VALIDATING] Step {step_idx}/{len(steps)}: {prompt_key}")

        if "qa_info" not in step_obj:
            step_obj["qa_info"] = {}

        # Resource Identification
        if "resource_analysis" not in step_obj["qa_info"]:
            try:
                logger.info(f"        [API CALL] Step {step_idx}/{len(steps)}: Resource analysis...")
                resource_context = {
                    "goal": goal, "phase": phase, "task": task,
                    "prompt_text": step_prompt, "constitution": constitution_str
                }
                resource_analysis = await call_with_retry(
                    RESOURCE_IDENTIFICATION_PROMPT, resource_context, config, is_structured=True
                )
                step_obj["qa_info"]["resource_analysis"] = resource_analysis
                logger.info(f"        [COMPLETE] Step {step_idx}/{len(steps)}: Resource analysis done")
            except Exception as e:
                logger.error(f"Error during resource analysis for step {prompt_key}: {e}", exc_info=True)
                step_obj["qa_info"]["resource_analysis_error"] = str(e)

        # Alignment/Clarity Check
        if "step_critique" not in step_obj["qa_info"]:
            try:
                logger.info(f"        [API CALL] Step {step_idx}/{len(steps)}: Step critique...")
                alignment_context = {
                    "goal": goal, "phase": phase, "task": task,
                    "steps_json": json.dumps({prompt_key: step_prompt}, indent=2),
                    "constitution": constitution_str
                }
                alignment_critique = await call_with_retry(
                    ALIGNMENT_CHECK_PROMPT, alignment_context, config, is_structured=True
                )
                step_obj["qa_info"]["step_critique"] = alignment_critique
                logger.info(f"        [COMPLETE] Step {step_idx}/{len(steps)}: Step critique done")
            except Exception as e:
                logger.error(f"Error during step critique for step {prompt_key}: {e}", exc_info=True)
                step_obj["qa_info"]["step_critique_error"] = str(e)
    
    logger.info(f"      [VALIDATION COMPLETE] All {len(steps)} steps processed for Task: {task}")
    return steps

async def validate_tasks(tasks: List[str], goal: str, phase: str, config: Dict[str, Any], constitution: Dict[str, Any], agent_name: str) -> List[str]:
    """
    Validates a list of tasks for a given phase.
    """
    logger.info(f"    Validating {len(tasks)} tasks for Phase: {phase}")
    # This is a placeholder for now.
    # In a real implementation, you might have a prompt to check if the tasks are logical for the phase.
    return tasks

async def validate_phases(phases: List[str], goal: str, config: Dict[str, Any], constitution: Dict[str, Any], agent_name: str) -> List[str]:
    """
    Validates a list of phases for a given goal.
    """
    logger.info(f"  Validating {len(phases)} phases for Goal: {goal}")
    # This is a placeholder for now.
    # In a real implementation, you might have a prompt to check if the phases are logical for the goal.
    return phases

async def analyze_and_annotate_plan(plan_data: dict, goal: str, config: Dict[str, Any],
                                   constitution: Dict[str, Any],
                                   resume: bool = True,
                                   input_path: str = None,
                                   output_path: str = None,
                                   agent_name: str = "qa_validator") -> dict:
    """
    Uses an LLM to analyze alignment, identify resources, and annotate the plan.

    Iterates through each step in the plan, calling the validation logic.
    Adds the results (or error messages) under a 'qa_info' key within each step object.
    
    Supports checkpoint and resume functionality if enabled.
    """
    annotated_plan = plan_data # Modify in place
    
    # Initialize checkpoint manager and variables for tracking progress
    checkpoint_manager = CheckpointManager()
    checkpoint_path = ""
    last_processed_phase = None
    last_processed_task = None
    last_processed_step_index = -1
    
    # Try to resume from checkpoint if enabled
    if resume and input_path and output_path:
        logger.info("Checking for existing QA checkpoints to resume from...")
        checkpoint_data, checkpoint_path = checkpoint_manager.find_latest_qa_checkpoint(input_path)
        
        if checkpoint_data:
            # Resume from checkpoint
            logger.info(f"Resuming QA analysis from checkpoint: {checkpoint_path}")
            
            # Check if the saved checkpoint data is for the same output path
            if checkpoint_data.get("output_path") == output_path:
                # Restore annotated plan from checkpoint
                saved_plan = checkpoint_data.get("validated_data")
                if saved_plan:
                    annotated_plan = saved_plan
                    logger.info("Restored annotated plan from checkpoint")
                
                # Restore progress tracking
                last_processed_phase = checkpoint_data.get("last_phase")
                last_processed_task = checkpoint_data.get("last_task")
                last_processed_step_index = checkpoint_data.get("last_step_index", -1)
                
                if last_processed_phase and last_processed_task:
                    logger.info(f"Will resume QA analysis from Phase: '{last_processed_phase}', "
                               f"Task: '{last_processed_task}', Step index: {last_processed_step_index + 1}")
            else:
                logger.warning("Found checkpoint is for a different output path, starting fresh")
                
    # Iterate through the plan phases, tasks, and steps
    phase_names = list(annotated_plan.keys())
    start_phase_idx = 0
    
    # Find the index of the last processed phase (if resuming)
    if last_processed_phase in phase_names:
        start_phase_idx = phase_names.index(last_processed_phase)
        
    # Iterate through phases
    for phase_idx, phase_name in enumerate(phase_names[start_phase_idx:], start_phase_idx):
        logger.info(f"  Analyzing Phase: {phase_name}")
        tasks = annotated_plan[phase_name]
        
        # Get list of task names in this phase
        task_names = list(tasks.keys())
        start_task_idx = 0
        
        # If we're resuming in this phase, find the index of the last processed task
        if phase_name == last_processed_phase and last_processed_task in task_names:
            start_task_idx = task_names.index(last_processed_task)
        
        # Iterate through tasks
        for task_idx, task_name in enumerate(task_names[start_task_idx:], start_task_idx):
            logger.info(f"    Analyzing Task: {task_name}")
            steps = tasks[task_name]
            
            if not steps:
                logger.info("      Skipping task analysis (no steps).")
                continue
            
            # Determine the starting step index
            start_step_idx = 0
            if (phase_name == last_processed_phase and 
                task_name == last_processed_task and 
                last_processed_step_index >= 0):
                start_step_idx = last_processed_step_index + 1  # Start from the next step
                if start_step_idx >= len(steps):
                    # All steps in this task were already processed
                    logger.info(f"      All steps in task '{task_name}' already processed")
                    continue

            # --- Step-level Analysis ---
            for step_idx in range(start_step_idx, len(steps)):
                step_obj = steps[step_idx]
                step_keys = list(step_obj.keys())
                prompt_key = next((k for k in step_keys if k != 'qa_info'), None)

                if not prompt_key:
                    logger.warning(f"      Skipping analysis for step {step_idx+1} in Task '{task_name}' - no prompt key found.")
                    continue

                step_prompt = step_obj.get(prompt_key, "") # Use get for safety
                if not step_prompt:
                     logger.warning(f"      Skipping analysis for step {prompt_key} in Task '{task_name}' - empty prompt.")
                     continue

                logger.info(f"      Analyzing Step: {prompt_key}")

                # Initialize QA info for the step if not present
                if "qa_info" not in step_obj:
                    step_obj["qa_info"] = {}

                # Select the appropriate LLM client
                _, _, call_with_retry = await select_llm_client(config, agent_name)
                
                # 1. Resource/Action Identification (skip if already done)
                if "resource_analysis" not in step_obj["qa_info"] and "resource_analysis_error" not in step_obj["qa_info"]:
                    try:
                        constitution_str = json.dumps(constitution, indent=2)
                        resource_context = {
                            "goal": goal,
                            "phase": phase_name,
                            "task": task_name,
                            "prompt_text": step_prompt,
                            "constitution": constitution_str
                        }
                        # Pass config to retry function
                        resource_analysis = await call_with_retry(
                            RESOURCE_IDENTIFICATION_PROMPT, resource_context, config, is_structured=True
                        )
                        step_obj["qa_info"]["resource_analysis"] = resource_analysis
                        logger.debug(f"        Resource analysis complete for {prompt_key}.")
                    except ApiCallError as e:
                        # Log API errors but allow processing to continue for other steps
                        logger.error(f"        API call failed during resource analysis for step {prompt_key}: {e}", exc_info=True)
                        step_obj["qa_info"]["resource_analysis_error"] = f"API Error: {e}"
                    except Exception as e:
                        # Catch other unexpected errors during this step's analysis
                        logger.error(f"        Unexpected error during resource analysis for step {prompt_key}: {e}", exc_info=True)
                        step_obj["qa_info"]["resource_analysis_error"] = f"Unexpected Error: {e}"

                    # Save checkpoint after each resource analysis
                    if input_path and output_path:
                        checkpoint_path = checkpoint_manager.save_qa_checkpoint(
                            input_path=input_path,
                            output_path=output_path,
                            validated_data=annotated_plan,
                            last_phase=phase_name,
                            last_task=task_name,
                            last_step_index=step_idx
                        )

                # 2. Alignment/Clarity Check (Individual Step) (skip if already done)
                if "step_critique" not in step_obj["qa_info"] and "step_critique_error" not in step_obj["qa_info"]:
                    try:
                        constitution_str = json.dumps(constitution, indent=2)
                        alignment_context = {
                            "goal": goal,
                            "phase": phase_name,
                            "task": task_name,
                            "steps_json": json.dumps({prompt_key: step_prompt}, indent=2), # Analyze one step
                            "constitution": constitution_str
                        }
                         # Pass config to retry function
                        alignment_critique = await call_with_retry(
                            ALIGNMENT_CHECK_PROMPT, alignment_context, config, is_structured=True
                        )
                        step_obj["qa_info"]["step_critique"] = alignment_critique
                        logger.debug(f"        Step critique complete for {prompt_key}.")
                    except ApiCallError as e:
                        logger.error(f"        API call failed during step critique for step {prompt_key}: {e}", exc_info=True)
                        step_obj["qa_info"]["step_critique_error"] = f"API Error: {e}"
                    except Exception as e:
                        logger.error(f"        Unexpected error during step critique for step {prompt_key}: {e}", exc_info=True)
                        step_obj["qa_info"]["step_critique_error"] = f"Unexpected Error: {e}"

                    # Save checkpoint after each step critique
                    if input_path and output_path:
                        checkpoint_path = checkpoint_manager.save_qa_checkpoint(
                            input_path=input_path,
                            output_path=output_path,
                            validated_data=annotated_plan,
                            last_phase=phase_name,
                            last_task=task_name,
                            last_step_index=step_idx
                        )

                # Update the last processed step index
                last_processed_step_index = step_idx
                
                # Add a small delay to avoid hitting API rate limits too quickly
                # Consider making this delay configurable
                api_delay = config.get('api', {}).get('delay_between_qa_calls_sec', 1)
                await asyncio.sleep(api_delay)
                
    # Delete the checkpoint since we completed successfully
    if checkpoint_path:
        checkpoint_manager.delete_checkpoint(checkpoint_path)

    return annotated_plan


# --- Main Execution ---

async def run_validation(input_path: str, output_path: str, config: Dict[str, Any], resume: bool = True, agent_name: str = "qa_validator", constitution: Optional[Dict[str, Any]] = None):
    """
    Orchestrates the QA validation workflow.

    Loads the plan from `input_path`, validates its structure, loads the
    original goal, analyzes and annotates the plan using Gemini, and saves
    the annotated result to `output_path`.

    Args:
        input_path: Absolute path to the input plan JSON file.
        output_path: Absolute path to save the validated/annotated plan JSON file.
        config: The application configuration dictionary.
        resume: Whether to attempt to resume from a checkpoint if available.

    Raises:
        PlannerFileNotFoundError: If the input plan file or the goal file cannot be found.
        FileReadError: If the input plan or goal file cannot be read.
        JsonParsingError: If the input plan file contains invalid JSON.
        PlanValidationError: If the plan structure validation fails.
        ConfigError: If the configuration is missing necessary keys (e.g., 'files.default_task').
        FileWriteError: If the output file cannot be written.
        JsonSerializationError: If the annotated plan cannot be serialized to JSON.
        ApiCallError: If a critical API call fails during the analysis phase.
        Exception: Catches and logs other unexpected errors during the process.
    """
    logger.info(f"Starting QA validation process for: {input_path}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    checkpoint_path = ""

    # 1. Load Plan
    try:
        logger.info(f"Loading plan from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        logger.info("Plan loaded successfully.")
    except FileNotFoundError: # Built-in
        logger.error(f"Input plan file '{input_path}' not found.")
        raise PlannerFileNotFoundError(f"Input plan file '{input_path}' not found.") from None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from '{input_path}': {e}", exc_info=True)
        raise JsonParsingError(f"Failed to parse JSON from '{input_path}': {e}") from e
    except IOError as e:
        logger.error(f"Error reading input file '{input_path}': {e}", exc_info=True)
        raise FileReadError(f"Error reading input file '{input_path}': {e}") from e


    # 2. Validate Structure
    logger.info("Validating plan structure...")
    structure_errors = validate_plan_structure(plan_data)
    if structure_errors:
        error_details = "\n".join([f"- {err}" for err in structure_errors])
        logger.error(f"Structural validation failed:\n{error_details}")
        raise PlanValidationError(f"Plan structure validation failed:\n{error_details}")
    else:
        logger.info("Plan structure validation successful.")

    # 3. Load Goal for Analysis
    # Determine goal file path relative to the input plan file's directory
    # Use the 'default_task' filename from config, assuming it's in the same dir as input_path
    try:
        task_filename = os.path.basename(config['files']['default_task'])
        goal_file_path = os.path.join(os.path.dirname(input_path) or '.', task_filename)
    except KeyError:
         logger.error("Configuration missing 'files.default_task' needed to locate goal file.")
         raise ConfigError("Configuration missing 'files.default_task' needed to locate goal file.")

    goal = None
    try:
        logger.info(f"Loading goal for analysis from: {goal_file_path}")
        with open(goal_file_path, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
             logger.error(f"Goal file '{goal_file_path}' is empty.")
             raise FileReadError(f"Goal file '{goal_file_path}' is empty, cannot proceed with analysis.")
        logger.info(f"Goal '{goal[:100]}...' loaded for analysis.")
    except FileNotFoundError: # Built-in
        logger.error(f"Goal file '{goal_file_path}' not found.")
        raise PlannerFileNotFoundError(f"Goal file '{goal_file_path}' not found, cannot proceed with analysis.") from None
    except IOError as e:
        logger.error(f"Error reading goal file '{goal_file_path}': {e}", exc_info=True)
        raise FileReadError(f"Error reading goal file '{goal_file_path}': {e}") from e


    # 4. Analyze and Annotate
    logger.info("Starting plan analysis and annotation (this may take time)...")
    try:
        if not constitution:
            constitution_path = "project_constitution.json"
            logger.info(f"Constitution not provided, loading from {constitution_path}")
            try:
                with open(constitution_path, 'r', encoding='utf-8') as f:
                    constitution = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise PlanValidationError(f"Could not load a valid project constitution from {constitution_path} for validation.") from e

        # Pass config down and include paths for checkpointing
        annotated_plan = await analyze_and_annotate_plan(
            plan_data,
            goal,
            config,
            constitution,
            resume=resume,
            input_path=input_path,
            output_path=output_path,
            agent_name=agent_name
        )
        logger.info("Plan analysis and annotation complete.")
    except ApiCallError:
        # Let API call errors from analysis propagate up if they weren't handled internally
        logger.error("A critical API call failed during analysis.")
        raise # Re-raise ApiCallError
    except Exception as e:
        # Catch other unexpected errors during analysis
        logger.error(f"An unexpected error occurred during plan analysis: {e}", exc_info=True)
        # Decide whether to raise or just log and save partially annotated plan
        # For now, let's log and continue to save what we have
        logger.warning("Saving potentially partially annotated plan due to analysis error.")
        annotated_plan = plan_data # Use potentially partially modified data


    # 5. Save Validated Plan
    try:
        logger.info(f"Saving validated plan to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_plan, f, indent=2, ensure_ascii=False)
        logger.info("Validated plan saved successfully.")
        
        # Delete checkpoint since we completed successfully
        if checkpoint_path:
            checkpoint_manager.delete_checkpoint(checkpoint_path)
    except IOError as e:
        logger.error(f"Error saving validated plan to '{output_path}': {e}", exc_info=True)
        raise FileWriteError(f"Error saving validated plan to '{output_path}': {e}") from e
    except TypeError as e:
        logger.error(f"Error serializing annotated plan to JSON: {e}", exc_info=True)
        raise JsonSerializationError(f"Error serializing annotated plan to JSON: {e}") from e


# Removed __main__ block as this module should be called from main.py
