import json
import asyncio
from gemini_client import call_gemini_with_retry # Reusing the retry logic

# --- Configuration ---
VALIDATED_OUTPUT_FILE = "reasoning_tree_validated.json"
INPUT_FILE = "reasoning_tree.json" # Default input, can be overridden

# --- Prompt Templates for QA ---

ALIGNMENT_CHECK_PROMPT = """
You are a meticulous project plan reviewer.
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

def validate_plan_structure(plan_data: dict) -> list:
    """
    Programmatically validates the basic structure of the reasoning tree.

    Args:
        plan_data: The loaded JSON data as a Python dictionary.

    Returns:
        A list of error messages. An empty list indicates a valid structure.
    """
    errors = []
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
                if len(step_obj) != 1:
                    errors.append(f"Step {step_index} dictionary in Task '{task_name}', Phase '{phase_name}' must have exactly one key-value pair.")
                    continue
                step_key = list(step_obj.keys())[0]
                step_value = list(step_obj.values())[0]
                if not step_key.lower().startswith("step"):
                     errors.append(f"Step {step_index} key ('{step_key}') in Task '{task_name}', Phase '{phase_name}' should start with 'step'.")
                if not isinstance(step_value, str) or not step_value:
                    errors.append(f"Step {step_index} value ('{step_key}') in Task '{task_name}', Phase '{phase_name}' must be a non-empty string (the prompt).")

    return errors

async def analyze_and_annotate_plan(plan_data: dict, goal: str) -> dict:
    """
    Uses Gemini to analyze alignment, identify resources, and annotate the plan.

    Args:
        plan_data: The structurally validated plan data.
        goal: The overall goal string.

    Returns:
        The annotated plan data.
    """
    annotated_plan = plan_data # Start with the original data

    for phase_name, tasks in annotated_plan.items():
        print(f"  Analyzing Phase: {phase_name}")
        for task_name, steps in tasks.items():
            print(f"    Analyzing Task: {task_name}")
            if not steps:
                print("      Skipping task analysis (no steps).")
                continue

            # --- Task-level Alignment Check (Optional - could analyze all steps together) ---
            # For simplicity, we'll analyze step-by-step for resources first,
            # then potentially do a broader alignment check if needed.

            # --- Step-level Analysis ---
            for i, step_obj in enumerate(steps):
                step_key = list(step_obj.keys())[0]
                step_prompt = step_obj[step_key]
                print(f"      Analyzing Step: {step_key}")

                # Initialize QA info for the step
                if "qa_info" not in step_obj:
                    step_obj["qa_info"] = {}

                # 1. Resource/Action Identification
                try:
                    resource_context = {
                        "goal": goal,
                        "phase": phase_name,
                        "task": task_name,
                        "prompt_text": step_prompt
                    }
                    resource_analysis = await call_gemini_with_retry(
                        RESOURCE_IDENTIFICATION_PROMPT, resource_context, is_structured=True
                    )
                    step_obj["qa_info"]["resource_analysis"] = resource_analysis
                    print(f"        Resource analysis complete.")
                except Exception as e:
                    print(f"        Error during resource analysis for step {step_key}: {e}")
                    step_obj["qa_info"]["resource_analysis_error"] = str(e)

                # 2. Alignment/Clarity Check (Individual Step)
                try:
                    alignment_context = {
                        "goal": goal,
                        "phase": phase_name,
                        "task": task_name,
                        "steps_json": json.dumps({step_key: step_prompt}, indent=2) # Analyze one step
                    }
                    alignment_critique = await call_gemini_with_retry(
                        ALIGNMENT_CHECK_PROMPT, alignment_context, is_structured=True
                    )
                    step_obj["qa_info"]["step_critique"] = alignment_critique
                    print(f"        Step critique complete.")
                except Exception as e:
                    print(f"        Error during step critique for step {step_key}: {e}")
                    step_obj["qa_info"]["step_critique_error"] = str(e)

                # Add a small delay to avoid hitting API rate limits too quickly
                await asyncio.sleep(1) # Adjust as needed

    return annotated_plan


# --- Main Execution ---

async def run_validation(input_path: str = INPUT_FILE, output_path: str = VALIDATED_OUTPUT_FILE):
    """Loads, validates, analyzes, annotates, and saves the plan."""
    print(f"Starting QA validation process for: {input_path}")

    # 1. Load Plan
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        print("Plan loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input plan file '{input_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from '{input_path}': {e}")
        return

    # 2. Validate Structure
    print("Validating plan structure...")
    structure_errors = validate_plan_structure(plan_data)
    if structure_errors:
        print("Structural validation failed:")
        for error in structure_errors:
            print(f"- {error}")
        # Decide whether to stop or continue with analysis despite structural errors
        # For now, let's stop if the structure is fundamentally broken.
        print("Aborting QA process due to structural errors.")
        return
    else:
        print("Plan structure validation successful.")

    # 3. Analyze and Annotate (Requires Goal - how to get it?)
    # We need the original goal. Let's assume it's passed or read from task.txt
    # For now, let's add a placeholder. This needs refinement in integration.
    try:
        # Attempt to read goal from task.txt relative to the input file's dir
        goal_file_path = os.path.join(os.path.dirname(input_path) or '.', 'task.txt')
        with open(goal_file_path, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
             raise ValueError("Goal file 'task.txt' is empty.")
        print(f"Goal '{goal}' loaded for analysis.")
    except Exception as e:
        print(f"Error loading goal from '{goal_file_path}': {e}")
        print("Cannot proceed with content analysis without the goal.")
        # Optionally save just the structurally validated file?
        return

    print("Starting plan analysis and annotation (this may take time)...")
    try:
        annotated_plan = await analyze_and_annotate_plan(plan_data, goal)
        print("Plan analysis and annotation complete.")
    except Exception as e:
        print(f"An error occurred during plan analysis: {e}")
        # Decide how to handle partial annotation
        print("Saving potentially partially annotated plan.")
        annotated_plan = plan_data # Revert to original data or keep partial? For now, keep partial.

    # 4. Save Validated Plan
    try:
        print(f"Saving validated plan to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_plan, f, indent=2, ensure_ascii=False)
        print("Validated plan saved successfully.")
    except Exception as e:
        print(f"Error saving validated plan to '{output_path}': {e}")

if __name__ == "__main__":
    import os
    # Basic check for API key before running analysis
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: The GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the QA validation script.")
    else:
        # Example: Run validation on the default input file
        # Assumes reasoning_tree.json and task.txt are in the same directory
        # as qa_validator.py when run directly.
        input_file = INPUT_FILE
        output_file = VALIDATED_OUTPUT_FILE
        asyncio.run(run_validation(input_file, output_file))
