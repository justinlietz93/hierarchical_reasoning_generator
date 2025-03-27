import asyncio
import json
import os
import argparse # For command-line arguments
from gemini_client import generate_structured_content, generate_content
from qa_validator import run_validation as run_qa_validation # Import the QA function

# --- Configuration ---
DEFAULT_TASK_FILE = "task.txt"
DEFAULT_OUTPUT_FILE = "reasoning_tree.json"
DEFAULT_VALIDATED_OUTPUT_FILE = "reasoning_tree_validated.json"
MAX_RETRIES = 3 # Number of retries for API calls

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

# --- Helper Functions ---

async def call_gemini_with_retry(prompt_template: str, context: dict, is_structured: bool = True):
    """Calls the appropriate Gemini client function with retries."""
    prompt = prompt_template.format(**context)
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            if is_structured:
                # Use generate_structured_content for JSON expected responses
                response = await generate_structured_content(prompt)
            else:
                 # Use generate_content for plain text (though we mostly expect JSON here)
                 # This branch might not be used if all prompts request JSON
                response = await generate_content(prompt)
            return response
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
            else:
                print(f"Max retries reached for prompt:\n{prompt}")
                raise last_exception # Re-raise the last exception after all retries fail

# --- Main Logic ---

async def generate_plan(task_file: str, output_file: str):
    """Generates the initial reasoning tree JSON."""
    print("Starting hierarchical planning process...")

    # 1. Read Goal
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            goal = f.read().strip()
        if not goal:
            print(f"Error: Task file '{task_file}' is empty.")
            return None, None # Return None for tree and goal if error
        print(f"Goal read from {task_file}: {goal}")
    except FileNotFoundError:
        print(f"Error: Task file '{task_file}' not found.")
        return None, None

    reasoning_tree = {}
    script_dir = os.path.dirname(__file__) or '.' # Handle running from root

    try:
        # 2. Generate Phases
        print("Generating phases...")
        phase_context = {"goal": goal}
        phase_response = await call_gemini_with_retry(PHASE_GENERATION_PROMPT, phase_context)
        phases = phase_response.get("phases", [])
        if not phases:
            print("Error: Could not generate phases from Gemini.")
            return
        print(f"Generated {len(phases)} phases.")

        # 3. Generate Tasks for each Phase
        for phase in phases:
            print(f"\nGenerating tasks for Phase: {phase}")
            task_context = {"goal": goal, "phase": phase}
            task_response = await call_gemini_with_retry(TASK_GENERATION_PROMPT, task_context)
            tasks = task_response.get("tasks", [])
            if not tasks:
                print(f"Warning: Could not generate tasks for phase '{phase}'. Skipping.")
                reasoning_tree[phase] = {} # Add phase even if no tasks generated
                continue
            print(f"Generated {len(tasks)} tasks for this phase.")
            reasoning_tree[phase] = {}

            # 4. Generate Steps for each Task
            for task in tasks:
                print(f"  Generating steps for Task: {task}")
                step_context = {"goal": goal, "phase": phase, "task": task}
                step_response = await call_gemini_with_retry(STEP_GENERATION_PROMPT, step_context)
                steps = step_response.get("steps", [])
                if not steps:
                     print(f"Warning: Could not generate steps for task '{task}'. Skipping.")
                     reasoning_tree[phase][task] = [] # Add task even if no steps generated
                     continue
                print(f"  Generated {len(steps)} steps for this task.")
                reasoning_tree[phase][task] = steps

        # 5. Write Output JSON
        output_path = os.path.join(script_dir, output_file)
        print(f"\nWriting initial reasoning tree to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reasoning_tree, f, indent=2, ensure_ascii=False)
        print("Initial planning process completed successfully.")
        return reasoning_tree, goal # Return the generated tree and goal

    except Exception as e:
        print(f"\nAn error occurred during the initial planning process: {e}")
        # Optionally save partial results if needed
        # partial_output_path = os.path.join(script_dir, "partial_" + output_file)
        # with open(partial_output_path, 'w', encoding='utf-8') as f:
        #     json.dump(reasoning_tree, f, indent=2, ensure_ascii=False)
        # print(f"Partial results saved to {partial_output_path}")
        return None, goal # Return None for tree on error, but keep goal if read


async def main_workflow(task_file: str, output_file: str, validated_output_file: str, skip_qa: bool):
    """Orchestrates the full workflow: plan generation and optional QA."""

    # Step 1: Generate the initial plan
    reasoning_tree, goal = await generate_plan(task_file, output_file)

    if reasoning_tree is None:
        print("Plan generation failed. Exiting.")
        return

    # Step 2: Run QA Validation (if not skipped)
    if not skip_qa:
        print("\n--- Starting QA Validation Step ---")
        script_dir = os.path.dirname(__file__) or '.'
        initial_plan_path = os.path.join(script_dir, output_file)
        validated_plan_path = os.path.join(script_dir, validated_output_file)

        # Ensure the goal is available for the QA step
        if not goal:
             print("Error: Goal was not loaded correctly, cannot run QA validation.")
             return

        try:
            # run_qa_validation expects paths relative to where it's run or absolute
            # Assuming main.py and qa_validator.py are in the same dir, relative paths should work
            await run_qa_validation(input_path=initial_plan_path, output_path=validated_plan_path)
            print("--- QA Validation Step Completed ---")
        except Exception as e:
            print(f"An error occurred during QA validation: {e}")
            print("--- QA Validation Step Failed ---")
            # Decide if failure here should stop everything
    else:
        print("\n--- Skipping QA Validation Step ---")

    print("\nWorkflow finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally validate a hierarchical reasoning plan.")
    parser.add_argument(
        "--task-file",
        type=str,
        default=DEFAULT_TASK_FILE,
        help=f"Path to the file containing the high-level task (default: {DEFAULT_TASK_FILE})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save the initial reasoning tree JSON (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--validated-output-file",
        type=str,
        default=DEFAULT_VALIDATED_OUTPUT_FILE,
        help=f"Path to save the validated reasoning tree JSON (default: {DEFAULT_VALIDATED_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip the QA validation and annotation step."
    )

    args = parser.parse_args()

    # Check for API key before running
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: The GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        print("You can set it in your environment or create a '.env' file in the script's directory:")
        print("GEMINI_API_KEY='YOUR_API_KEY_HERE'")
    else:
        asyncio.run(main_workflow(
            task_file=args.task_file,
            output_file=args.output_file,
            validated_output_file=args.validated_output_file,
            skip_qa=args.skip_qa
        ))
