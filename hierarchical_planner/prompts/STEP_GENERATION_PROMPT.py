
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