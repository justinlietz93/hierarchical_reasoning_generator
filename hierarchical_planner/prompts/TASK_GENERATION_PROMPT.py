
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