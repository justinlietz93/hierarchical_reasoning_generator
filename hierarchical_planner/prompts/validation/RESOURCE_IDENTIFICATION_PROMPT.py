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