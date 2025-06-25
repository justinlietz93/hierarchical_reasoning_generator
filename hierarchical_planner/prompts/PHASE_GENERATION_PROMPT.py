
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