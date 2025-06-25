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