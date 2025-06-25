"""
Parser for extracting structured information from persona card text using an LLM.

This module uses an LLM to understand the structure and content of plain text 
persona cards and converts them into structured hierarchical data (JSON).
"""

import json
import logging
import os
import sys
from typing import Dict, Any, Optional

# Ensure the package can be imported when run as a script
from pathlib import Path
import sys

# Add project root to sys.path to aid imports when run directly or as module
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use absolute imports assuming the package is installed/accessible
try:
    # Import the exceptions
    from ..exceptions import HierarchicalPlannerError, ApiCallError, JsonParsingError, ApiResponseError
    # Import the LLM selector
    from .llm_selector import select_llm_client
    # Import the LLM clients for initialization
    from .. import gemini_client
    from .. import anthropic_client
except ImportError as e:
    print(f"Import Error in parser.py: {e}. Ensure hierarchical_planner package is installed or in PYTHONPATH.")
    # Define dummy classes/exceptions if imports fail completely
    gemini_client = None
    anthropic_client = None
    HierarchicalPlannerError = Exception
    ApiCallError = Exception
    JsonParsingError = Exception
    ApiResponseError = Exception
    select_llm_client = None

# Configure logging
logger = logging.getLogger(__name__)

# --- LLM Prompt Template ---

PERSONA_PARSING_PROMPT = """
You are an expert text structure analyzer. Your task is to parse the provided Persona Card text and convert it into a structured JSON format.

The persona card uses headings (like Roman numerals I., II.), subheadings, and potentially bullet points or nested lists to define different aspects of a persona.

Analyze the following Persona Card text:
```text
{persona_card_text}
```

**Instructions:**

1.  **Identify the main title** of the persona card if present (usually the first prominent line). Include it in the JSON output under the `title` key.
2.  **Identify the Persona's Name:** Determine the primary name or identifier of the persona being described (e.g., "Prometheus"). Include this name as a string under the top-level `persona_name` key.
3.  **Identify the main sections** indicated by headings (e.g., "I. Core Identity & Origin", "II. Mission & Motivation", "VII. Action Protocol"). Use the text following the Roman numeral (e.g., "Core Identity & Origin") as the key for the `sections` dictionary.
4.  **Identify any top-level instructions** or guidelines that apply to the overall persona, often appearing near the beginning or end without a specific section heading. If found, include this text under a top-level `instructions` key in the JSON output.
5.  **Generate Personality Profile:** Based *only* on the persona description provided in the text, select the 5 to 7 most defining personality concepts from the list provided below the prompt (e.g., Emotionality, Resilience, Intellect, Duty, Empathy, Integrity, Skepticism). For each selected concept, choose the single word from its associated slider that best represents the persona. Add this profile as a dictionary under the top-level `personality_profile` key (e.g., `"personality_profile": {{"Emotionality": "Calm", "Resilience": "Resilient", ...}}`).
6.  **Identify specific requirements for response output format or style**. If found, include this text under a top-level `response_output_requirements` key.
7.  **Identify any mention of available tools or capabilities**. If found, include this text or list under a top-level `tools_available` key.
8.  **For each main section:**
    *   Extract the **primary textual content** directly under the main heading but before any subheadings. Store this text in the `content` field for that section. If there's no text before subheadings, this field can be empty or omitted.
    *   Identify any **subsections** within the main section (e.g., "Name:", "Nature:", "Primary Objective:"). Use the subheading text (before the colon/period) as the key for the `subsections` dictionary.
    *   For each subsection, extract its **full textual content**.
    *   If the subsection content primarily consists of bullet points (using '-', '*', '•', or numbers like '1.'), represent the text of *each bullet point* as a separate string in a list under the `items` key. Include any introductory text before the bullets in the `content` field for the subsection.
    *   If the subsection content is primarily paragraph text (not bullets), store the full text in the `content` field for that subsection and omit the `items` key.
5.  **Output Format:** Return *only* a valid JSON object with the following structure, ensuring all relevant text content is included:

```json
{{
  "title": "Overall Persona Card Title (string, optional)",
  "instructions": "Overall instructions or guidelines for the persona (string, optional)",
  "sections": {{
    "Section Title 1": {{ // e.g., "Core Identity & Origin"
      "content": "Main content for Section 1...",
      "subsections": {{
        "Subsection Title 1.1": {{ // e.g., "Name"
          "content": "Content for Subsection 1.1...",
          "items": [ // Only if bullet points exist
            "Bullet point 1...",
            "Bullet point 2..."
          ]
        }},
        "Subsection Title 1.2": {{ // e.g., "Nature"
          "content": "Content for Subsection 1.2..."
        }}
        // ... more subsections
      }}
    }},
    "Section Title 2": {{ // e.g., "Mission & Motivation"
      "content": "...",
      "subsections": {{ ... }}
    }}
    // ... more sections
  }},
  "personality_profile": {{ // Generated based on analysis and the provided trait list
      "TraitConcept1": "SelectedSliderWord1", // e.g., "Emotionality": "Calm"
      "TraitConcept2": "SelectedSliderWord2", // e.g., "Resilience": "Resilient"
      // ... 5-7 key traits
  }},
  "response_output_requirements": "Specific instructions on how the persona should format its responses (string, optional)",
  "tools_available": "Description or list of tools the persona can use (string or list, optional)"
}}
```

**Important:**
*   Include the top-level `persona_name` key with the identified name.
*   Generate the `personality_profile` dictionary by selecting 5-7 relevant traits from the list below the prompt and choosing the best fitting slider word for each based *only* on the input text.
*   Preserve the original titles for sections and subsections accurately as keys in the JSON.
*   Include **all relevant textual content** within the appropriate `content` or `items` fields.
*   Handle nested structures correctly.
*   If a section/subsection has only bullets, `content` can be empty/omitted.
*   If a section/subsection has only paragraph text, omit `items`.
*   Include the top-level `instructions`, `response_output_requirements`, and `tools_available` keys *only if* relevant text is found in the input card.
*   Ensure the output is a single, valid JSON object and nothing else. No explanations before or after.

```json
""" # The LLM should continue the JSON object here

# --- Personality Trait List for Reference (for LLM) ---
# 1. Neuroticism: Emotionality (Stoic - Calm - Temperate - Sensitive - Passionate - Intense), Resilience (Fragile - Steady - Resilient - Tough - Unbreakable), Stability (Volatile - Balanced - Stable - Calm - Serene)
# 2. Extraversion: Physicality (Dormant - Subdued - Active - Energetic - Vigorous - Tireless), Sociability (Reclusive - Reserved - Amiable - Sociable - Outgoing - Gregarious), Assertiveness (Passive - Timid - Confident - Assertive - Dominant - Commanding), Proactivity (Passive - Reactive - Engaged - Proactive - Driven - Relentless)
# 3. Openness: Intellect (Concrete - Average - Bright - Clever - Brilliant - Genius), Imagination (Literal - Practical - Inventive - Imaginative - Visionary - Fantastical), Curiosity (Apathetic - Indifferent - Interested - Curious - Eager - Probing), Analytical Depth (Superficial - Observant - Perceptive - Insightful - Deep - Profound), Adaptability (Rigid - Stubborn - Flexible - Adaptable - Agile - Fluid)
# 4. Conscientiousness: Duty (Lax - Casual - Reliable - Responsible - Dutiful - Devoted), Precision (Careless - Loose - Accurate - Precise - Meticulous - Flawless), Logicality (Intuitive - Instinctive - Rational - Logical - Systematic - Analytical)
# 5. Agreeableness: Empathy (Aloof - Objective - Considerate - Empathetic - Compassionate), Politeness (Blunt - Direct - Civil - Polite - Courteous - Diplomatic), Trustingness (Suspicious - Wary - Neutral - Trusting - Naive - Gullible), Competitiveness (Cooperative - Agreeable - Ambitious - Competitive - Driven - Ruthless)
# 6. Honesty-Humility: Integrity (Deceptive - Flexible - Honest - Principled - Upright - Virtuous), Sincerity (Cunning - Guarded - Candid - Open - Sincere - Transparent), Humility (Arrogant - Proud - Confident - Modest - Humble - Unassuming)
# 7. Ungrouped: Impulsivity (Cautious - Deliberate - Spontaneous - Impulsive - Rash - Reckless), Skepticism (Gullible - Trusting - Questioning - Skeptical - Cynical - Disbelieving), Reflectiveness (Reactive - Quick - Thoughtful - Reflective - Contemplative - Philosophical)
# --- End Personality Trait List ---


class PersonaParserError(HierarchicalPlannerError):
    """Custom exception for PersonaParser errors."""
    pass


class PersonaParser:
    """
    Parses persona card text into structured hierarchical data using an LLM.
    """

    # Default model if not specified in config
    DEFAULT_PARSING_MODEL = "gemini-1.5-pro-latest" # Default to a Gemini model

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PersonaParser.

        Args:
            config (Dict[str, Any]): The main application configuration dictionary,
                                     which includes API settings needed by the LLM clients.
        """
        if not gemini_client and not anthropic_client:
            raise PersonaParserError("LLM client modules could not be imported.")
        if not config:
             raise PersonaParserError("Configuration dictionary is required.")

        self.config = config
        # Configure the LLM clients using the provided config
        try:
            # Configure Gemini client
            if gemini_client:
                gemini_client.configure_client(config.get('api', {}).get('resolved_key'))
                # Optionally configure DeepSeek fallback if needed for parsing
                if 'deepseek' in config:
                    gemini_client.configure_deepseek_fallback(config)
            
            # Configure Anthropic client
            if anthropic_client and 'anthropic' in config:
                anthropic_client.configure_client(config.get('anthropic', {}).get('api_key'))
        except Exception as e:
             # Log the error but allow initialization to continue; model fetching will fail later if needed
             logger.error(f"Error configuring LLM clients during PersonaParser init: {e}")
             # Depending on requirements, could raise PersonaParserError here

        # Determine the model to use for parsing.
        # Prioritize 'persona_parsing_model', fallback to main 'model_name', then internal default.
        default_parser_model = config.get('api', {}).get('model_name', self.DEFAULT_PARSING_MODEL)
        self.parsing_model_name = config.get('api', {}).get('persona_parsing_model', default_parser_model)
        logger.info(f"PersonaParser initialized, will use model: {self.parsing_model_name}")


    async def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse persona card text into a structured JSON format using an LLM.

        Args:
            text (str): The persona card text to parse.

        Returns:
            Dict[str, Any]: A structured JSON representation of the persona card.

        Raises:
            PersonaParserError: If the LLM call fails or the response is invalid.
        """
        if not text:
            raise ValueError("Input text cannot be empty.")

        prompt = PERSONA_PARSING_PROMPT.format(persona_card_text=text)

        logger.info(f"Sending persona card text to LLM for parsing...")

        try:
            # Prepare config for parsing
            parse_config = self.config.copy()
            if 'api' in parse_config:
                parse_config['api'] = parse_config['api'].copy() # Avoid modifying original config
                parse_config['api']['model_name'] = self.parsing_model_name
                # Adjust other generation params if needed for parsing
                parse_config['api']['temperature'] = 0.2 # Lower temp for JSON
            
            # Select the appropriate LLM client
            _, _, call_with_retry = await select_llm_client(parse_config)

            # Use the selected LLM client to parse the persona card
            parsed_json = await call_with_retry(
                prompt_template=PERSONA_PARSING_PROMPT, # Use the template directly
                context={"persona_card_text": text},    # Provide context
                config=parse_config,                    # Pass the (potentially modified) config
                is_structured=True                      # Expecting JSON
            )
            logger.debug("LLM response received and parsed successfully.")

        except (ApiCallError, ApiResponseError, JsonParsingError) as e:
            logger.error(f"LLM call/parsing failed during persona parsing: {e}")
            raise PersonaParserError(f"LLM call/parsing failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
            raise PersonaParserError(f"Unexpected LLM error: {e}") from e

        # Basic validation
        if not isinstance(parsed_json, dict) or "sections" not in parsed_json or not isinstance(parsed_json["sections"], dict):
            logger.warning(f"LLM response JSON missing 'sections' dictionary or invalid structure: {parsed_json}")
            raise PersonaParserError("Parsed JSON structure is invalid - missing 'sections' dictionary.")

        return parsed_json

# Example usage (for testing within this file)
if __name__ == "__main__":
    import asyncio # Need asyncio for the new parser

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing PersonaParser with LLM...")

    # Requires API keys set in environment or a valid config.yaml/llm_config.json
    # Load a sample persona card (e.g., from task.txt)
    sample_card_path = project_root / "hierarchical_planner" / "task.txt" # Adjusted path
    if not sample_card_path.exists():
         print(f"Sample persona card not found at {sample_card_path}. Cannot run test.")
         sys.exit(1)

    with open(sample_card_path, 'r', encoding='utf-8') as f:
        sample_text = f.read()

    async def run_parser_test():
        try:
            # Load main config using the function from config_loader
            from hierarchical_planner.config_loader import load_config
            # Assumes config.yaml is in hierarchical_planner/config relative to project root
            main_config = load_config('config/config.yaml')

            # Initialize parser with the loaded config
            parser = PersonaParser(config=main_config)
            parsed_structure = await parser.parse(sample_text) # Await the async parse method

            print("\n--- Parsed Persona Structure (from LLM) ---")
            print(json.dumps(parsed_structure, indent=2))

        except PersonaParserError as e:
            print(f"\nError during parsing: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    # Run the async test function
    asyncio.run(run_parser_test())
