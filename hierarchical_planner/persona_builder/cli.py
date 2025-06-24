"""
Command-line interface for the Persona Builder module.

Handles multiple personas separated by a delimiter in the input file,
parsing each using an LLM, and generating structured outputs (YAML, JSON, XML, MD)
with filenames based on the identified persona name.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import re
import asyncio
import json
import yaml
from typing import Optional, Dict, Any, List # Import necessary types

# Ensure the package can be imported when run as a script
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use absolute imports assuming the package is installed/accessible
try:
    from hierarchical_planner.persona_builder.parser import PersonaParser
    from hierarchical_planner.persona_builder.xml_generator import XmlGenerator
    from hierarchical_planner.persona_builder.chunker import PersonaChunker
    from hierarchical_planner.persona_builder.markdown_generator import MarkdownGenerator # Import new generator
    from hierarchical_planner.persona_builder.output_saver import OutputSaver # Import saver
    from hierarchical_planner.config_loader import load_config
    # Import the LLM selector
    from hierarchical_planner.persona_builder.llm_selector import select_llm_client
except ImportError as e:
    print(f"Import Error in cli.py: {e}. Ensure hierarchical_planner package and PyYAML are installed or in PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Delimiter for separating multiple personas in the input file
PERSONA_DELIMITER = "~[PERSONA]"

# generate_safe_filename moved to OutputSaver

async def process_persona_text(
    persona_text: str,
    parser: PersonaParser,
    xml_generator: XmlGenerator,
    md_generator: MarkdownGenerator # Add markdown generator instance
) -> Optional[Dict[str, Any]]:
    """
    Parses a single persona text block and generates output formats.

    Args:
        persona_text (str): The text content for a single persona.
        parser (PersonaParser): Initialized PersonaParser instance.
        xml_generator (XmlGenerator): Initialized XmlGenerator instance.

    Returns:
        Optional[dict]: A dictionary containing parsed data and generated formats,
                       or None if processing fails for this block.
    """
    if not persona_text.strip():
        logger.debug("Skipping empty persona block.")
        return None

    logger.info("Processing persona block...")
    parsed_data = None
    try:
        parsed_data = await parser.parse(persona_text)
        # Extract persona name for filename
        persona_name = parsed_data.get('persona_name')
        if not persona_name or not isinstance(persona_name, str):
             title = parsed_data.get('title', 'UnknownPersona')
             persona_name = title.split('-')[0].strip()
             if not persona_name: persona_name = "UnknownPersona"
             logger.warning(f"LLM did not provide 'persona_name', using fallback: '{persona_name}'")
        else:
             logger.info(f"Persona name identified by LLM: '{persona_name}'")

        logger.info(f"Persona block parsed successfully for '{persona_name}'.")
    except Exception as e:
        logger.error(f"Error parsing persona block: {e}", exc_info=True)
        return None

    # Generate other formats
    xml_content = None
    try:
        xml_content = xml_generator.generate(parsed_data)
        logger.info(f"XML generated for '{persona_name}'.")
    except Exception as e:
        logger.error(f"Error generating XML for '{persona_name}': {e}")

    markdown_content = None
    try:
        markdown_content = md_generator.generate(parsed_data) # Use the generator instance
        logger.info(f"Markdown generated for '{persona_name}'.")
    except Exception as e:
        logger.error(f"Error generating Markdown for '{persona_name}': {e}")

    # Generate YAML string from parsed data
    yaml_content = None
    try:
        yaml_content = yaml.dump(parsed_data, sort_keys=False, allow_unicode=True, indent=2, default_flow_style=False)
        logger.info(f"YAML generated for '{persona_name}'.")
    except Exception as e:
        logger.error(f"Error generating YAML string for '{persona_name}': {e}")

    return {
        "persona_name": persona_name,
        "json_data": parsed_data,
        "xml": xml_content,
        "yaml": yaml_content,
        "markdown": markdown_content
    }

# save_output_files moved to OutputSaver

async def main_async(args):
    """Asynchronous main logic."""
    # Initialize components first
    parser = None
    xml_generator = None
    md_generator = None
    output_saver = None
    chunker = PersonaChunker() # Use default delimiter
    try:
        main_config = load_config()
        logger.info("Main configuration loaded successfully for CLI.")
        parser = PersonaParser(config=main_config)
        xml_generator = XmlGenerator() # Uses default schema
        md_generator = MarkdownGenerator()
        output_saver = OutputSaver()
    except Exception as e:
         logger.error(f"Failed to load main config or initialize components: {e}", exc_info=True)
         return

    # Chunk the input file
    try:
        persona_blocks = chunker.chunk_file(args.input_file)
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed to read or chunk input file: {e}")
        return

    if not persona_blocks:
         # chunk_file logs a warning if delimiter not found but file has content
         logger.error("No processable persona content found in the input file.")
         return

    # Create tasks for processing each block concurrently
    tasks = [process_persona_text(block, parser, xml_generator, md_generator) for block in persona_blocks]

    logger.info(f"Processing {len(tasks)} persona blocks concurrently...")
    all_results = await asyncio.gather(*tasks)
    logger.info("Finished processing all persona blocks.")

    # Process results
    successful_results = [res for res in all_results if res]
    if not successful_results:
         logger.error("Failed to process any persona blocks successfully.")
         return

    if args.output_dir:
        output_path = Path(args.output_dir)
        logger.info(f"Saving outputs to directory: {output_path}")
        for result_data in successful_results:
            output_saver.save_all_formats(result_data, output_path) # Use saver instance
    else:
        # If not saving to files, print the YAML output for the *first* persona
        first_result = successful_results[0]
        print(f"\n--- Generated YAML for Persona: {first_result.get('persona_name', 'Unknown')} ---")
        yaml_output = first_result.get("yaml")
        if yaml_output is None: # Regenerate if needed
             yaml_output = yaml.dump(first_result["json_data"], sort_keys=False, allow_unicode=True, indent=2, default_flow_style=False)
        print(yaml_output)

# --- Helper function for Markdown Generation (Moved to markdown_generator.py) ---

def main():
    """Main CLI entry point."""

    # Define the help epilog explaining the output structure
    output_formats_help = f"""
Output Formats:
---------------
The script uses an LLM to parse the input text file, splitting it by the
delimiter '{PERSONA_DELIMITER}'. For each persona found, it generates multiple
output files in the specified output directory (or prints YAML for the first
persona to console if -o is omitted):

- [PersonaName]_persona.yaml: Structured data in YAML format.
- [PersonaName]_persona.json: Structured data in JSON format.
- [PersonaName]_persona.xml: Structured data in XML format.
- [PersonaName]_persona.md: Structured data formatted as Markdown.

The structure attempts to capture hierarchy identified by the LLM, including:
persona_name, title, instructions, personality_profile, response_output_requirements,
tools_available, and nested sections/subsections with content and/or bullet items.
Filenames are generated based on the 'persona_name' identified by the LLM.
"""

    parser = argparse.ArgumentParser(
        description=f"Convert one or more persona cards (separated by '{PERSONA_DELIMITER}') in a text file into structured formats (YAML, JSON, XML, MD) using an LLM.",
        epilog=output_formats_help,
        formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting of epilog
    )
    parser.add_argument(
        "input_file",
        help="Path to the text file containing one or more persona cards (e.g., task.txt)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save the generated output files. If not provided, prints YAML for the *first* persona to console."
    )
    # Removed prefix/suffix arguments

    args = parser.parse_args()

    # Run the async main logic
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
