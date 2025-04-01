"""
Handles saving generated persona data to various file formats.
"""

import logging
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OutputSaver:
    """Saves structured persona data to multiple file formats."""

    def generate_safe_filename(self, name: str, suffix: str) -> str:
        """Generates a safe filename from a persona name."""
        if not name or not isinstance(name, str):
            name = "UnknownPersona"
        # Remove invalid characters
        safe_name = re.sub(r'[<>:"/\\|?*\s]+', '_', name.strip())
        # Limit length
        safe_name = safe_name[:50]
        # Ensure it's not empty
        if not safe_name:
            safe_name = "persona"
        return f"{safe_name}{suffix}"

    def save_all_formats(self, results: Dict[str, Any], output_dir: str):
        """
        Saves the generated formats (JSON, YAML, XML, Markdown) to files
        based on the persona name included in the results.

        Args:
            results (dict): A dictionary containing the parsed data and generated content strings.
                            Expected keys: 'persona_name', 'json_data', 'xml', 'yaml', 'markdown'.
            output_dir (str): The directory path to save the files.
        """
        if not results:
            logger.warning("No results provided to save.")
            return

        persona_name = results.get("persona_name", "UnknownPersona")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_filename = self.generate_safe_filename(persona_name, "")

        logger.info(f"Saving output files for persona '{persona_name}' to {output_path}")

        try:
            # Save JSON
            if "json_data" in results:
                json_file = output_path / f"{base_filename}_persona.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results["json_data"], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved JSON to: {json_file}")
            else:
                 logger.warning("JSON data missing, cannot save JSON file.")

            # Save YAML
            yaml_content_to_save = results.get("yaml")
            if yaml_content_to_save is None and "json_data" in results: # Regenerate if needed
                 yaml_content_to_save = yaml.dump(results["json_data"], sort_keys=False, allow_unicode=True, indent=2, default_flow_style=False)

            if yaml_content_to_save:
                yaml_file = output_path / f"{base_filename}_persona.yaml"
                with open(yaml_file, 'w', encoding='utf-8') as f:
                    f.write(yaml_content_to_save)
                logger.info(f"Saved YAML to: {yaml_file}")
            else:
                 logger.warning("YAML content missing or could not be generated, cannot save YAML file.")


            # Save XML
            if results.get("xml"):
                xml_file = output_path / f"{base_filename}_persona.xml"
                with open(xml_file, 'w', encoding='utf-8') as f:
                    f.write(results["xml"])
                logger.info(f"Saved XML to: {xml_file}")
            else:
                 logger.warning("XML content missing, cannot save XML file.")

            # Save Markdown
            if results.get("markdown"):
                md_file = output_path / f"{base_filename}_persona.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(results["markdown"])
                logger.info(f"Saved Markdown to: {md_file}")
            else:
                 logger.warning("Markdown content missing, cannot save Markdown file.")

        except Exception as e:
            logger.error(f"Error saving output files for '{persona_name}': {e}", exc_info=True)
