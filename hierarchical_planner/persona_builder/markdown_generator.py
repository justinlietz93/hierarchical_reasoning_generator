"""
Generates Markdown representation from structured persona data.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MarkdownGenerator:
    """Generates Markdown from a dictionary representing parsed persona data."""

    def generate(self, data: dict, level: int = 1) -> str:
        """
        Recursively generates Markdown from the parsed data dictionary.

        Args:
            data (dict): The dictionary containing parsed persona data.
            level (int): The current heading level for Markdown sections.

        Returns:
            str: A string containing the generated Markdown.
        """
        md_lines = []
        indent = "  " * (level - 1) # For list indentation

        # Title (only at top level)
        if level == 1 and data.get('title'):
            md_lines.append(f"# {data['title']}\n")

        # Top-level optional fields (only at top level)
        if level == 1:
            top_level_keys = ['persona_name', 'instructions', 'personality_profile', 'response_output_requirements', 'tools_available']
            for key in top_level_keys:
                if data.get(key):
                    header_text = key.replace('_', ' ').title()
                    md_lines.append(f"## {header_text}")
                    value = data[key]
                    if key == 'personality_profile' and isinstance(value, dict):
                        for trait, trait_value in value.items():
                            md_lines.append(f"- **{trait}:** {trait_value}")
                    elif isinstance(value, list):
                         for item in value:
                             md_lines.append(f"- {item}")
                    else:
                         md_lines.append(str(value)) # Ensure string conversion
                    md_lines.append("") # Add spacing

        # Sections (can appear at level 1 or nested under subsections)
        if 'sections' in data:
            # Determine starting level for sections based on whether top-level fields were present
            section_level = level
            if level == 1: # Adjust starting level if top-level fields exist
                 section_level = 2 if any(data.get(k) for k in ['persona_name', 'instructions', 'personality_profile', 'response_output_requirements', 'tools_available']) else 1
                 md_lines.append(f"{'#' * section_level} Sections")
                 md_lines.append("")
            else:
                 # If called recursively for subsections' sections (unlikely with current parser structure, but for robustness)
                 section_level = level + 1


            for title, section_data in data['sections'].items():
                md_lines.append(f"{indent}{'#' * (section_level + 1)} {title}") # Sections start one level deeper
                if section_data.get('content'):
                    content_lines = str(section_data['content']).split('\n')
                    for line in content_lines:
                         md_lines.append(f"{indent}  {line}") # Indent section content
                if section_data.get('subsections'):
                    # Recursive call for subsections
                    md_lines.append(self.generate(section_data['subsections'], section_level + 2)) # Subsections are deeper
                elif section_data.get('items'): # Handle items directly under section if no subsections
                     for item in section_data['items']:
                         md_lines.append(f"{indent}  - {item}") # Indent items under section
                md_lines.append("") # Add spacing

        # Subsections (when called recursively)
        # This handles the case where the input 'data' is the subsections dict directly
        elif level > 1 and isinstance(data, dict) and 'sections' not in data:
             for title, sub_data in data.items():
                 md_lines.append(f"{indent}{'#' * level} {title}") # Subsection title
                 if sub_data.get('content'):
                     content_lines = str(sub_data['content']).split('\n')
                     for line in content_lines:
                          md_lines.append(f"{indent}  {line}") # Indent further
                 if sub_data.get('items'):
                     for item in sub_data['items']:
                         md_lines.append(f"{indent}  - {item}") # Indent further
                 md_lines.append("") # Add spacing

        return "\n".join(md_lines).strip()
