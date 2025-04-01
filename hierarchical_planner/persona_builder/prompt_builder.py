"""
Prompt Builder for formatting XML persona representations into usable system prompts.

This module handles the transformation of generated XML structures into system
prompts formatted for use with different LLM implementations.
"""

import re
import enum
from typing import Dict, Optional, Union


class PersonaPromptFormat(enum.Enum):
    """
    Formats available for generating persona system prompts.
    """
    XML = "xml"       # Full XML structure with formatting
    COMPACT = "compact"  # XML with minimal whitespace
    PLAIN_TEXT = "plain_text"  # Human-readable text without XML tags


class PromptBuilder:
    """
    Builds formatted system prompts from XML persona representations.
    
    This class takes the XML output from the XmlGenerator and formats it
    into system prompts suitable for different LLM systems, with various
    formatting options.
    """
    
    # Default wrapper text to include around the persona
    DEFAULT_WRAPPER = {
        "prefix": (
            "You are to adopt the persona defined in the XML structure below.\n"
            "This structure hierarchically defines your identity, behaviors, and constraints.\n"
            "Follow these guidelines strictly when responding to the user.\n\n"
        ),
        "suffix": (
            "\n\nRemember to stay true to this persona's characteristics in all interactions."
        )
    }
    
    def __init__(self):
        """Initialize the PromptBuilder."""
        pass
    
    def build(
        self, 
        xml_content: str, 
        format: Union[str, PersonaPromptFormat] = PersonaPromptFormat.XML,
        wrapper_text: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Build a system prompt from XML content.
        
        Args:
            xml_content (str): The XML representation of the persona.
            format (str or PersonaPromptFormat): The desired output format.
            wrapper_text (Dict[str, str], optional): Custom wrapper text for the prompt.
                Format: {'prefix': '...', 'suffix': '...'}
                
        Returns:
            str: The formatted system prompt.
        """
        # Convert string format to enum if needed
        if isinstance(format, str):
            try:
                format = PersonaPromptFormat(format)
            except ValueError:
                format = PersonaPromptFormat.XML
        
        # Apply the appropriate formatter based on format
        if format == PersonaPromptFormat.XML:
            formatted_content = self._format_xml(xml_content)
        elif format == PersonaPromptFormat.COMPACT:
            formatted_content = self._format_compact(xml_content)
        elif format == PersonaPromptFormat.PLAIN_TEXT:
            formatted_content = self._format_plain_text(xml_content)
        else:
            formatted_content = xml_content  # Fallback to unmodified content
        
        # Apply wrapper text
        wrapper = wrapper_text or self.DEFAULT_WRAPPER
        prefix = wrapper.get('prefix', self.DEFAULT_WRAPPER['prefix'])
        suffix = wrapper.get('suffix', self.DEFAULT_WRAPPER['suffix'])
        
        # Assemble final prompt
        prompt = f"{prefix}{formatted_content}{suffix}"
        
        return prompt
    
    def _format_xml(self, xml_content: str) -> str:
        """Format XML content with proper indentation and structure."""
        # This format preserves the original XML formatting and structure
        return xml_content
    
    def _format_compact(self, xml_content: str) -> str:
        """Format XML content in a compact form with minimal whitespace."""
        # Remove pretty-printing whitespace but maintain basic structure
        compact = re.sub(r'\n\s+', ' ', xml_content)
        compact = re.sub(r'>\s+<', '><', compact)
        compact = re.sub(r'\s{2,}', ' ', compact)
        return compact
    
    def _format_plain_text(self, xml_content: str) -> str:
        """
        Format XML content as plain text without XML tags.
        
        This attempts to create a human-readable representation of the
        XML structure without the tags.
        """
        import xml.etree.ElementTree as ET
        try:
            # Parse the XML
            root = ET.fromstring(xml_content)
            
            # Build text recursively
            lines = []
            self._process_element_as_text(root, lines, 0)
            
            return "\n".join(lines)
        except Exception as e:
            # If XML parsing fails, do a simple tag removal
            # This is a fallback and won't preserve structure well
            plain = re.sub(r'<[^>]+>', '', xml_content)
            plain = re.sub(r'\n\s*\n', '\n', plain)
            return plain.strip()
    
    def _process_element_as_text(self, element, lines, depth):
        """
        Recursively process an XML element into plain text.
        
        Args:
            element: The XML element to process.
            lines: List to accumulate text lines.
            depth: Current indentation depth.
        """
        # Skip the XML declaration
        if element.tag == '?xml':
            return
        
        # Get indentation string
        indent = "  " * depth
        
        # For the root element, just process children
        if depth == 0:
            # Handle title specially
            title_elem = element.find("title")
            if title_elem is not None and title_elem.text:
                lines.append(f"{title_elem.text}")
                lines.append("")  # Empty line after title
            
            # Process all sections
            sections_elem = element.find("sections")
            if sections_elem is not None:
                for section in sections_elem:
                    self._process_element_as_text(section, lines, depth + 1)
            
            return
        
        # Get the element title from attribute
        title = element.get("title", element.tag.title().replace("_", " "))
        
        # Add section heading
        if depth == 1:
            lines.append(f"{indent}{title}:")
        else:
            lines.append(f"{indent}{title}:")
        
        # Add content if present
        content_elem = element.find("content")
        if content_elem is not None and content_elem.text:
            # Indent and add the content text, preserving line breaks
            content_lines = content_elem.text.strip().split('\n')
            for line in content_lines:
                lines.append(f"{indent}  {line}")
        
        # Add items if present
        items_elem = element.find("items")
        if items_elem is not None:
            for item in items_elem.findall("item"):
                if item.text:
                    # Format each item as a bullet point
                    item_lines = item.text.strip().split('\n')
                    lines.append(f"{indent}  • {item_lines[0]}")
                    for line in item_lines[1:]:
                        lines.append(f"{indent}    {line}")
        
        # Process subsections if present
        subsections_elem = element.find("subsections")
        if subsections_elem is not None:
            for subsection in subsections_elem:
                self._process_element_as_text(subsection, lines, depth + 1)
        
        # Add empty line after sections at the top level
        if depth == 1:
            lines.append("")  # Empty line between main sections
