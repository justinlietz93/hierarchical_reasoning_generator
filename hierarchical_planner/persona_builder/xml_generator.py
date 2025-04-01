"""
XML generator for transforming parsed persona data into structured XML.

This module converts parsed persona data structures into hierarchical XML
representations based on a schema definition.
"""

import re
import logging
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union

from .schemas import PersonaSchema, PersonaSection, PersonaSectionType

# Configure logging
logger = logging.getLogger(__name__)


class XmlGenerator:
    """
    Generates XML from structured persona data.
    
    This class takes parsed persona data and converts it into a hierarchical XML
    structure according to a provided schema.
    """
    
    def __init__(self, schema: Optional[PersonaSchema] = None):
        """
        Initialize the XML generator.
        
        Args:
            schema (PersonaSchema, optional): The schema to use for structuring the XML.
                If not provided, the default schema will be used.
        """
        self.schema = schema or PersonaSchema.default_schema()
    
    def generate(self, parsed_data: Dict[str, Any]) -> str:
        """
        Generate XML from parsed persona data.
        
        Args:
            parsed_data (Dict[str, Any]): The parsed persona data structure.
            
        Returns:
            str: A formatted XML string representation of the persona.
        """
        # Create root element
        root = ET.Element(self.schema.root_element)
        
        # Add title if present
        if 'title' in parsed_data:
            title_elem = ET.SubElement(root, "title")
            title_elem.text = parsed_data['title']

        # Add top-level optional fields if present
        for key in ['persona_name', 'instructions', 'response_output_requirements', 'tools_available']:
             if key in parsed_data and parsed_data[key]:
                 elem = ET.SubElement(root, key)
                 # Handle list for tools_available if needed, otherwise just text
                 if key == 'tools_available' and isinstance(parsed_data[key], list):
                      for item in parsed_data[key]:
                           item_elem = ET.SubElement(elem, "tool")
                           item_elem.text = item
                 else:
                      elem.text = str(parsed_data[key]) # Ensure string

        # Add personality profile if present
        if 'personality_profile' in parsed_data and isinstance(parsed_data['personality_profile'], dict):
            profile_elem = ET.SubElement(root, "personality_profile")
            for trait, value in parsed_data['personality_profile'].items():
                 trait_tag = self._normalize_tag_name(trait) # Use normalized tag
                 trait_elem = ET.SubElement(profile_elem, trait_tag)
                 trait_elem.set("trait_name", trait) # Keep original name as attribute
                 trait_elem.text = value

        # Process sections
        if 'sections' in parsed_data:
            self._add_sections(root, parsed_data['sections'])

        # Generate pretty XML string
        xml_str = ET.tostring(root, encoding='unicode')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove empty lines that can appear in the pretty-printed output
        # This regex finds all empty lines that may have whitespace and removes them
        pretty_xml = re.sub(r'\n\s*\n', '\n', pretty_xml)
        
        return pretty_xml
    
    def _add_sections(self, parent_elem: ET.Element, sections: Dict[str, Any]) -> None:
        """
        Add sections to the parent element.
        
        Args:
            parent_elem (ET.Element): The parent XML element.
            sections (Dict[str, Any]): The sections data to add.
        """
        # Create a container for all sections
        sections_elem = ET.SubElement(parent_elem, "sections")
        
        # Loop through each section
        for section_name, section_data in sections.items():
            # Create section element with normalized tag name
            section_tag = self._normalize_tag_name(section_name)
            section_elem = ET.SubElement(sections_elem, section_tag)
            
            # Add title attribute with original section name
            section_elem.set("title", section_name)
            
            # Determine section type and add as attribute
            section_type = self._get_section_type(section_name)
            section_elem.set("type", section_type.value)
            
            # Add main content if present
            if isinstance(section_data, dict) and 'content' in section_data and section_data['content']:
                content_elem = ET.SubElement(section_elem, "content")
                content_elem.text = section_data['content']
            elif isinstance(section_data, str) and section_data:
                content_elem = ET.SubElement(section_elem, "content")
                content_elem.text = section_data
            
            # Add bullet points if present
            if isinstance(section_data, dict) and 'items' in section_data and section_data['items']:
                items_elem = ET.SubElement(section_elem, "items")
                for item in section_data['items']:
                    item_elem = ET.SubElement(items_elem, "item")
                    item_elem.text = item
            
            # Add subsections if present
            if isinstance(section_data, dict) and 'subsections' in section_data and section_data['subsections']:
                self._add_subsections(section_elem, section_data['subsections'])
    
    def _add_subsections(self, parent_elem: ET.Element, subsections: Dict[str, Any]) -> None:
        """
        Add subsections to the parent element.
        
        Args:
            parent_elem (ET.Element): The parent XML element.
            subsections (Dict[str, Any]): The subsections data to add.
        """
        # Create a container for all subsections
        subsections_elem = ET.SubElement(parent_elem, "subsections")
        
        # Loop through each subsection
        for subsection_name, subsection_data in subsections.items():
            # Create subsection element with normalized tag name
            subsection_tag = self._normalize_tag_name(subsection_name)
            subsection_elem = ET.SubElement(subsections_elem, subsection_tag)
            
            # Add title attribute with original subsection name
            subsection_elem.set("title", subsection_name)
            
            # Add main content if present
            if isinstance(subsection_data, dict) and 'content' in subsection_data and subsection_data['content']:
                content_elem = ET.SubElement(subsection_elem, "content")
                content_elem.text = subsection_data['content']
            elif isinstance(subsection_data, str) and subsection_data:
                content_elem = ET.SubElement(subsection_elem, "content")
                content_elem.text = subsection_data
            
            # Add bullet points if present
            if isinstance(subsection_data, dict) and 'items' in subsection_data and subsection_data['items']:
                items_elem = ET.SubElement(subsection_elem, "items")
                for item in subsection_data['items']:
                    item_elem = ET.SubElement(items_elem, "item")
                    item_elem.text = item
    
    def _normalize_tag_name(self, name: str) -> str:
        """
        Normalize a string to be used as an XML tag name.
        
        Args:
            name (str): The string to normalize.
            
        Returns:
            str: A valid XML tag name.
        """
        # Convert to lowercase
        tag = name.lower()
        
        # Remove non-alphanumeric characters (except underscore)
        tag = re.sub(r'[^a-z0-9_]', '_', tag)
        
        # Ensure starts with letter or underscore (XML requirement)
        if tag and not (tag[0].isalpha() or tag[0] == '_'):
            tag = 'section_' + tag
        
        # Handle empty string
        if not tag:
            tag = 'section'
        
        return tag
    
    def _get_section_type(self, section_name: str) -> PersonaSectionType:
        """
        Determine the type of a section based on its name, using schema mappings if available.
        
        Args:
            section_name (str): The name of the section.
            
        Returns:
            PersonaSectionType: The type of the section.
        """
        # Try to find a matching section in the schema first
        section_name_lower = section_name.lower()
        for section in self.schema.sections:
            if section.title.lower() == section_name_lower:
                return section.type
            
            # Check if section_name is similar to any schema section
            similarity_threshold = 0.7  # Adjust as needed
            schema_words = set(section.title.lower().split())
            name_words = set(section_name_lower.split())
            common_words = schema_words.intersection(name_words)
            
            if common_words and len(common_words) / max(len(schema_words), len(name_words)) >= similarity_threshold:
                return section.type

        # If no match in schema, default to OTHER
        logger.debug(f"Section '{section_name}' not found in schema, defaulting type to OTHER.")
        return PersonaSectionType.OTHER

    def apply_schema_structure(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reorganize parsed data to match the schema structure.
        
        This method attempts to map the parsed sections to the schema's structure,
        combining or splitting sections as needed.
        
        Args:
            parsed_data (Dict[str, Any]): The original parsed data.
            
        Returns:
            Dict[str, Any]: The restructured data that matches the schema.
        """
        # Currently a placeholder - this would implement more sophisticated
        # matching between parsed sections and schema sections
        # This is a more advanced feature that could be implemented in the future
        
        return parsed_data  # Return unchanged for now
