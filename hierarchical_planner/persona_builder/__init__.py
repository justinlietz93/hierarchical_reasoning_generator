"""
Persona Builder module for converting persona card text to hierarchical XML prompts.

This module provides functionality to parse persona card descriptions and generate
structured XML system prompts for use with LLMs, enabling more consistent 
role-playing and persona adherence.
"""

from .parser import PersonaParser
from .xml_generator import XmlGenerator
# from .prompt_builder import PromptBuilder, PersonaPromptFormat # Not needed
from .schemas import PersonaSchema
from .chunker import PersonaChunker
from .markdown_generator import MarkdownGenerator # Import new generator
from .output_saver import OutputSaver # Import saver
# from hierarchical_planner.universal_LLM_client import UniversalLLMClient # Not needed

__all__ = ['PersonaBuilder', 'PersonaParser', 'XmlGenerator', 'PersonaChunker',
           'MarkdownGenerator', 'OutputSaver', 'PersonaSchema'] # Added new classes


class PersonaBuilder:
    """
    Main class for building persona prompts from persona card descriptions.
    
    This is the primary entry point for the persona_builder module, providing
    a simplified API for converting persona cards to structured data using an LLM.
    (Note: This class might become less relevant if the CLI handles orchestration directly).
    """

    def __init__(self, config: dict = None, schema: PersonaSchema = None):
        """
        Initialize a new PersonaBuilder.

        Args:
            config (dict, optional): Main application configuration. If None, attempts to load default.
            schema (PersonaSchema, optional): The schema to use for structuring the persona XML.
                If not provided, a default schema will be used.
        """
        if not config:
             # Attempt to load default config if none provided
             try:
                 from hierarchical_planner.config_loader import load_config
                 config = load_config()
             except Exception as e:
                 raise ValueError(f"Configuration is required and default load failed: {e}")

        # Initialize parser with config
        self.parser = PersonaParser(config=config)
        self.xml_generator = XmlGenerator(schema=schema)
        self.md_generator = MarkdownGenerator() # Instantiate new generator
        self.saver = OutputSaver() # Instantiate saver
        self._parsed_data = None # Stores the JSON structure from the LLM
        self._xml_content = None # Stores generated XML

    # Note: parse methods might need to be async now
    async def parse_from_file(self, file_path):
        """
        Parse a persona card from a text file. (Handles single persona).

        Args:
            file_path (str): Path to the file containing the persona card text.

        Returns:
            self: For method chaining.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Assuming parse_from_text is now async
        return await self.parse_from_text(content)

    async def parse_from_text(self, text):
        """
        Parse a persona card from a text string. (Handles single persona).

        Args:
            text (str): The persona card text content.

        Returns:
            self: For method chaining.
        """
        # Use await for the async parser method
        self._parsed_data = await self.parser.parse(text)
        return self

    def generate_xml(self): # This can remain synchronous
        """
        Generate XML representation of the parsed persona.
        
        Returns:
            str: XML string representation of the persona.
        
        Raises:
            ValueError: If no persona has been parsed yet.
        """
        if not self._parsed_data:
            raise ValueError("No persona data has been parsed. Call parse_from_text() or parse_from_file() first.")
        
        self._xml_content = self.xml_generator.generate(self._parsed_data)
        return self._xml_content

    # generate_system_prompt removed as CLI handles output formats directly

    def get_parsed_data(self): # This can remain synchronous
        """Get the current parsed persona data structure."""
        return self._parsed_data
    
    def get_xml_content(self):
        """Get the current XML representation."""
        return self._xml_content
