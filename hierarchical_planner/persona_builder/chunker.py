"""
Module responsible for splitting input text containing multiple personas.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Delimiter for separating multiple personas in the input file
PERSONA_DELIMITER = "~[PERSONA]"

class PersonaChunker:
    """
    Splits a text file containing multiple persona descriptions into individual chunks.
    """
    def __init__(self, delimiter: str = PERSONA_DELIMITER):
        """
        Initializes the chunker with a specific delimiter.

        Args:
            delimiter (str): The string used to separate persona descriptions.
        """
        self.delimiter = delimiter
        logger.debug(f"PersonaChunker initialized with delimiter: '{self.delimiter}'")

    def chunk_file(self, file_path: str) -> List[str]:
        """
        Reads a file and splits its content into persona chunks.

        Args:
            file_path (str): The path to the input text file.

        Returns:
            List[str]: A list of non-empty persona text blocks.

        Raises:
            FileNotFoundError: If the input file does not exist.
            IOError: If there's an error reading the file.
        """
        input_path = Path(file_path)
        if not input_path.exists():
            logger.error(f"Input file not found for chunking: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Reading and chunking file: {file_path}")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            logger.error(f"Error reading input file for chunking: {e}")
            raise IOError(f"Error reading input file: {e}") from e

        # Split into persona blocks
        persona_blocks = full_text.split(self.delimiter)

        # Filter out potentially empty strings resulting from delimiters at start/end or consecutive delimiters
        non_empty_blocks = [block.strip() for block in persona_blocks if block.strip()]

        logger.info(f"Found {len(non_empty_blocks)} non-empty persona block(s) using delimiter '{self.delimiter}'.")
        if not non_empty_blocks and full_text.strip():
             logger.warning("Delimiter not found, treating entire file as a single persona block.")
             return [full_text.strip()] # Treat whole file as one block if delimiter not found but file has content

        return non_empty_blocks
