import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages loading and accessing LLM prompts from a specified directory.
    """
    def __init__(self, prompts_dir: str = 'prompts'):
        """
        Initializes the PromptManager by loading all prompts from the directory.

        Args:
            prompts_dir: The directory where prompt files are stored, relative
                         to this file's location.
        """
        self.prompts = {}
        self._load_prompts(prompts_dir)

    def _load_prompts(self, prompts_dir: str):
        """Loads all .txt files from the specified directory."""
        script_dir = Path(__file__).parent
        abs_prompts_dir = script_dir / prompts_dir
        
        if not abs_prompts_dir.is_dir():
            logger.error(f"Prompts directory not found: {abs_prompts_dir}")
            return

        logger.info(f"Loading prompts from: {abs_prompts_dir}")
        for filename in os.listdir(abs_prompts_dir):
            if filename.endswith(".txt") and not filename.startswith("__"):
                prompt_name = filename[:-4]  # Remove .txt extension
                try:
                    with open(abs_prompts_dir / filename, 'r', encoding='utf-8') as f:
                        self.prompts[prompt_name] = f.read()
                    logger.debug(f"Loaded prompt: {prompt_name}")
                except IOError as e:
                    logger.error(f"Failed to read prompt file {filename}: {e}")

    def get_prompt(self, name: str) -> str:
        """
        Retrieves a loaded prompt by its name.

        Args:
            name: The name of the prompt (filename without extension).

        Returns:
            The prompt text as a string, or an error message if not found.
        """
        return self.prompts.get(name, f"Error: Prompt '{name}' not found.")
