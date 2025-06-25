import unittest
import os
import json
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

# Ensure the test can find the necessary modules
import sys
# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from hierarchical_planner.project_builder import ProjectBuilder
from hierarchical_planner.exceptions import ProjectBuilderError

class TestProjectBuilder(unittest.TestCase):

    def setUp(self):
        """Set up a test environment before each test."""
        self.test_dir = Path("test_project_builder_temp")
        self.project_dir = self.test_dir / "generated_project"
        self.config_dir = self.test_dir / "config"
        
        # Create test directories
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

        # Create dummy reasoning tree
        self.reasoning_tree_path = self.test_dir / "reasoning_tree.json"
        self.dummy_tree = {
            "Phase 1: Setup": {
                "Task 1.1: Create initial files": [
                    {"step 1": "Create a README.md with 'Hello World'."},
                    {"step 2": "Create a directory named 'src'."}
                ]
            }
        }
        with open(self.reasoning_tree_path, 'w') as f:
            json.dump(self.dummy_tree, f)

        # Create dummy config file
        self.config_path = self.config_dir / "config.yaml"
        self.dummy_config = {
            "llm": {
                "executor_provider": "gemini",
                "validator_provider": "deepseek"
            },
            "project_builder": {
                "max_retries": 1
            }
        }
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(self.dummy_config, f)

    def tearDown(self):
        """Clean up the test environment after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('hierarchical_planner.project_builder.UniversalLLMClient')
    def test_initialization(self, mock_llm_client):
        """Test that the ProjectBuilder initializes correctly."""
        builder = ProjectBuilder(
            reasoning_tree_path=str(self.reasoning_tree_path),
            config_path=str(self.config_path),
            project_dir=str(self.project_dir)
        )
        self.assertEqual(builder.reasoning_tree, self.dummy_tree)
        self.assertTrue(self.project_dir.exists())
        self.assertEqual(mock_llm_client.call_count, 2) # Executor and Validator

    def test_initialization_no_tree_file(self):
        """Test initialization fails if the reasoning tree file does not exist."""
        with self.assertRaises(ProjectBuilderError):
            ProjectBuilder(
                reasoning_tree_path="non_existent_file.json",
                config_path=str(self.config_path),
                project_dir=str(self.project_dir)
            )

    @patch('hierarchical_planner.project_builder.UniversalLLMClient')
    def test_parse_llm_response_for_actions_file(self, mock_llm_client):
        """Test parsing of a file creation action from an LLM response."""
        builder = ProjectBuilder(
            reasoning_tree_path=str(self.reasoning_tree_path),
            config_path=str(self.config_path),
            project_dir=str(self.project_dir)
        )
        response_text = "=== File: src/main.py ===\nprint('Hello, World!')"
        actions = builder._parse_llm_response_for_actions(response_text)
        
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]['type'], 'file')
        self.assertEqual(actions[0]['path'], 'src/main.py')
        self.assertEqual(actions[0]['content'], "print('Hello, World!')")

    @patch('hierarchical_planner.project_builder.UniversalLLMClient')
    def test_parse_llm_response_for_actions_mkdir(self, mock_llm_client):
        """Test parsing of a directory creation action from an LLM response."""
        builder = ProjectBuilder(
            reasoning_tree_path=str(self.reasoning_tree_path),
            config_path=str(self.config_path),
            project_dir=str(self.project_dir)
        )
        response_text = "mkdir -p src/components"
        actions = builder._parse_llm_response_for_actions(response_text)
        
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]['type'], 'command')
        self.assertEqual(actions[0]['command'], 'mkdir -p src/components')

    @patch('hierarchical_planner.project_builder.UniversalLLMClient')
    def test_parse_llm_response_for_actions_analysis(self, mock_llm_client):
        """Test parsing of an analysis response from an LLM."""
        builder = ProjectBuilder(
            reasoning_tree_path=str(self.reasoning_tree_path),
            config_path=str(self.config_path),
            project_dir=str(self.project_dir)
        )
        response_text = "This step requires analyzing the data structure."
        actions = builder._parse_llm_response_for_actions(response_text)
        
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]['type'], 'analysis')
        self.assertEqual(actions[0]['summary'], response_text)

if __name__ == '__main__':
    unittest.main()
