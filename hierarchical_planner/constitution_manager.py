import json
import logging
from typing import Dict, Any

from .llm_client_selector import select_llm_client

logger = logging.getLogger(__name__)

CONSTITUTION_UPDATE_PROMPT = """
You are a meticulous project architect. Your task is to analyze a reasoning tree (a hierarchical plan) and identify any new requirements, dependencies, or architectural elements that are not already captured in the project constitution.

Project Constitution:
{constitution}

Reasoning Tree:
{reasoning_tree}

Analyze the reasoning tree and identify any new:
-   **Global Dependencies**: New libraries, tools, or APIs mentioned.
-   **Non-Functional Requirements**: Implicit or explicit requirements like performance, security, or data retention.
-   **Key Data Structures**: New data structures or entities that should be formally defined.
-   **Project File Map**: New file paths or directory structures mentioned in the reasoning tree.

Return a JSON object with the keys "global_dependencies_and_interfaces", "non_functional_requirements", "key_data_structures", and "project_file_map". For each key, provide a list of new items to be added to the constitution. If no new items are found for a category, return an empty list or an empty object for the file map.

**project_file_map**:
-   The `project_file_map` should be a nested JSON object representing the file system.
-   Each key is a file or directory name.
-   Each value is an object with a "type" (file or directory) and a "description".
-   Directories can have a "children" object with the same structure.
-   Example: If a step mentions creating a "new_module/utils.ts" file, the output should be:
    "project_file_map": {{
        "new_module": {{
            "type": "directory",
            "description": "A new module for utility functions.",
            "children": {{
                "utils.ts": {{
                    "type": "file",
                    "description": "Utility functions for the new module."
                }}
            }}
        }}
    }}
"""

class ConstitutionManager:
    def __init__(self, constitution_path: str = "project_constitution.json"):
        self.constitution_path = constitution_path
        self.constitution = self.load()

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.constitution_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        with open(self.constitution_path, 'w', encoding='utf-8') as f:
            json.dump(self.constitution, f, indent=2, ensure_ascii=False)

    async def update_from_plan(self, reasoning_tree: Dict[str, Any], config: Dict[str, Any]):
        logger.info("Analyzing reasoning tree to update constitution...")
        _, _, call_with_retry = await select_llm_client(config, agent_name="constitution_manager")
        
        context = {
            "constitution": json.dumps(self.constitution, indent=2),
            "reasoning_tree": json.dumps(reasoning_tree, indent=2)
        }
        
        try:
            update_data = await call_with_retry(CONSTITUTION_UPDATE_PROMPT, context, config)
            
            if update_data:
                self._merge_updates(update_data)
                self.save()
                logger.info("Constitution updated and saved.")
        except Exception as e:
            logger.error(f"Failed to update constitution from plan: {e}", exc_info=True)

    def _merge_updates(self, update_data: Dict[str, Any]):
        for key, new_items in update_data.items():
            if key == "project_file_map":
                if key not in self.constitution:
                    self.constitution[key] = {}
                self._merge_file_map(self.constitution[key], new_items)
            elif key in self.constitution and isinstance(self.constitution[key], list):
                existing_items = {json.dumps(item, sort_keys=True) for item in self.constitution[key]}
                for item in new_items:
                    item_str = json.dumps(item, sort_keys=True)
                    if item_str not in existing_items:
                        self.constitution[key].append(item)
                        existing_items.add(item_str)

    def _merge_file_map(self, target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                if "children" in target[key] and "children" in value:
                    self._merge_file_map(target[key]["children"], value["children"])
                else:
                    target[key].update(value)
            else:
                target[key] = value
