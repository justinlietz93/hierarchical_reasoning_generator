# hierarchical_planner/project_builder.py

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config_loader import ConfigLoader
from .gemini_client import GeminiClient
# Assuming DeepSeekV3Client exists or is handled by UniversalLLMClient
# from .deepseek_v3_client import DeepSeekV3Client
from .universal_LLM_client import UniversalLLMClient
from .exceptions import ProjectBuilderError, LLMClientError, ValidationError
from .logger_setup import LoggerSetup

# Configure logging
LoggerSetup.setup_logging()
logger = logging.getLogger(__name__)

class ProjectBuilder:
    """
    Builds a project by executing steps defined in a reasoning tree,
    using one LLM for execution and another for validation.
    """

    def __init__(self, reasoning_tree_path: str, config_path: str, project_dir: str):
        """
        Initializes the ProjectBuilder.

        Args:
            reasoning_tree_path: Path to the reasoning_tree.json file.
            config_path: Path to the configuration file (e.g., config.yaml).
            project_dir: Path to the target directory where the project will be built.
        """
        self.reasoning_tree_path = Path(reasoning_tree_path)
        self.config_path = Path(config_path)
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True) # Ensure project dir exists

        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.llm_config = self.config_loader.load_llm_config() # Assuming llm_config.json exists

        # TODO: Select clients based on config (Gemini for execution, DeepSeek for validation)
        # For now, using UniversalLLMClient placeholders
        self.executor_llm = UniversalLLMClient(
            provider=self.llm_config.get("executor_provider", "gemini"), # Default to gemini
            api_key=os.getenv(self.llm_config.get("executor_api_key_env", "GEMINI_API_KEY")),
            model=self.llm_config.get("executor_model", "gemini-pro") # Example model
        )
        self.validator_llm = UniversalLLMClient(
            provider=self.llm_config.get("validator_provider", "deepseek"), # Default to deepseek
            api_key=os.getenv(self.llm_config.get("validator_api_key_env", "DEEPSEEK_API_KEY")),
            model=self.llm_config.get("validator_model", "deepseek-coder") # Example model
        )
        # TODO: Add specific GeminiClient/DeepSeekV3Client if needed

        self.reasoning_tree = self._parse_reasoning_tree()
        self.max_retries = self.config.get("project_builder", {}).get("max_retries", 1)
        self.test_runner_command = self.config.get("project_builder", {}).get("test_runner_command", "pytest") # Example

        logger.info(f"ProjectBuilder initialized for project directory: {self.project_dir}")

    def _parse_reasoning_tree(self) -> Dict:
        """Loads and potentially validates the reasoning tree JSON."""
        logger.info(f"Parsing reasoning tree from: {self.reasoning_tree_path}")
        try:
            with open(self.reasoning_tree_path, 'r', encoding='utf-8') as f:
                # Handle potential large file size if necessary - consider streaming/chunking later
                tree = json.load(f)
            # TODO: Add validation of the tree structure if needed
            logger.info("Reasoning tree parsed successfully.")
            return tree
        except FileNotFoundError:
            logger.error(f"Reasoning tree file not found: {self.reasoning_tree_path}")
            raise ProjectBuilderError(f"Reasoning tree file not found: {self.reasoning_tree_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding reasoning tree JSON: {e}")
            raise ProjectBuilderError(f"Error decoding reasoning tree JSON: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing the reasoning tree: {e}")
            raise ProjectBuilderError(f"Failed to parse reasoning tree: {e}")

    def _parse_llm_response_for_actions(self, response: str) -> List[Dict]:
        """
        Parses the LLM response to extract file operations or analysis.
        Looks for patterns like '=== File: path/to/file ===\ncontent...'
        or commands like 'mkdir path/to/dir'.
        """
        actions = []
        # Simple pattern matching for file blocks
        file_pattern = r"=== File: (.*?) ===\n(.*?)(?=\n=== File: |\Z)"
        import re
        matches = re.finditer(file_pattern, response, re.DOTALL | re.MULTILINE)

        files_found = False
        for match in matches:
            files_found = True
            path = match.group(1).strip()
            content = match.group(2).strip()
            actions.append({"type": "file", "path": path, "content": content})
            logger.debug(f"Parsed file action: path='{path}'")

        # Simple pattern matching for mkdir commands (example)
        mkdir_pattern = r"mkdir\s+(-p\s+)?(.*)"
        mkdir_matches = re.finditer(mkdir_pattern, response, re.IGNORECASE)
        for match in mkdir_matches:
            # files_found = True # Don't mark as file found for mkdir
            path = match.group(2).strip().strip("'\"")
            actions.append({"type": "command", "command": f"mkdir -p {path}"}) # Use -p for safety
            logger.debug(f"Parsed command action: mkdir -p '{path}'")

        # If no specific actions found, treat the whole response as analysis/summary
        if not actions:
             actions.append({"type": "analysis", "summary": response})
             logger.debug("Parsed response as analysis/summary.")

        return actions

    def _execute_step(self, instruction: str, context: Dict) -> Dict:
        """Sends instruction to the executor LLM and prepares actions/tool calls."""
        logger.info(f"Executing step: {instruction}")
        # This list will hold descriptions of actions needed (tool calls or internal state changes)
        prepared_actions = []
        analysis_summary = None

        try:
            # Construct prompt for Gemini
            prompt = f"""You are an expert software developer tasked with building a project step-by-step.
Current Project Context:
{json.dumps(context, indent=2, default=str)}

Instruction:
{instruction}

Perform the instruction precisely.
- If creating or modifying a file, use the format:
=== File: path/relative/to/project/root/file.ext ===
[COMPLETE file content here]
=== File: path/relative/to/project/root/another_file.ext ===
[COMPLETE file content here]
- If creating a directory, output the command: `mkdir -p path/relative/to/project/root/directory`
- If the instruction requires analysis or research without file output, provide a concise summary of your findings.
- Ensure all paths are relative to the project root directory: '{self.project_dir}'.
"""
            response = self.executor_llm.generate_text(prompt)
            logger.debug(f"Executor LLM raw response received.")

            # Parse response for actions
            actions = self._parse_llm_response_for_actions(response)

            # Prepare tool calls based on parsed actions
            for action in actions:
                action_type = action.get("type")
                if action_type == "file":
                    rel_path = action.get("path")
                    content = action.get("content")
                    if rel_path:
                        target_path = self.project_dir / rel_path
                        # Basic safety check
                        if self.project_dir not in target_path.parents:
                             logger.warning(f"Skipping file action outside project dir: {rel_path}")
                             continue
                        try:
                            # Prepare write_to_file tool call data
                            prepared_actions.append({
                                "tool_name": "write_to_file",
                                "params": {"path": str(target_path), "content": content},
                                "log_message": f"Prepared write_to_file action for: {target_path}"
                            })
                        except Exception as e: # Catch potential errors during path manipulation or logging
                            # Correctly indented block
                            logger.error(f"Error preparing file action for {rel_path}: {e}")
                            # Re-raise as ProjectBuilderError to be caught by the outer handler
                            raise ProjectBuilderError(f"Error preparing file action for {rel_path}: {e}")
                    else: # This else corresponds to 'if rel_path:'
                        logger.warning("File action found but path is missing.")

                elif action_type == "command":
                    command = action.get("command")
                    if command and command.startswith("mkdir"):
                        mkdir_path_str = command.split(maxsplit=2)[-1].strip().strip("'\"")
                        target_dir = (self.project_dir / mkdir_path_str).resolve()
                        # Basic safety check
                        if self.project_dir.resolve() in target_dir.parents or target_dir == self.project_dir.resolve():
                            mkdir_target_path_abs = target_dir
                            mkdir_rel_path = mkdir_target_path_abs.relative_to(self.project_dir.resolve())
                            mkdir_cmd_in_proj = f"mkdir -p \"{mkdir_rel_path}\""
                            full_mkdir_cmd = f"cd \"{self.project_dir.resolve()}\" && {mkdir_cmd_in_proj}"
                            # Prepare execute_command tool call data
                            prepared_actions.append({
                                "tool_name": "execute_command",
                                "params": {"command": full_mkdir_cmd, "requires_approval": False},
                                "log_message": f"Prepared execute_command action: {full_mkdir_cmd}"
                            })
                        else:
                            logger.warning(f"Skipping command action outside project dir: {command}")
                    else:
                        logger.warning(f"Unsupported command action: {command}")

                elif action_type == "analysis":
                    analysis_summary = action.get("summary")
                    logger.info("Step resulted in analysis/summary.")
                    # No tool call needed, just record the summary

            # Return the raw response and the list of prepared actions
            # The caller (AI loop) will execute these actions using tools
            return {
                "status": "pending_actions", # Indicate actions need execution
                "output": response, # Keep raw response for validation
                "prepared_actions": prepared_actions,
                "analysis_summary": analysis_summary
                # 'files_modified' and 'commands_executed' will be populated by the caller after tool execution
            }

        except LLMClientError as e:
            logger.error(f"Executor LLM error during step execution: {e}")
            return {"status": "error", "error_message": str(e)}
        except ProjectBuilderError as e: # Catch errors from applying actions
             logger.error(f"Error applying action during step execution: {e}")
             return {"status": "error", "error_message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during step execution preparation: {e}", exc_info=True)
            return {"status": "error", "error_message": f"Unexpected error: {e}"}


    def _validate_step(self, instruction: str, execution_summary: Dict, context: Dict) -> Dict:
        """
        Sends execution summary (after tools are run) to the validator LLM.
        Note: execution_summary should contain info about actual tool results.
        """
        logger.info(f"Validating step: {instruction}")
        try:
            # Determine step type based on summary of executed actions
            step_type = "unknown"
            files_modified = execution_summary.get("files_modified", [])
            commands_executed = execution_summary.get("commands_executed", [])
            analysis_summary = execution_summary.get("analysis_summary") # Passed from execute_step

            if commands_executed and any(cmd.startswith("mkdir") for cmd in commands_executed):
                 step_type = "file_op"
            elif files_modified and any(f.endswith(('.py', '.js', '.ts', '.java', '.cs', '.go', '.rs')) for f in files_modified):
                step_type = "code"
            elif files_modified and any(f.endswith('.md') for f in files_modified):
                step_type = "docs"
            elif analysis_summary:
                 step_type = "analysis"
            elif files_modified:
                 step_type = "file_op"

            # Prepare validation prompt for DeepSeek
            validation_context = {
                "instruction": instruction,
                "execution_summary": execution_summary, # Use the summary passed in
                # TODO: Optionally add read_file calls here if content needed for validation
            }

            validation_prompt = f"""
Context:
{json.dumps(context, indent=2, default=str)}

Instruction Given To Executor:
{instruction}

Executor Action Summary (Result of Tool Execution):
{json.dumps(validation_context['execution_summary'], indent=2)}

Task: Validate if the executor successfully completed the instruction based on the summary of actions taken.

Validation Criteria ({step_type}):
- Adherence: Did the actions taken (files created/modified, commands run, analysis provided) directly address the instruction?
- Completeness: Do the actions seem complete for the given instruction? (e.g., if asked to create file X, was file X created according to the summary?)
- Correctness (Basic Check): Are there obvious errors suggested by the summary? (e.g., command failed, wrong file type modified).
- Relevance: Are the actions relevant to the instruction and context?

Provide a clear 'PASS' or 'FAIL' status on the first line, followed by specific feedback, especially on failure.
Example PASS:
PASS
The executor correctly created the specified file 'src/main.py' according to the summary.

Example FAIL:
FAIL
The instruction asked to create 'README.md', but the summary indicates 'config.yaml' was modified instead.
"""
            # Use the validator LLM client
            validation_response = self.validator_llm.generate_text(validation_prompt)
            logger.debug(f"Validator LLM raw response: {validation_response}")

            # Parse validation response
            response_lines = validation_response.strip().split('\n', 1)
            status_line = response_lines[0].strip().upper()
            feedback = response_lines[1].strip() if len(response_lines) > 1 else "No feedback provided."

            if status_line == "PASS":
                logger.info("Validation PASSED.")
                return {"status": "pass", "feedback": feedback}
            elif status_line == "FAIL":
                 logger.warning(f"Validation FAILED: {feedback}")
                 return {"status": "fail", "feedback": feedback}
            else:
                 # Treat ambiguous response as failure
                 logger.warning(f"Ambiguous validation response (treating as FAIL): {validation_response}")
                 return {"status": "fail", "feedback": f"Ambiguous response: {validation_response}"}

        except LLMClientError as e:
            logger.error(f"Validator LLM error during step validation: {e}")
            return {"status": "error", "feedback": f"Validator LLM error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during step validation: {e}", exc_info=True)
            return {"status": "error", "feedback": f"Unexpected validation error: {e}"}


    def _generate_tests(self, file_path: str, code: str, instruction: str) -> List[Dict]:
        """Asks an LLM to generate unit tests and prepares file write actions."""
        logger.info(f"Generating tests for file: {file_path}")
        prepared_actions = []
        try:
            language = "python" # Default
            if file_path.endswith(".js") or file_path.endswith(".ts"):
                language = "javascript"

            test_framework_hint = "pytest" if language == "python" else "jest"

            prompt = f"""Instruction that generated the code:
{instruction}

Code generated for file '{file_path}':
```
{code}
```

Task: Generate relevant unit tests for the provided code, focusing on the functionality described or implied by the original instruction.
- Use the {test_framework_hint} testing framework.
- Place the test code within the standard test file naming convention (e.g., 'tests/test_{os.path.basename(file_path)}'). Ensure the path is relative to the project root.
- Ensure tests cover basic functionality and potential edge cases mentioned or implied in the instruction.
- Output the test file content using the format:
=== File: path/to/test_file.ext ===
[COMPLETE test file content here]
"""
            response = self.executor_llm.generate_text(prompt)
            logger.debug("Test generation LLM response received.")

            # Parse response for the test file(s)
            actions = self._parse_llm_response_for_actions(response)
            for action in actions:
                if action.get("type") == "file":
                    test_path_rel = action.get("path")
                    test_content = action.get("content")
                    if test_path_rel and test_content:
                        # Prepare write_to_file tool call data
                        test_target_path = self.project_dir / test_path_rel
                        if self.project_dir not in test_target_path.parents:
                             logger.warning(f"Skipping test file outside project dir: {test_path_rel}")
                             continue
                        prepared_actions.append({
                            "tool_name": "write_to_file",
                            "params": {"path": str(test_target_path), "content": test_content},
                            "log_message": f"Prepared write_to_file action for test file: {test_target_path}"
                        })
                        logger.info(f"Prepared test file action: {test_target_path}")

            return prepared_actions

        except LLMClientError as e:
            logger.error(f"LLM error during test generation: {e}")
            return [] # Return empty list on error
        except Exception as e:
            logger.error(f"Unexpected error during test generation: {e}", exc_info=True)
            return []


    def _run_tests(self, test_file_paths: List[str]) -> Dict:
        """Prepares the execute_command action for running tests."""
        if not test_file_paths:
             return {"status": "skipped", "output": "No test files provided."} # No action needed

        logger.info(f"Preparing to run tests for: {test_file_paths}")
        activate_cmd = ""
        # Basic check for venv (adjust path separators for OS)
        venv_path = self.project_dir / "venv"
        activate_script_path = ""
        if os.name == 'nt':
            activate_script_path = venv_path / "Scripts" / "activate.bat"
        else:
            activate_script_path = venv_path / "bin" / "activate"

        if activate_script_path.exists():
             activate_cmd = f"\"{activate_script_path}\" && " if os.name == 'nt' else f"source \"{activate_script_path}\" && "

        test_command = self.test_runner_command
        # Command needs to cd into project dir, activate venv (if exists), then run tests
        full_command = f"cd \"{self.project_dir.resolve()}\" && {activate_cmd}{test_command}"
        logger.debug(f"Prepared test execution command: {full_command}")

        # Prepare execute_command tool call data
        action = {
            "tool_name": "execute_command",
            "params": {"command": full_command, "requires_approval": False},
            "log_message": f"Prepared execute_command action for running tests: {full_command}"
        }
        # This method returns the action to be executed by the caller
        # The caller needs to interpret the result of the execute_command tool
        return {"status": "pending_action", "action": action}


    def _attempt_fix(self, file_path: str, code: str, test_failures: str) -> List[Dict]:
        """Asks an LLM to fix code and prepares file write actions."""
        logger.info(f"Attempting to fix code in file: {file_path}")
        prepared_actions = []
        try:
            prompt = f"""The following code in file '{file_path}' failed unit tests:
```
{code}
```

The failing test output is:
```
{test_failures}
```

Task: Analyze the code and the test failures. Modify the original code to fix the issues highlighted by the failing tests.
- Only provide the corrected code for the specified file.
- Ensure the fix directly addresses the reported errors.
- Output the corrected file content using the format:
=== File: {file_path} ===
[COMPLETE corrected file content here]
"""
            response = self.executor_llm.generate_text(prompt)
            logger.debug("Code fix LLM response received.")

            # Parse response for the fixed file
            actions = self._parse_llm_response_for_actions(response)
            for action in actions:
                 # Ensure it's the correct file being fixed
                if action.get("type") == "file" and action.get("path") == file_path:
                    fixed_content = action.get("content")
                    if fixed_content:
                        # Prepare write_to_file tool call data
                        target_path = self.project_dir / file_path
                        prepared_actions.append({
                            "tool_name": "write_to_file",
                            "params": {"path": str(target_path), "content": fixed_content},
                            "log_message": f"Prepared write_to_file action for fixed code: {target_path}"
                        })
                        logger.info(f"Prepared potential fix action for file: {file_path}")
                        break # Assume only one fix per response

            return prepared_actions

        except LLMClientError as e:
            logger.error(f"LLM error during code fix attempt: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during code fix attempt: {e}", exc_info=True)
            return []


    def build(self):
        """
        Iterates through the reasoning tree, prepares actions for each step,
        and relies on the caller (AI loop) to execute tools and provide results.
        This version is primarily simulation and logging due to tool execution constraints.
        """
        logger.info("Starting project build process (Simulation Mode)...")
        if not self.reasoning_tree:
            logger.error("Reasoning tree is not loaded. Aborting build.")
            return

        context = {"project_dir": str(self.project_dir)}
        all_test_paths = [] # Keep track of all generated test files

        # --- Outer loop (AI Controller Simulation) ---
        # This loop would normally be outside the build method, managed by the AI controller.
        # It would call build_step, execute tools, and feed results back.
        # Here, we simulate one pass.

        for phase_name, tasks in self.reasoning_tree.items():
            logger.info(f"--- Starting Phase: {phase_name} ---")
            context["current_phase"] = phase_name
            for task_name, steps in tasks.items():
                logger.info(f"--- Starting Task: {task_name} ---")
                context["current_task"] = task_name
                if not isinstance(steps, list):
                    logger.warning(f"Steps for task '{task_name}' is not a list. Skipping.")
                    continue

                for step_data in steps:
                    if not isinstance(step_data, dict) or len(step_data) != 1:
                        logger.warning(f"Invalid step format in task '{task_name}'. Skipping step: {step_data}")
                        continue

                    step_key, instruction = list(step_data.items())[0]
                    logger.info(f"--- Processing Step: {step_key} ---")
                    context["current_step"] = step_key

                    step_success = False
                    for attempt in range(self.max_retries + 1):
                        logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1} for step {step_key}")

                        # 1. Execute Step (Prepare Actions)
                        step_result = self._execute_step(instruction, context)
                        if step_result["status"] == "error":
                            logger.error(f"Step execution preparation failed: {step_result['error_message']}")
                            break # Cannot proceed with this step

                        # --- Tool Execution Simulation (External Loop) ---
                        # The AI controller would now iterate through step_result["prepared_actions"]
                        # and execute the tools (write_to_file, execute_command).
                        # We simulate this and collect assumed results.
                        simulated_execution_summary = {
                            "files_modified": [],
                            "commands_executed": [],
                            "analysis_summary": step_result.get("analysis_summary"),
                            "tool_errors": []
                        }
                        logger.info("--- Simulating Tool Execution ---")
                        for action_data in step_result.get("prepared_actions", []):
                            logger.info(action_data["log_message"])
                            # Simulate success, populate summary
                            if action_data["tool_name"] == "write_to_file":
                                simulated_execution_summary["files_modified"].append(
                                    str(Path(action_data["params"]["path"]).relative_to(self.project_dir))
                                )
                            elif action_data["tool_name"] == "execute_command":
                                 simulated_execution_summary["commands_executed"].append(
                                     action_data["params"]["command"]
                                 )
                        logger.info("--- End Tool Execution Simulation ---")
                        # --- End Tool Execution Simulation ---

                        # 2. Validate Step (Using Simulated Summary)
                        validation_result = self._validate_step(instruction, simulated_execution_summary, context)
                        if validation_result["status"] == "error":
                            logger.error(f"Step validation failed: {validation_result['feedback']}")
                            break
                        elif validation_result["status"] == "fail":
                            logger.warning(f"Validation failed: {validation_result['feedback']}")
                            if attempt < self.max_retries:
                                logger.info("Retrying step after validation failure.")
                                context["last_validation_feedback"] = validation_result['feedback']
                                time.sleep(1) # Shorter delay for simulation
                                continue # Next attempt
                            else:
                                logger.error("Max retries reached after validation failure.")
                                break # Failed validation

                        # 3. Test Step Simulation (if applicable)
                        code_files_modified = simulated_execution_summary.get("files_modified", [])
                        code_files_for_testing = [
                            f for f in code_files_modified
                            if f.endswith(('.py', '.js', '.ts', '.java', '.cs', '.go', '.rs'))
                        ]
                        needs_testing = bool(code_files_for_testing)

                        if needs_testing:
                            logger.info(f"Code files modified ({code_files_for_testing}), proceeding with testing simulation.")
                            all_tests_passed_simulation = True

                            for file_path_rel in code_files_for_testing:
                                # 3a. Generate Tests (Prepare Actions)
                                # TODO: Need actual code content. Read file? Assume available in context?
                                # For simulation, use placeholder content.
                                logger.warning(f"Simulating test generation for {file_path_rel}. Needs actual file content.")
                                code_content_placeholder = f"# Placeholder code for {file_path_rel}"
                                test_gen_actions = self._generate_tests(file_path_rel, code_content_placeholder, instruction)

                                if test_gen_actions:
                                    # --- Tool Execution Simulation (Write Test Files) ---
                                    logger.info("--- Simulating Writing Test Files ---")
                                    current_test_paths_rel = []
                                    for test_action in test_gen_actions:
                                        logger.info(test_action["log_message"])
                                        test_file_rel = str(Path(test_action["params"]["path"]).relative_to(self.project_dir))
                                        current_test_paths_rel.append(test_file_rel)
                                        if test_file_rel not in all_test_paths:
                                             all_test_paths.append(test_file_rel)
                                    logger.info("--- End Test File Writing Simulation ---")
                                    # --- End Tool Execution Simulation ---

                                    # 3b. Run Tests (Prepare Action)
                                    run_tests_result = self._run_tests(all_test_paths) # Use all known test paths

                                    if run_tests_result.get("status") == "pending_action":
                                        test_run_action = run_tests_result["action"]
                                        # --- Tool Execution Simulation (Run Tests) ---
                                        logger.info("--- Simulating Test Execution ---")
                                        logger.info(test_run_action["log_message"])
                                        # Simulate test result (e.g., always pass for now)
                                        simulated_test_tool_result = {"status": "pass", "output": "Simulated test pass"}
                                        logger.info(f"Simulated Test Result: {simulated_test_tool_result['status']}")
                                        logger.info("--- End Test Execution Simulation ---")
                                        # --- End Tool Execution Simulation ---

                                        if simulated_test_tool_result["status"] == "error":
                                            logger.error(f"Test execution failed: {simulated_test_tool_result.get('output')}")
                                            all_tests_passed_simulation = False
                                            break # Stop testing for this step
                                        elif simulated_test_tool_result["status"] == "fail":
                                            logger.warning(f"Tests failed: {simulated_test_tool_result.get('output')}")
                                            all_tests_passed_simulation = False

                                            # 3c. Attempt Fix Simulation (Prepare Actions)
                                            if attempt < self.max_retries:
                                                logger.info("Attempting to fix code based on test failures (Simulation).")
                                                # TODO: Need actual code content
                                                fix_actions = self._attempt_fix(file_path_rel, code_content_placeholder, simulated_test_tool_result.get("output", ""))
                                                if fix_actions:
                                                    # --- Tool Execution Simulation (Apply Fix) ---
                                                    logger.info("--- Simulating Applying Code Fix ---")
                                                    for fix_action in fix_actions:
                                                         logger.info(fix_action["log_message"])
                                                    logger.info("--- End Code Fix Simulation ---")
                                                    # --- End Tool Execution Simulation ---
                                                    context["last_test_failure"] = simulated_test_tool_result.get("output")
                                                    time.sleep(1)
                                                    break # Break file loop, retry step

                                                else:
                                                    logger.error("Code fix attempt failed to generate fix.")
                                                    break # Break file loop

                                            else:
                                                logger.error("Max retries reached after test failure.")
                                                break # Break file loop
                                        else: # Tests passed
                                            logger.info(f"Simulated tests passed for {file_path_rel}.")
                                            pass # Continue file loop
                                    else: # _run_tests returned skipped or error
                                         logger.error("Could not prepare test run action.")
                                         all_tests_passed_simulation = False
                                         break # Stop testing

                                else: # Test generation failed
                                    logger.warning(f"Test generation failed or skipped for {file_path_rel}.")
                                    # Continue simulation, assuming non-critical failure

                            # Check if file loop was broken
                            if not all_tests_passed_simulation or 'last_test_failure' in context:
                                break # Break attempt loop

                            # If loop finished and all tests passed
                            if all_tests_passed_simulation:
                                logger.info("All simulated tests passed for this step.")
                                step_success = True
                                break # Success

                        else: # No code testing needed
                            logger.info("Validation passed. No code testing required for this step.")
                            step_success = True
                            break # Success

                    # End of retry loop for a step
                    if not step_success:
                         if attempt < self.max_retries and ('last_validation_feedback' in context or 'last_test_failure' in context):
                             continue # Go to next attempt

                         logger.error(f"Failed to complete step {step_key} after {self.max_retries + 1} attempts. Halting build.")
                         return # Halt build

                    # Update context (simplified for simulation)
                    context["last_step_summary"] = simulated_execution_summary
                    context.pop("last_validation_feedback", None)
                    context.pop("last_test_failure", None)

                    logger.info(f"--- Successfully Completed Step Simulation: {step_key} ---")
                    time.sleep(0.5) # Shorter delay

                logger.info(f"--- Completed Task Simulation: {task_name} ---")
                context.pop("current_task", None)
                context.pop("current_step", None)

            logger.info(f"--- Completed Phase Simulation: {phase_name} ---")
            context.pop("current_phase", None)

        logger.info("Project build process simulation completed.")


# Example usage (for testing purposes, typically called from main.py)
if __name__ == "__main__":
    # This part is for basic testing/debugging, not production use
    logging.basicConfig(level=logging.INFO)
    logger.info("Running ProjectBuilder example...")

    # Create dummy files/dirs if they don't exist
    dummy_tree_path = Path("dummy_reasoning_tree.json")
    dummy_config_path = Path("dummy_config.yaml")
    dummy_llm_config_path = Path("dummy_llm_config.json")
    dummy_project_dir = Path("dummy_generated_project")

    if not dummy_tree_path.exists():
        dummy_tree = {
            "Phase 1": {
                "Task 1.1": [
                    {"step 1": "Create a README.md file with 'Hello World'."},
                    {"step 2": "Create a src directory."},
                    {"step 3": "Create a file src/main.py with a print statement."}
                ]
            }
        }
        with open(dummy_tree_path, 'w') as f:
            json.dump(dummy_tree, f, indent=2)
        logger.info(f"Created dummy reasoning tree: {dummy_tree_path}")

    if not dummy_config_path.exists():
        dummy_config_content = """
project_builder:
  max_retries: 1
  test_runner_command: pytest
"""
        with open(dummy_config_path, 'w') as f:
            f.write(dummy_config_content)
        logger.info(f"Created dummy config: {dummy_config_path}")

    if not dummy_llm_config_path.exists():
        # Need to link llm_config path in ConfigLoader or create dummy
        # Assuming ConfigLoader looks for llm_config relative to main config
        dummy_llm_config_content = """
{
  "executor_provider": "gemini",
  "executor_api_key_env": "GEMINI_API_KEY",
  "executor_model": "gemini-pro",
  "validator_provider": "deepseek",
  "validator_api_key_env": "DEEPSEEK_API_KEY",
  "validator_model": "deepseek-coder"
}
"""
        # Need to adjust ConfigLoader logic or place this file correctly
        # For now, just create it next to the dummy config
        with open(dummy_llm_config_path, 'w') as f:
             f.write(dummy_llm_config_content)
        logger.info(f"Created dummy LLM config: {dummy_llm_config_path}")


    # Make sure API keys are set as environment variables for this example to run
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("DEEPSEEK_API_KEY"):
         logger.warning("API keys (GEMINI_API_KEY, DEEPSEEK_API_KEY) not found in environment. LLM calls will likely fail.")

    try:
        builder = ProjectBuilder(
            reasoning_tree_path=str(dummy_tree_path),
            config_path=str(dummy_config_path),
            project_dir=str(dummy_project_dir)
        )
        # builder.build() # Commented out by default to avoid running LLMs unintentionally
        logger.info("ProjectBuilder instance created (build() not called in example).")
    except Exception as e:
        logger.error(f"Error creating ProjectBuilder instance: {e}", exc_info=True)

    # Clean up dummy files (optional)
    # dummy_tree_path.unlink(missing_ok=True)
    # dummy_config_path.unlink(missing_ok=True)
    # dummy_llm_config_path.unlink(missing_ok=True)
    # import shutil
    # shutil.rmtree(dummy_project_dir, ignore_errors=True)