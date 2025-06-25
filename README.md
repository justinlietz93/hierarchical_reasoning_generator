# Hierarchical Reasoning Generator

A sophisticated Python application that automates software project planning and generation by breaking down high-level goals into structured, actionable steps using Large Language Models (LLMs). The system can then use the generated plan to build a complete project structure, including source code, documentation, and tests.

## Features

-   **Multi-Provider LLM Support**: Utilizes various LLM providers (Google Gemini, Anthropic, Deepseek) for intelligent task breakdown, execution, and validation.
-   **Hierarchical Planning**: Generates a comprehensive planning structure (Phases → Tasks → Steps) stored in `reasoning_tree.json`.
-   **Project Constitution**: Establishes foundational rules for a project in a `project_constitution.json` file to ensure consistency and prevent context drift.
-   **QA Validation (Optional)**: Analyzes the generated plan for structural integrity, goal alignment, and clarity, outputting an annotated plan.
-   **Project Builder**: An execution engine that interprets the reasoning tree, using a dual-LLM system (an "executor" and a "validator") to write code, create files, and run tests in a self-correcting loop.
-   **Persona Builder**: A utility to generate and manage AI personas from unstructured text, creating structured profiles to guide LLM behavior.
-   **Configuration Management**: Manages API keys, model parameters, file paths, and logging settings via `hierarchical_planner/config/config.yaml` and `.env` files.
-   **Checkpointing & Resumption**: Automatically saves progress during plan generation and can resume from the last checkpoint if interrupted.
-   **Logging**: Implements configurable logging to console and/or file.
-   **Error Handling**: Includes custom exceptions and retry mechanisms for API calls and file operations.
-   **Unit Tests**: Provides `pytest` unit tests for core components.

## Project Structure

```
hierarchical_reasoning_generator/
├── .gitignore
├── README.md
├── PROJECT_REVIEW.md
└── hierarchical_planner/
    ├── config/
    │   ├── config.yaml
    │   ├── llm_config.json
    │   └── project_constitution_schema.json
    ├── persona_builder/
    │   ├── personas/
    │   ├── cli.py
    │   ├── parser.py
    │   └── ...
    ├── tests/
    │   ├── test_config_loader.py
    │   └── ...
    ├── .env                # Recommended for API keys
    ├── main.py             # Main script for planning and building
    ├── project_builder.py  # Core project generation logic
    ├── qa_validator.py     # QA validation logic
    ├── universal_LLM_client.py # Client for multiple LLM providers
    ├── requirements.txt
    └── task.txt            # Input file for the high-level goal
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/justinlietz93/hierarchical_reasoning_generator.git
    cd hierarchical_reasoning_generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r hierarchical_planner/requirements.txt
    # For development/testing:
    pip install -r hierarchical_planner/requirements-dev.txt
    ```

4.  **Configure API Keys:**
    *   Create a `.env` file in the `hierarchical_planner/` directory.
    *   Add your API keys to the `.env` file. The application will automatically load them. Example:
        ```dotenv
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
        DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
        ```

5.  **Review Configuration:**
    *   Modify `hierarchical_planner/config/config.yaml` and `hierarchical_planner/config/llm_config.json` to adjust models, API parameters, file paths, or logging settings as needed.

## Usage

### 1. Define Goal
Edit `hierarchical_planner/task.txt` with your high-level goal (e.g., "build me an IDE").

### 2. Run the Planner
Navigate to the `hierarchical_planner` directory and run `main.py`.

```bash
cd hierarchical_planner
python main.py [OPTIONS]
```

**Common Workflows:**

*   **Generate a Plan:**
    ```bash
    python main.py
    ```
    This generates `project_constitution.json` and `reasoning_tree.json`.

*   **Generate and Validate a Plan:**
    ```bash
    python main.py
    ```
    (By default, QA is enabled). This produces `reasoning_tree_validated.json`. To skip: `python main.py --skip-qa`.

*   **Build a Project from a Plan:**
    ```bash
    python main.py --build
    ```
    This will use the generated `reasoning_tree.json` (or the validated one if it exists) to build the project in the `generated_project` directory.

### Command-Line Options

*   `--task-file PATH`: Specify a different input task file.
*   `--output-file PATH`: Specify a different output file for the plan.
*   `--validated-output-file PATH`: Specify a different output file for the QA-validated plan.
*   `--skip-qa`: Skip the QA validation step.
*   `--no-resume`: Start a new plan from scratch, ignoring any existing checkpoints.
*   `--build`: Run the Project Builder to generate the project from the reasoning tree.
*   `--project-dir PATH`: Specify the directory for the generated project.
*   `--provider [gemini|anthropic|deepseek]`: Force the use of a specific LLM provider.
*   `--validate-only`: Run only the QA validation on an existing plan.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
