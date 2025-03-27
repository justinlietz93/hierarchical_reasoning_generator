# Hierarchical Planner with Gemini AI

A Python application that breaks down high-level goals into structured, actionable steps using Google's Gemini AI model. This tool creates a hierarchical planning structure by generating:

1. **Phases** - Major stages of the project
2. **Tasks** - Specific actionable items within each phase
3. **Steps** - Detailed instructions for completing each task

## Features

- Utilizes Google's Gemini 2.5 Pro model for intelligent task breakdown
- Generates a comprehensive hierarchical planning structure
- Outputs results as a structured JSON file
- Includes retry mechanisms and error handling

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gemini.git
   cd gemini
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r hierarchical_planner/requirements.txt
   ```

4. Set up your API key:
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. Define your high-level goal in the `hierarchical_planner/task.txt` file.
   - Example: "build me an IDE"

2. Run the planner:
   ```
   cd hierarchical_planner
   python main.py
   ```

3. Examine the generated plan in `reasoning_tree.json`

## Output Structure

The planner generates a JSON file with the following structure:

```json
{
  "Phase 1: Planning and Design": {
    "Task 1.1: Define requirements": [
      {"step 1": "Create a document outlining core IDE features..."},
      {"step 2": "Research existing IDEs..."}
    ]
  },
  "Phase 2: Core Implementation": {
    "Task 2.1: Set up project structure": [
      {"step 1": "Create the main project directory..."}
    ]
  }
}
```

## Autonomous Project Development

The generated reasoning tree serves as a comprehensive blueprint for autonomous project development:

- **AI Agent Guidance**: The hierarchical structure provides clear, step-by-step instructions that can guide AI coding agents through complex development processes
- **Autonomous Execution**: Each step is designed to be actionable and specific, enabling automated systems to execute tasks with minimal human intervention
- **Project Tracking**: The structured format facilitates automated progress tracking and reporting throughout the development lifecycle
- **Scalable Development**: By breaking down complex goals into manageable pieces, the system enables autonomous handling of increasingly complex projects
- **Integration with Development Tools**: The JSON output can be integrated with CI/CD pipelines, project management tools, or other AI systems to automate the entire development workflow

The detailed prompts generated at the step level contain sufficient context and specificity for AI systems to independently implement each component, gradually building towards the complete project goal.

## Requirements

- Python 3.8+
- Google Gemini API key
- Required Python packages (see requirements.txt):
  - google-generativeai
  - python-dotenv

## License

MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- This project uses Google's Generative AI (Gemini) for content generation