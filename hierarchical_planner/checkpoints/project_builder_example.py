#!/usr/bin/env python3

"""
project_builder.py

An AI-powered project builder that:
1) Takes a user's project description/requirements
2) Plans and designs the project architecture
3) Creates a project structure with folders and files
4) Implements each file with appropriate code
5) Provides assembly instructions and usage guidance

This system walks an LLM through creating a complete, functional project
from scratch based on the user's requirements.

IMPORTANT: This module is NOT meant to be run directly. It must be run through orchestrator.py
with the --build flag and appropriate arguments. Example:
  python orchestrator.py --build <model> [vision] [--start-step N] [--start-substep ID]

The --start-step and --start-substep arguments can be used to resume from a specific point
if the build process was interrupted.
"""

import os
import sys
import argparse 
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import subprocess
import re

from ai_clients import AIOrchestrator
from utils import ProjectFile, SubStep, read_project_files, write_project_file, parse_ai_response_and_apply

# Global token management
TOKEN_SAFETY_THRESHOLD = 12000  # Reduced to avoid rate limits

# Project directory where all files will be created
PROJECT_DIR = "generated_project"

# Add a flag to control linting (enabled by default)
ENABLE_SYNTAX_CHECKING = True

# Define alphabetical characters for sub-step IDs
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Define the project building steps with substeps
BUILDER_STEPS = [
    {
        "phase_name": "Project Planning",
        "system_prompt": """You are an expert software architect with extensive experience in project planning and design.
Your task is to analyze project requirements and create a comprehensive project plan.""",
        "user_prompt_template": """# Project Planning Phase

## Project Description:
{vision}

## Your Task:
Create a detailed project plan that outlines the vision, technology stack, and structure.

Output your work in the following files:
""",
        "sub_steps": [
            SubStep("1A", "Project Overview", """You are an expert software architect. Your task is to analyze the project requirements and create a high-level overview.

Project Description:
{vision}

Create a project overview that includes:
1. Vision Summary
2. Project Objectives
3. Expected Outcomes

Output your overview in `=== File: doc/01_project_plan_overview.md ===`"""),
            
            SubStep("1B", "Technology Stack", """You are an expert software architect. Review the project requirements and previous overview to define the technology stack.

Previous Output:
{step1A}

Project Description:
{vision}

Define a comprehensive technology stack including:
1. Core Technologies
2. Supporting Libraries
3. Development Tools

Focus on choosing technologies that are well-suited for this specific project.
Output your technology analysis in `=== File: doc/01_project_plan_tech.md ===`"""),
            
            SubStep("1C", "Project Structure", """You are an expert software architect. Based on the previous analysis, define the project's organization.

Previous Outputs:
Overview: {step1A}
Tech Stack: {step1B}

Project Description:
{vision}

Create a detailed project structure including:
1. Directory organization
2. Key files and their purposes
3. Module organization
4. Naming conventions

Output your structure plan in `=== File: doc/01_project_plan_structure.md ===`""")
        ]
    },
    {
        "phase_name": "System Architecture",
        "system_prompt": """You are an expert software architect with deep knowledge of software design patterns,
system architecture, and best practices across different technology stacks.
Your task is to design a detailed architecture for the project, focusing on components,
interfaces, data flow, and technical decisions.""",
        "user_prompt_template": """# System Architecture Phase

## Previous Phase Output:
{step1}

## Your Task:
1. Design a detailed system architecture for the project
2. Define all major components, modules, and services
3. Specify interfaces between components and external systems
4. Create diagrams (using ASCII art or text descriptions) showing system structure and data flow
5. Make clear technology choices for each component with justification
6. Design a clean, maintainable, and extensible architecture

Be specific in your architectural decisions. Consider scalability, maintainability, 
performance, and security. Make decisions that align with modern best practices.

Output your architecture document in `=== File: doc/02_system_architecture.md ===`
""",
        "sub_steps": [
            SubStep("2A", "Component Architecture", """Design the high-level component architecture:
1. Define major system components
2. Specify component responsibilities
3. Document component interactions
4. Create component diagrams
5. Explain architectural patterns used

Output in `=== File: doc/02A_component_architecture.md ===`"""),
            
            SubStep("2B", "Data Architecture", """Design the data architecture:
1. Define data models and schemas
2. Specify data flow between components
3. Document data storage solutions
4. Address data security considerations
5. Plan for data scalability

Output in `=== File: doc/02B_data_architecture.md ===`"""),
            
            SubStep("2C", "Interface Design", """Design system interfaces:
1. Define API contracts
2. Specify communication protocols
3. Document interface patterns
4. Address error handling
5. Plan for versioning

Output in `=== File: doc/02C_interface_design.md ===`""")
        ]
    },
    {
        "phase_name": "Project Structure",
        "system_prompt": """You are an expert software developer with deep knowledge of project organization,
file structures, and code organization practices. Your task is to create a complete project 
structure with all necessary folders and files.""",
        "user_prompt_template": """# Project Structure Phase

## Previous Phase Output:
{step1}

## System Architecture:
{step2}

## Your Task:
Create a comprehensive project structure that follows best practices.

Output your work in the following files:
""",
        "sub_steps": [
            SubStep("3A", "Directory Structure", """Create the directory structure:
1. Define all necessary directories
2. Explain directory purposes
3. Document organization patterns
4. Plan for future expansion
5. Consider build/deployment needs

Output in `=== File: doc/03A_directory_structure.md ===`"""),
            
            SubStep("3B", "File Templates", """Create file templates:
1. Define core file templates
2. Document file naming conventions
3. Specify file organization rules
4. Create example files
5. Include configuration templates

Output in `=== File: doc/03B_file_templates.md ===`"""),
            
            SubStep("3C", "Build Configuration", """Set up build configuration:
1. Create build scripts
2. Define dependency management
3. Configure development tools
4. Set up CI/CD templates
5. Document build process

Output in `=== File: doc/03C_build_configuration.md ===`""")
        ]
    },
    {
        "phase_name": "Implementation Plan",
        "system_prompt": """You are an expert software developer and technical project manager.
Your task is to create a detailed implementation plan that outlines how to approach building
the project in a systematic way. Focus on breaking down the implementation into manageable tasks.""",
        "user_prompt_template": """# Implementation Plan Phase

## Project Plan:
{step1}

## System Architecture:
{step2}

## Project Structure:
{step3}

## Your Task:
1. Create a detailed implementation plan with clear steps
2. Prioritize files in order of implementation (which should be built first, second, etc.)
3. Group related implementation tasks
4. Identify any dependencies between files or components
5. Estimate relative complexity for each implementation task
6. Propose a testing strategy for the project

This plan should guide a developer through implementing the entire project in the most efficient order.

Output your implementation plan in `=== File: doc/04_implementation_plan.md ===`

Additionally, create a script that will set up the basic project structure:
Output this setup script in `=== File: setup_project.py ===`
The script should create all necessary directories and empty files based on your project structure.
"""
    },
    {
        "phase_name": "File Implementation",
        "system_prompt": """You are an expert software developer with deep knowledge of various programming languages,
frameworks, and technologies. Your task is to implement code for specific files in the project,
ensuring they follow best practices, are well-documented, and fulfill their purpose in the overall system.""",
        "user_prompt_template": """# File Implementation Phase

## Project Plan:
{step1}

## System Architecture:
{step2}

## Project Structure:
{step3}

## Implementation Plan:
{step4}

## Your Task:
Now it's time to implement each file in the project. For this phase, you'll be implementing one file at a time.

For the following file:
- {current_file}

Implement the complete code for this file based on:
1. The file's purpose as defined in the project structure
2. The architectural design decisions from previous phases
3. Best practices for the technology/language being used
4. Proper documentation, error handling, and any necessary tests

Be thorough and create production-quality code that is ready to use.

Output the code in `=== File: {current_file} ===`
"""
    },
    {
        "phase_name": "Assembly and Usage Guide",
        "system_prompt": """You are an expert software developer and technical writer.
Your task is to create comprehensive documentation that explains how to assemble, configure,
and use the project. Focus on clarity, completeness, and making the project accessible to others.""",
        "user_prompt_template": """# Assembly and Usage Guide Phase

## Project Plan:
{step1}

## System Architecture:
{step2}

## Your Task:
Now that all files have been implemented, create comprehensive documentation for the project:

1. Create a detailed README.md with:
   - Project overview and purpose
   - Installation instructions
   - Configuration instructions
   - Usage examples
   - API documentation (if applicable)
   - Deployment guidelines
   - Troubleshooting tips

2. Create any additional documentation needed:
   - User manual (if applicable)
   - Developer guide
   - Maintenance instructions
   - Known limitations or issues

Be thorough but clear in your documentation. The goal is to make it easy for someone
to understand, set up, and use this project.

Output the README in `=== File: README.md ===`
Output any additional documentation in appropriate files in the doc/ directory.
"""
    }
]

def smart_text_trimmer(text: str, max_length: int, preserve_headers: bool = True) -> str:
    """
    Intelligently trim text to a maximum length while preserving structure.
    
    Args:
        text: The text to trim
        max_length: Maximum length to trim to
        preserve_headers: Whether to prioritize keeping markdown headers
        
    Returns:
        Trimmed text with indication of trimming
    """
    if not text or len(text) <= max_length:
        return text
        
    # If we need to trim, let's do it intelligently
    if preserve_headers:
        # Split by headers (##) to preserve structure
        sections = []
        current_section = ""
        for line in text.splitlines():
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)
                    current_section = line + "\n"
                else:
                    current_section = line + "\n"
            else:
                current_section += line + "\n"
                
        if current_section:
            sections.append(current_section)
            
        # If we have multiple sections, be selective
        if len(sections) > 1:
            # Calculate budget per section
            budget = max(250, max_length // len(sections))
            result = []
            remaining_budget = max_length
            
            # Process headers and critical sections first
            for section in sections:
                # Always include full header lines
                header_lines = [line for line in section.splitlines() if line.startswith('#')]
                headers = "\n".join(header_lines) + "\n"
                
                # Calculate how much content we can include
                content_budget = min(remaining_budget - len(headers), budget)
                if content_budget <= 20:  # Too small to be useful
                    # Just include the header
                    result.append(headers)
                else:
                    # Get the first content_budget chars after the header
                    content = section[len(headers):len(headers) + content_budget]
                    if len(section) > len(headers) + content_budget:
                        content += "...(content truncated)"
                    result.append(headers + content)
                
                remaining_budget -= len(result[-1])
                if remaining_budget <= 0:
                    break
                    
            return "\n".join(result)
    
    # Simple truncation with ellipsis if other methods don't apply
    return text[:max_length - 25] + "...(content truncated)"

def build_user_prompt(step_index: int, step_info: dict, sub_step_index: int, sub_step: SubStep, 
                     step_outputs: Dict[int, str], sub_step_outputs: Dict[str, str]) -> str:
    """
    Build a focused prompt for a specific substep with all necessary context.
    """
    # Using the global token budget
    global TOKEN_SAFETY_THRESHOLD
    
    # Set base importance weights (not percentages) - higher = more important
    importance_weights = {
        'vision': 3,           # Project vision is always necessary
        'prior_step': 5,       # Most relevant prior step has highest priority
        'prev_substep': 4,     # Previous substep has high importance for continuity
        'instructions': 1      # Instructions are fixed but essential
    }
    
    def estimate_tokens(text: str) -> int:
        # A rough estimator: ~4 characters per token for English text
        return len(text) // 4
    
    # Start with fixed overhead for headers and formatting
    fixed_overhead_tokens = 500  # Estimate for all the headers, formatting, etc.
    
    # Calculate instructions tokens (fixed portion)
    instructions_text = """Requirements for Your Response:
1. Be extremely thorough and detailed in your response
2. Consider all edge cases and potential issues
3. Provide clear rationale for all decisions
4. Include examples where appropriate
5. Consider security, scalability, and maintainability
6. Make your response production-ready
7. Include all necessary technical details
8. Write with clarity and precision
9. Address potential challenges and their solutions
10. Ensure completeness - don't leave aspects unexplored

## Output Format
Place your response for Sub-Step {sub_step.id} in `=== File: doc/STEP{step_index}_SUBSTEP_{sub_step.id}.md ===`
Focus only on completing this specific sub-step.

Remember: This is a professional project that requires production-quality output.
Your response should demonstrate deep expertise and thorough analysis."""
    
    # Calculate task tokens (specific to this substep)
    task_text = f"""## Your Task
{sub_step.prompt}"""
    
    # Calculate fixed tokens that are always included
    instructions_tokens = estimate_tokens(instructions_text)
    task_tokens = estimate_tokens(task_text)
    
    # Available tokens for dynamic content
    available_content_budget = TOKEN_SAFETY_THRESHOLD - fixed_overhead_tokens - instructions_tokens - task_tokens
    # Target using 95% of available token budget for safety margin
    target_content_tokens = int(available_content_budget * 0.95)
    
    # Get raw content (before trimming)
    vision = step_outputs.get('vision', '(No vision provided)')
    vision_tokens = estimate_tokens(vision)
    
    # Get prior step content - ALWAYS get the most relevant prior step content
    # regardless of whether we skipped steps or not
    prior_step_content = ""
    prior_step_tokens = 0
    if step_index > 1:
        # Determine which prior step is most relevant
        relevant_step = 0
        if step_index == 2: relevant_step = 1
        elif step_index == 3: relevant_step = 2
        elif step_index == 4: relevant_step = 3
        elif step_index == 5: relevant_step = 4
        
        if relevant_step > 0:
            # First check if we have a summary file for the entire prior step
            step_summary_file = os.path.join(PROJECT_DIR, "doc", f"STEP{relevant_step}_SUMMARY.md")
            prev_output = None
            step_title = BUILDER_STEPS[relevant_step-1].get('phase_name', f"Step {relevant_step}")
            
            if os.path.exists(step_summary_file):
                try:
                    pass
                    with open(step_summary_file, 'r', encoding='utf-8') as file:
                        prev_output = file.read()
                        print(f"Reading prior step content from summary file: {step_summary_file}")
                except Exception as e:
                    print(f"Error reading step summary file: {e}")
            
            # If no summary file, try using the in-memory step output
            if not prev_output and relevant_step in step_outputs:
                prev_output = step_outputs.get(relevant_step)
                print(f"Using in-memory prior step content for step {relevant_step}")
            
            # If we have output, add it to the context
            if prev_output and prev_output != '(No output)':
                prior_step_content = f"\n\n{step_title} Decisions:\n{prev_output}\n\n"
                prior_step_tokens = estimate_tokens(prior_step_content)
                print(f"Including prior step {relevant_step} content: {prior_step_tokens} tokens")
            else:
                # Handle missing prior step content by combining all substep files
                print(f"No summary for step {relevant_step}, attempting to reconstruct from substep files...")
                step_files = []
                
                # Get all files for the prior step
                doc_dir = os.path.join(PROJECT_DIR, "doc")
                if os.path.exists(doc_dir):
                    for filename in os.listdir(doc_dir):
                        if filename.startswith(f"STEP{relevant_step}_SUBSTEP_") and filename.endswith(".md"):
                            file_path = os.path.join(doc_dir, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as file:
                                    step_files.append((filename, file.read()))
                            except Exception as e:
                                print(f"Error reading file {file_path}: {e}")
                
                if step_files:
                    # Sort by substep ID
                    step_files.sort(key=lambda x: x[0])
                    combined_output = "\n\n".join([content for _, content in step_files])
                    prior_step_content = f"\n\n{step_title} Decisions:\n{combined_output}\n\n"
                    prior_step_tokens = estimate_tokens(prior_step_content)
                    print(f"Reconstructed prior step {relevant_step} from {len(step_files)} substep files: {prior_step_tokens} tokens")
    
    # Get previous substep content - This is where we need to be consistent
    prev_substep_content = ""
    prev_substep_tokens = 0
    
    # Get the MOST RECENT previous substep by reading directly from the files
    if sub_step_index > 0:
        # First try to get the immediately previous substep in this step
        prev_sub_step = step_info["sub_steps"][sub_step_index - 1]
        prev_file_path = os.path.join(PROJECT_DIR, "doc", f"STEP{step_index}_SUBSTEP_{prev_sub_step.id}.md")
        
        if os.path.exists(prev_file_path):
            try:
                with open(prev_file_path, 'r', encoding='utf-8') as file:
                    prev_sub_step_output = file.read()
                    prev_substep_content = f"\nPrevious Sub-Step Output ({prev_sub_step.name}):\n{prev_sub_step_output}\n\n"
                    prev_substep_tokens = estimate_tokens(prev_substep_content)
                    print(f"Including previous substep from file {prev_file_path}: {prev_substep_tokens} tokens")
            except Exception as e:
                print(f"Error reading previous substep file: {e}")
        else:
            # If we skipped directly to this substep, find the last completed substep file
            print(f"Direct previous substep file not found, looking for other substep files...")
            
            # Build a list of all possible previous substeps in reverse order
            possible_substeps = []
            
            # Add previous substeps from current step
            for i in range(sub_step_index - 1, -1, -1):
                if i < len(step_info["sub_steps"]):
                    possible_sub_step = step_info["sub_steps"][i]
                    possible_substeps.append((f"STEP{step_index}_SUBSTEP_{possible_sub_step.id}.md", possible_sub_step.name))
            
            # Add substeps from previous step if this is not step 1
            if step_index > 1:
                prev_step_info = BUILDER_STEPS[step_index - 2]  # -2 because 0-indexed list, -1 for prev step
                if "sub_steps" in prev_step_info and prev_step_info["sub_steps"]:
                    prev_sub_steps = prev_step_info["sub_steps"]
                    for i in range(len(prev_sub_steps) - 1, -1, -1):
                        possible_sub_step = prev_sub_steps[i]
                        possible_substeps.append((f"STEP{step_index-1}_SUBSTEP_{possible_sub_step.id}.md", possible_sub_step.name))
            
            # Try each possible substep in order until we find one
            for filename, sub_step_name in possible_substeps:
                file_path = os.path.join(PROJECT_DIR, "doc", filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            output = file.read()
                            prev_substep_content = f"\nPrevious Sub-Step Output ({sub_step_name}):\n{output}\n\n"
                            prev_substep_tokens = estimate_tokens(prev_substep_content)
                            print(f"Including nearest available substep from file {file_path}: {prev_substep_tokens} tokens")
                            break
                    except Exception as e:
                        print(f"Error reading substep file {file_path}: {e}")
    
    # Calculate total raw content tokens
    total_raw_tokens = vision_tokens + prior_step_tokens + prev_substep_tokens
    
    # If we have more content than our target, we need to trim
    if total_raw_tokens > target_content_tokens:
        # Calculate how much each section should get based on importance weights
        total_weight = sum(importance_weights.values())
        # Calculate token allocation for each section
        token_allocation = {
            'vision': int((importance_weights['vision'] / total_weight) * target_content_tokens),
            'prior_step': int((importance_weights['prior_step'] / total_weight) * target_content_tokens),
            'prev_substep': int((importance_weights['prev_substep'] / total_weight) * target_content_tokens),
        }
        
        # Apply consistent reduction to previous substep regardless of entry point
        # Use a fixed moderate reduction to balance context and token limits
        token_allocation['prev_substep'] = int(token_allocation['prev_substep'] * 0.8)  # Consistent 20% reduction
        print(f"Applied standard 20% reduction to previous substep content")
        
        # Trim each section to fit its allocation
        if vision_tokens > token_allocation['vision']:
            vision = smart_text_trimmer(vision, token_allocation['vision'] * 4, preserve_headers=True)
            print(f"Trimmed vision to fit within {token_allocation['vision']} token budget")
        
        if prior_step_tokens > token_allocation['prior_step']:
            prior_step_content = smart_text_trimmer(prior_step_content, token_allocation['prior_step'] * 4, preserve_headers=True)
            print(f"Trimmed prior step content to fit within {token_allocation['prior_step']} token budget")
        
        if prev_substep_tokens > token_allocation['prev_substep']:
            prev_substep_content = smart_text_trimmer(prev_substep_content, token_allocation['prev_substep'] * 4, preserve_headers=True)
            print(f"Trimmed previous substep to fit within {token_allocation['prev_substep']} token budget")
    else:
        # We have room to include everything, no trimming needed
        print(f"Including all available context ({total_raw_tokens} tokens)")
    
    # Build the final prompt
    prompt = f"""Step {step_index}, Sub-Step {sub_step.id}: {sub_step.name}

IMPORTANT: You are an expert software architect and developer with extensive experience in building
production-grade systems. Your task is to provide detailed, well-thought-out solutions
that could be immediately used in a professional development environment.

## Project Description
{vision}

## Relevant Context"""
    
    # Add the prior step content to the prompt
    prompt += prior_step_content
    
    # Add the previous substep content to the prompt
    prompt += prev_substep_content
    
    # Add specific instructions for this sub-step
    prompt += f"""## Your Task
{sub_step.prompt}

## Requirements for Your Response:
1. Be extremely thorough and detailed in your response
2. Consider all edge cases and potential issues
3. Provide clear rationale for all decisions
4. Include examples where appropriate
5. Consider security, scalability, and maintainability
6. Make your response production-ready
7. Include all necessary technical details
8. Write with clarity and precision
9. Address potential challenges and their solutions
10. Ensure completeness - don't leave aspects unexplored

## Output Format
Place your response for Sub-Step {sub_step.id} in `=== File: doc/STEP{step_index}_SUBSTEP_{sub_step.id}.md ===`
Focus only on completing this specific sub-step.

Remember: This is a professional project that requires production-quality output.
Your response should demonstrate deep expertise and thorough analysis."""
    
    return prompt

def execute_substep(orchestrator: AIOrchestrator, step_info: dict, step_index: int, 
                   sub_step_index: int, sub_step: SubStep, file_map: Dict[str, ProjectFile], 
                   step_outputs: Dict[int, str], sub_step_outputs: Dict[str, str]) -> bool:
    """Execute a single substep in the project building process."""
    
    # Reference the global TOKEN_SAFETY_THRESHOLD
    global TOKEN_SAFETY_THRESHOLD
    
    print(f"\nExecuting substep {sub_step.id}: {sub_step.name}")
    
    # DEBUG: Show what's in step_outputs for relevant steps
    print(f"\n--- DEBUG: Available Step Outputs ---")
    print(f"Current step: {step_index}")
    print(f"Looking for prior step: {step_index - 1 if step_index > 1 else 'None'}")
    print(f"Available keys in step_outputs: {list(step_outputs.keys())}")
    if step_index > 1 and step_index - 1 not in step_outputs:
        print(f"WARNING: Missing expected prior step {step_index - 1} output!")
    for key in step_outputs.keys():
        if isinstance(key, int):
            content_len = len(step_outputs[key]) if step_outputs[key] else 0
            print(f"  - Step {key}: {content_len} chars")
    print(f"-----------------------------------")
    
    # Build the focused prompt with all necessary context
    prompt = build_user_prompt(step_index, step_info, sub_step_index, sub_step, step_outputs, sub_step_outputs)
    
    # TOKEN DEBUGGING ONLY - no other changes
    def estimate_tokens(text: str) -> int:
        # A rough estimator: ~4 characters per token for English text
        return len(text) // 4
    
    # Analyze token usage
    system_prompt = step_info.get("system_prompt", "You are an expert software architect and system designer.")
    system_tokens = estimate_tokens(system_prompt)
    prompt_tokens = estimate_tokens(prompt)
    total_tokens = system_tokens + prompt_tokens
    
    print(f"\n===== TOKEN USAGE DEBUG =====")
    print(f"System prompt tokens (est.): {system_tokens}")
    print(f"User prompt tokens (est.): {prompt_tokens}")
    print(f"Total input tokens (est.): {total_tokens}")
    
    # SAFETY CHECK: If we're approaching token limits, ask for permission to trim further
    if total_tokens > TOKEN_SAFETY_THRESHOLD:
        print(f"\n⚠️ WARNING: Token count ({total_tokens}) exceeds safety threshold ({TOKEN_SAFETY_THRESHOLD})")
        print("This may cause rate limit errors with the AI API.")
        user_choice = input("Options:\n"
                          "1. Continue anyway (not recommended)\n"
                          "2. Apply aggressive trimming to reduce tokens\n"
                          "3. Skip this substep\n"
                          "Enter choice (1/2/3): ")
        
        if user_choice == "2":
            print("Applying aggressive trimming to reduce token usage...")
            
            # Determine how much we need to trim
            excess_tokens = total_tokens - TOKEN_SAFETY_THRESHOLD + 3000  # More conservative buffer (increased from 2000)
            chars_to_trim = excess_tokens * 4  # Approximate chars per token
            
            # Rebuild prompt with more aggressive trimming - use a reduced percentage of the threshold
            # This will trigger the token budgeting system to be more aggressive
            # Temporarily reduce the threshold to force more aggressive trimming
            original_threshold = TOKEN_SAFETY_THRESHOLD
            TOKEN_SAFETY_THRESHOLD = int(original_threshold * 0.8)  # More aggressive trimming (80% instead of 90%)
            
            # Rebuild with reduced threshold
            prompt = build_user_prompt(step_index, step_info, sub_step_index, sub_step, step_outputs, sub_step_outputs)
            
            # Restore original threshold
            TOKEN_SAFETY_THRESHOLD = original_threshold
            
            # Re-estimate after trimming
            new_tokens = estimate_tokens(prompt)
            print(f"After aggressive trimming: {new_tokens} tokens (est.)")
            
            # If still over threshold, trim even more drastically
            if new_tokens > TOKEN_SAFETY_THRESHOLD:
                print("Still exceeding threshold after first trimming, applying more aggressive reduction...")
                TOKEN_SAFETY_THRESHOLD = int(original_threshold * 0.7)  # Even more aggressive (70%)
                prompt = build_user_prompt(step_index, step_info, sub_step_index, sub_step, step_outputs, sub_step_outputs)
                TOKEN_SAFETY_THRESHOLD = original_threshold
                new_tokens = estimate_tokens(prompt)
                print(f"After second trimming: {new_tokens} tokens (est.)")
        
        elif user_choice == "3":
            print("Skipping this substep as requested.")
            return False
        else:
            print("Continuing with potentially risky token count...")
    
    # Add a delay before API call to help manage rate limits
    if os.environ.get("ENABLE_RATE_LIMIT_DELAY", "true").lower() == "true":
        print("Adding delay before API call to manage rate limits...")
        time.sleep(5)  # 5 second delay between API calls
    
    # Set max output tokens based on the step
    # For file implementation (step 5), use the highest possible value
    # For other steps, use a more moderate value
    max_output_tokens = 64000  # Maximum allowed for Claude models
    print(f"Setting max output tokens to {max_output_tokens} for {'file implementation' if step_index == 5 else 'regular step'}")
    
    # Use temperature 0.0 for file implementation (step 5) for deterministic output
    # For other steps, use a moderate temperature for more creativity
    temperature = 0.0 if step_index == 5 else 0.2
    print(f"Using temperature {temperature} for {'file implementation' if step_index == 5 else 'regular step'}")
    
    # Customize system prompt for file implementation step to emphasize complete functional code
    if step_index == 5:
        system_prompt = """You are an expert software engineer implementing a critical file in a complex project.
CRITICAL INSTRUCTIONS:
1. Write COMPLETE, FUNCTIONAL code that can be used in a production environment
2. Do NOT write pseudo-code or example code
3. Do NOT include comments like "This is just a demonstration" or "This is a simplified version"
4. Implement FULL functionality according to requirements
5. Include ALL necessary imports, constants, error handling, and logic
6. Your code will be directly saved to a file and must work without modifications. You are responsible for ensuring the code is correct and functional."""
    
    # Call the AI with the focused prompt
    ai_response = orchestrator.call_llm(system_prompt, prompt, max_tokens=max_output_tokens, temperature=temperature)
    
    if not ai_response or ai_response.startswith("ERROR"):
        print(f"Error in substep {sub_step.id}: {ai_response}")
        return False
        
    # Store the output - store with the substep ID for consistency
    sub_step_key = f"step{step_index}_{sub_step.id}"
    sub_step_outputs[sub_step_key] = ai_response
    
    # Process any file markers in the response
    current_file_map = {}  # Use a temporary map for just this substep
    parse_ai_response_and_apply(ai_response, current_file_map)
    
    # Write only the new files to disk and update the main file_map
    for rel_path, pf in current_file_map.items():
        write_project_file(PROJECT_DIR, pf)
        file_map[rel_path] = pf  # Update main file_map with new/updated files
    
    return True

def extract_files_from_structure(structure_content: str) -> List[str]:
    """
    Extract file paths from the project structure document.
    
    Args:
        structure_content: Content of the project structure document
        
    Returns:
        List[str]: List of file paths to implement
    """
    files = []
    lines = structure_content.splitlines()
    
    for line in lines:
        # Look for file paths in the document
        # This is a simple heuristic and might need adjustment based on format
        if line.strip().endswith((".py", ".js", ".html", ".css", ".md", ".json", ".txt", ".yml", ".yaml", ".xml", ".sh", ".bat")):
            # Extract the file path - this is a simplified approach
            path = line.strip()
            # Clean up the path (remove bullets, etc.)
            path = path.lstrip('- */').split()[0] if path.split() else ""
            if path and "." in path:  # Ensure it's likely a file
                files.append(path)
    
    return files

def prioritize_files(files: List[str], implementation_plan: str) -> List[str]:
    """
    Prioritize files based on the implementation plan.
    
    Args:
        files: List of files to prioritize
        implementation_plan: Content of the implementation plan document
        
    Returns:
        List[str]: Prioritized list of files
    """
    # This is a simple implementation that could be improved
    # Ideally, we'd parse the implementation plan to get the exact order
    
    # Move core files and configuration files to the front
    core_files = []
    config_files = []
    other_files = []
    
    for file in files:
        if "config" in file or "settings" in file:
            config_files.append(file)
        elif "core" in file or "main" in file or "__init__" in file:
            core_files.append(file)
        else:
            other_files.append(file)
    
    return core_files + config_files + other_files

def generate_structure_script(structure_content: str, output_script_path: str, orchestrator: AIOrchestrator, model_name: str) -> bool:
    """
    Generate a bash script from the project structure document that will create
    all directories and empty files.
    
    Args:
        structure_content: The content of the project structure markdown file
        output_script_path: Where to save the generated bash script
        orchestrator: The AI orchestrator instance
        model_name: The model to use for script generation
        
    Returns:
        bool: True if script was successfully generated, False otherwise
    """
    print(f"Generating structure script at {output_script_path}")
    
    prompt = f"""
    I need you to create a bash script that will set up a project structure precisely in the current directory where the script is run.
    
    CRITICAL INSTRUCTIONS ABOUT PATHS:
    1. The script will be executed directly inside the project directory (C:\\git\\project_maker\\generated_project\\)
    2. ALL files and directories must be created INSIDE this directory
    3. DO NOT create any project root directory - your current working directory IS already the project root
    4. If the structure documentation shows: "neuroca/src/main.py", just create "./src/main.py" 
    5. NEVER use absolute paths or parent directory references (like ../) in your script
    6. All paths should be relative to the current directory
    7. Only use ./ or direct subdirectory references like "src/" or "api/"
    
    The script should:
    1. Create all directories first using mkdir -p
    2. Create all empty files using touch
    3. Print progress as it creates directories and files
    
    Here's the project structure document:
    
    {structure_content}
    
    IMPORTANT REMINDER: Your script will be run FROM INSIDE the project directory. Do not try to navigate to different directories.
    """
    
    try:
        # Use the orchestrator to generate the script
        system_prompt = "You are an expert in bash scripting. Your task is to convert a project structure description into a bash script that creates all directories and files."
        script_content = orchestrator.call_llm(system_prompt, prompt, max_tokens=8000, temperature=0.2)
        
        # Add a header comment to clarify the purpose and execution directory
        script_header = """#!/bin/bash
# Project structure setup script
# This script should be run from inside the project root directory
# It will create all directories and files for the project structure

# Ensure we're creating files in the current directory
echo "Creating project structure in: $(pwd)"

"""
        script_content = script_header + script_content
        
        # Write the script to file
        with open(output_script_path, 'w') as f:
            f.write(script_content)
            
        print(f"Script generated at: {output_script_path}")
        
        # Make the script executable
        os.chmod(output_script_path, 0o755)
        
        print(f"✅ Structure script generated at {output_script_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate structure script: {str(e)}")
        return False

def execute_structure_script(script_path: str) -> bool:
    """
    Execute the generated structure script to create all directories and files.
    
    Args:
        script_path: Path to the bash script to execute
        
    Returns:
        bool: True if script executed successfully, False otherwise
    """
    print("Executing project structure script...")
    
    try:
        # Make sure PROJECT_DIR exists
        project_dir_abs = os.path.abspath(PROJECT_DIR)
        print(f"Target project directory: {project_dir_abs}")
        os.makedirs(project_dir_abs, exist_ok=True)
        
        # Create a proper app directory structure regardless of script content
        app_dir = os.path.join(project_dir_abs, "app")
        print(f"Creating main application directory: {app_dir}")
        os.makedirs(app_dir, exist_ok=True)
        
        # Standard directories to create
        standard_dirs = [
            "app/src",
            "app/src/api",
            "app/src/components",
            "app/src/config",
            "app/src/data",
            "app/src/models",
            "app/src/services",
            "app/src/utils",
            "app/tests",
            "app/docs",
        ]
        
        # Create these standard directories
        for dir_path in standard_dirs:
            # Convert to system path
            full_path = os.path.join(project_dir_abs, dir_path.replace('/', os.sep))
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")
        
        # Also process the script to extract any additional directories and files
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
                
            # Extract mkdir commands with very flexible pattern matching
            mkdir_pattern = r'mkdir\s+-p\s+[\'"`]?([^\'"`<>|;\r\n]+)[\'"`]?'
            directories = re.findall(mkdir_pattern, script_content)
            
            # Extract touch commands with very flexible pattern matching
            touch_pattern = r'touch\s+[\'"`]?([^\'"`<>|;\r\n]+)[\'"`]?'
            files = re.findall(touch_pattern, script_content)
            
            # Extract echo commands that create files
            echo_pattern = r'echo\s+.*?>\s+[\'"`]?([^\'"`<>|;\r\n]+)[\'"`]?'
            echo_files = re.findall(echo_pattern, script_content)
            files.extend(echo_files)
            
            # More aggressive pattern to find potential paths
            general_path_pattern = r'[\'"`]([a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]+)[\'"`]'
            potential_paths = re.findall(general_path_pattern, script_content)
            
            # Add directories and files from general paths
            for path in potential_paths:
                if path not in directories and path not in files:
                    if '.' in os.path.basename(path):
                        files.append(path)
                    else:
                        directories.append(path)
            
            print(f"Found {len(directories)} additional directories and {len(files)} files to create from script")
            
            # Create additional directories
            for directory in directories:
                # Skip if already a standard directory
                if f"app/{directory}" in standard_dirs or directory in standard_dirs:
                    continue
                    
                # Make sure it starts with app/
                if not directory.startswith("app/"):
                    directory = f"app/{directory}"
                    
                # Convert to system path
                full_path = os.path.join(project_dir_abs, directory.replace('/', os.sep))
                os.makedirs(full_path, exist_ok=True)
                print(f"Created additional directory: {full_path}")
            
            # Create files
            for file_path in files:
                # Make sure it starts with app/
                if not file_path.startswith("app/"):
                    file_path = f"app/{file_path}"
                    
                # Convert to system path
                full_path = os.path.join(project_dir_abs, file_path.replace('/', os.sep))
                # Ensure directory exists
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                # Create empty file
                with open(full_path, 'w') as f:
                    pass
                print(f"Created file: {full_path}")
                
        except Exception as e:
            print(f"Warning: Error processing script (continuing anyway): {e}")
        
        print("✅ Project structure created successfully")
        return True
            
    except Exception as e:
        print(f"❌ Error during structure creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def discover_all_files(directory: str) -> List[str]:
    """
    Recursively discover all files in the generated project directory that need implementation.
    
    Args:
        directory: The root directory to start discovery from
        
    Returns:
        List[str]: List of all file paths, relative to the directory
    """
    all_files = []
    
    # Define which files to include based on extensions
    source_extensions = {
        # Python and related
        '.py', '.pyi', '.pyx', '.pyd', '.pyc', '.pyw', '.ipynb',
        # Web
        '.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.json', '.vue',
        # C/C++
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
        # Java
        '.java', '.kotlin', '.kt', '.scala',
        # C#
        '.cs', '.csproj', '.sln',
        # Shell scripts
        '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd',
        # Markup and config
        '.md', '.rst', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg',
        # Other common source files
        '.go', '.rb', '.php', '.pl', '.sql', '.lua', '.r', '.swift'
    }
    
    # Files and directories to skip
    skip_files = {'setup_project_structure.sh', '.git', '__pycache__', 'node_modules', '.venv', '.env'}
    skip_dirs = {'doc', '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env', 'build', 'dist', 'target'}
    
    # Check if we're in the root directory and there's an app subdirectory
    app_dir = os.path.join(directory, "app")
    if os.path.isdir(app_dir):
        print(f"Found app directory, using it as the source directory")
        # Use app directory as the actual source directory
        search_dir = app_dir
    else:
        search_dir = directory
    
    # For debugging
    print(f"Searching for source files in {os.path.abspath(search_dir)}")
    
    for root, dirs, files in os.walk(search_dir):
        # Skip excluded directories (modify dirs in-place to prevent os.walk from descending into them)
        for skip_dir in skip_dirs:
            if skip_dir in dirs:
                dirs.remove(skip_dir)
        
        # Skip doc directory specifically 
        if "doc" in os.path.relpath(root, search_dir).split(os.sep):
            continue
            
        for file in files:
            # Skip specific files we don't want to implement
            if file in skip_files:
                continue
                
            # Get path relative to the directory
            file_path = os.path.join(root, file)
            
            # First make relative to search_dir
            rel_to_search = os.path.relpath(file_path, search_dir)
            
            # Then if search_dir is app, adjust the path to make it relative to original directory
            if search_dir == app_dir:
                rel_path = os.path.join("app", rel_to_search)
            else:
                rel_path = rel_to_search
            
            # Make sure the path uses forward slashes for consistency
            rel_path = rel_path.replace(os.sep, '/')
            
            # Check if the file has a source code extension or has no extension but is not too large
            _, ext = os.path.splitext(file)
            
            # Debug info
            # print(f"Found file: {rel_path} with extension {ext}")
            
            if ext.lower() in source_extensions:
                # Include source code files
                all_files.append(rel_path)
            elif not ext and os.path.getsize(file_path) < 1048576:  # 1MB limit for extensionless files
                # Include extensionless files like "Makefile", "Dockerfile", etc.
                all_files.append(rel_path)
            elif os.path.getsize(file_path) == 0:
                # Include any empty file regardless of extension
                all_files.append(rel_path)
    
    # Debug info
    print(f"Found {len(all_files)} source files that need implementation")
    if len(all_files) < 10:  # Print all files if there are fewer than 10
        for file in all_files:
            print(f" - {file}")
    else:  # Otherwise print first 5 and last 5
        for file in all_files[:5]:
            print(f" - {file}")
        print(f" - ... and {len(all_files) - 10} more files ...")
        for file in all_files[-5:]:
            print(f" - {file}")
    
    return all_files

def run_project_builder(vision: str, model_name: str, start_step: int = 1, start_substep: str = None, run_syntax_check_only: bool = False, track_dependencies: bool = False):
    """
    Run the project builder process.
    
    Args:
        vision: The project vision text
        model_name: The LLM model name to use
        start_step: Step number to start from (1-based)
        start_substep: Substep ID to start from (optional)
        run_syntax_check_only: Only run syntax checking on existing project
        track_dependencies: Whether to enable dependency tracking
    """
    
    # Just run syntax checking if specified
    if run_syntax_check_only:
        print("Running syntax check on existing project...")
        syntax_errors = run_syntax_check(PROJECT_DIR)
        return
    
    # Continue with normal project building
    print("\n=== Project Builder ===")
    print(f"Using model: {model_name}")
    
    # Initialize the orchestrator
    orchestrator = AIOrchestrator(model_name)
    
    # Create a map of all files created
    file_map = {}
    
    # Make sure required directories exist
    os.makedirs(os.path.join(PROJECT_DIR, "doc"), exist_ok=True)
    
    # Store outputs from each step
    step_outputs = {}
    # Vision is always step 0
    step_outputs[0] = vision
    
    # Store outputs from each sub-step
    sub_step_outputs = {}
    
    # Flag to track if we're using the script-generated file structure
    using_generated_structure = False
    
    # Initialize dependency tracking if requested
    dependency_resolver = None
    if track_dependencies:
        try:
            print("Initializing dependency tracking system...")
            # Check if dependency tracking is available
            try:
                from dependency_integration import initialize_dependency_tracking, restore_original_functions, perform_final_dependency_check
                DEPENDENCY_TRACKING_AVAILABLE = True
                print("Dependency tracking module found.")
            except ImportError:
                print("Dependency tracking module not found. Will not track dependencies.")
                DEPENDENCY_TRACKING_AVAILABLE = False
                track_dependencies = False
            
            # Initialize dependency tracking if available
            if DEPENDENCY_TRACKING_AVAILABLE:
                print("Initializing dependency tracking system...")
                # Initialize dependency tracking with the current module and project directory
                import sys
                try:
                    dependency_resolver = initialize_dependency_tracking(sys.modules[__name__], PROJECT_DIR)
                    print("✅ Dependency tracking initialized")
                except Exception as e:
                    print(f"❌ Error initializing dependency tracking: {e}")
                    DEPENDENCY_TRACKING_AVAILABLE = False
                    track_dependencies = False
        except Exception as e:
            print(f"❌ Error with dependency tracking: {e}")
            track_dependencies = False
    
    # Read the structure document if it exists (from previous runs)
    structure_content = None
    structure_doc_path = os.path.join(PROJECT_DIR, "doc", "STEP1_SUBSTEP_1C.md")
    if os.path.exists(structure_doc_path):
        with open(structure_doc_path, 'r', encoding='utf-8') as f:
            structure_content = f.read()
    
    # Execute steps in sequence
    for i, step in enumerate(BUILDER_STEPS, start=1):
        # Skip steps before the start_step
        if i < start_step:
            print(f"Skipping step {i} ({step['phase_name']})...")
            continue
        
        print(f"\n=== {step['phase_name']} ===")
        
        # Special handling for steps 4, 5, and 6 which don't have substeps
        if i >= 4:
            # Execute these steps directly
            if i == 4:  # Implementation Plan
                # Generate implementation plan
                print("Generating implementation plan...")
                
                # Make sure we have the structure document
                if not structure_content:
                    print("Error: Structure document (STEP1_SUBSTEP_1C.md) not found. Cannot generate implementation plan.")
                    return
                
                # Generate the bash script
                script_path = os.path.join(PROJECT_DIR, "setup_project_structure.sh")
                if generate_structure_script(structure_content, script_path, orchestrator, model_name):
                    print("✅ Implementation plan and structure script generated")
                    step_outputs[i] = f"Generated setup_project_structure.sh script based on project structure"
                
            elif i == 5:  # File Implementation
                print("\n=== Setting up Project Structure ===")
                
                # Execute the structure script
                script_path = os.path.join(PROJECT_DIR, "setup_project_structure.sh")
                if not os.path.exists(script_path):
                    print("Error: setup_project_structure.sh not found. Please run step 4 first.")
                    return
                
                if execute_structure_script(script_path):
                    # Discover all files that need implementation
                    all_files = discover_all_files(PROJECT_DIR)
                    print(f"Discovered {len(all_files)} files that need implementation")
                    
                    # Replace the normal file extraction with our discovered files
                    implementation_files = all_files
                    
                    # We can still use the prioritization logic if desired
                    implementation_plan = step_outputs.get(4, "")
                    if implementation_plan and implementation_plan.strip():
                        implementation_files = prioritize_files(all_files, implementation_plan)
                    
                    print(f"\n=== File Implementation ===")
                    print(f"Implementing {len(implementation_files)} files in priority order...")
                    
                    # Create a progress tracking file to support crash recovery
                    progress_file = os.path.join(PROJECT_DIR, "doc", "implementation_progress.json")
                    completed_files = []
                    
                    # Load previously completed files if the progress file exists
                    if os.path.exists(progress_file):
                        try:
                            with open(progress_file, 'r') as f:
                                completed_files = json.load(f)
                                print(f"Found {len(completed_files)} previously completed files.")
                        except Exception as e:
                            print(f"Error loading progress file: {e}")
                            completed_files = []
                    
                    # Filter out already completed files
                    files_to_implement = [f for f in implementation_files if f not in completed_files]
                    if len(files_to_implement) < len(implementation_files):
                        print(f"Skipping {len(implementation_files) - len(files_to_implement)} already implemented files.")
                    
                    for idx, file_path in enumerate(files_to_implement):
                        print(f"\nImplementing file {idx+1}/{len(files_to_implement)}: {file_path}")
                        # Use our implementation method and pass the dependency_resolver if available
                        success = implement_single_file(
                            file_path, 
                            structure_content, 
                            step_outputs, 
                            orchestrator, 
                            model_name,
                            file_map,
                            dependency_resolver if track_dependencies else None
                        )
                        
                        if success:
                            # Add to completed files and save progress after each successful implementation
                            completed_files.append(file_path)
                            try:
                                # Ensure the doc directory exists
                                os.makedirs(os.path.join(PROJECT_DIR, "doc"), exist_ok=True)
                                # Save progress
                                with open(progress_file, 'w') as f:
                                    json.dump(completed_files, f)
                            except Exception as e:
                                print(f"Warning: Could not save progress: {e}")
                        
                        print(f"Progress: {len(completed_files)}/{len(implementation_files)} files completed")
                    
                    step_outputs[i] = f"Implemented {len(completed_files)} files"
                    using_generated_structure = True
                
            elif i == 6:  # Assembly and Usage Guide
                print("\n=== Generating Assembly and Usage Guide ===")
                
                # Run syntax check on implemented files if enabled
                if ENABLE_SYNTAX_CHECKING:
                    syntax_errors = run_syntax_check(PROJECT_DIR)
                    if syntax_errors:
                        print("\nWarning: Syntax errors were found in some files.")
                        user_choice = input("Do you want to continue with documentation generation? (y/n): ")
                        if user_choice.lower() != 'y':
                            print("Documentation generation aborted. Please fix the syntax errors.")
                            continue
                
                # Create the README.md and other documentation
                readme_prompt = f"""
                You are tasked with creating a comprehensive README.md file for the project based on:
                
                1. Project Vision:
                {step_outputs.get(0, '(No vision provided)')}
                
                2. Project Structure:
                {structure_content[:2000] if structure_content else '(No structure provided)'}
                
                3. Implementation details:
                {step_outputs.get(5, '(No implementation details provided)')}
                
                The README should include:
                - Project overview and purpose
                - Features and capabilities
                - Installation instructions
                - Usage examples
                - API documentation (if applicable)
                - Contributing guidelines
                - License information
                
                Output your README in `=== File: README.md ===`
                """
                
                system_prompt = "You are an expert technical writer creating documentation for a software project."
                readme_response = orchestrator.call_llm(system_prompt, readme_prompt, max_tokens=8000, temperature=0.2)
                
                if readme_response and not readme_response.startswith("ERROR"):
                    # Process and store the README file
                    parse_ai_response_and_apply(readme_response, file_map)
                    
                    # Write all files
                    for rel_path, pf in file_map.items():
                        write_project_file(PROJECT_DIR, pf)
                    
                    print("✅ README.md and documentation generated")
                    step_outputs[i] = "Generated README.md and documentation"
            
            continue
        
        # Normal handling for steps with substeps (steps 1-3)
        sub_steps = step.get('sub_steps', [])
        
        if not sub_steps:
            print(f"Warning: Step {i} ({step['phase_name']}) does not have sub_steps defined. Skipping.")
            continue
        
        # Execute each sub-step in order
        for j, sub_step in enumerate(sub_steps):
            sub_step_id = ALPHA[j]
            sub_step_key = f"{i}{sub_step_id}"
            
            # Skip sub-steps before the start_substep if this is the start_step
            if i == start_step and start_substep and sub_step_id < start_substep:
                print(f"Skipping sub-step {i}{sub_step_id} ({sub_step.name})...")
                continue
                
            print(f"\n--- Step {i}{sub_step_id}: {sub_step.name} ---")
            
            # Execute the sub-step
            success = execute_substep(
                orchestrator=orchestrator,
                step_info=step,
                step_index=i,
                sub_step_index=j,
                sub_step=sub_step,
                file_map=file_map,
                step_outputs=step_outputs,
                sub_step_outputs=sub_step_outputs
            )
            
            if not success:
                print(f"Failed to execute sub-step {i}{sub_step_id}. Aborting.")
                return
                
            print("===========================\n")
        
        # Save the structure document after Step 1 (in case we just ran Step 1)
        if i == 1 and not structure_content:
            structure_doc_path = os.path.join(PROJECT_DIR, "doc", "STEP1_SUBSTEP_1C.md")
            if os.path.exists(structure_doc_path):
                with open(structure_doc_path, 'r', encoding='utf-8') as f:
                    structure_content = f.read()
    
    # Store all step outputs in a summary file
    output_summary = "# Project Generation Summary\n\n"
    for i, output in step_outputs.items():
        if i == 0:
            output_summary += f"## Vision\n{output}\n\n"
        else:
            step_name = BUILDER_STEPS[i-1].get('phase_name', f"Step {i}") if i-1 < len(BUILDER_STEPS) else f"Step {i}"
            output_summary += f"## {step_name}\n{output}\n\n"
    
    with open(os.path.join(PROJECT_DIR, "doc", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(output_summary)
    
    # Implement files if we haven't already and we're not using the generated structure
    if not using_generated_structure and start_step >= 5:
        print(f"\n=== File Implementation ===")
        
        # Make sure we have the structure document
        if not structure_content:
            print("Error: Structure document (STEP1_SUBSTEP_1C.md) not found. Cannot implement files.")
            return
        
        # Generate and execute the structure script
        script_path = os.path.join(PROJECT_DIR, "setup_project_structure.sh")
        if generate_structure_script(structure_content, script_path, orchestrator, model_name):
            if execute_structure_script(script_path):
                # Discover all files that need implementation
                all_files = discover_all_files(PROJECT_DIR)
                print(f"Discovered {len(all_files)} files that need implementation")
                
                # Set the implementation files
                implementation_files = all_files
                
                # Try to prioritize if we have an implementation plan
                implementation_plan = step_outputs.get(4, "")
                if implementation_plan and implementation_plan.strip():
                    implementation_files = prioritize_files(all_files, implementation_plan)
                
                print(f"Implementing {len(implementation_files)} files in priority order...")
                
                # Create a progress tracking file to support crash recovery
                progress_file = os.path.join(PROJECT_DIR, "doc", "implementation_progress.json")
                completed_files = []
                
                # Load previously completed files if the progress file exists
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, 'r') as f:
                            completed_files = json.load(f)
                            print(f"Found {len(completed_files)} previously completed files.")
                    except Exception as e:
                        print(f"Error loading progress file: {e}")
                        completed_files = []
                
                # Filter out already completed files
                files_to_implement = [f for f in implementation_files if f not in completed_files]
                if len(files_to_implement) < len(implementation_files):
                    print(f"Skipping {len(implementation_files) - len(files_to_implement)} already implemented files.")
                
                for idx, file_path in enumerate(files_to_implement):
                    print(f"\nImplementing file {idx+1}/{len(files_to_implement)}: {file_path}")
                    # Use our implementation method and pass the dependency_resolver if available
                    success = implement_single_file(
                        file_path, 
                        structure_content, 
                        step_outputs, 
                        orchestrator, 
                        model_name,
                        file_map,
                        dependency_resolver if track_dependencies else None
                    )
                    
                    if success:
                        # Add to completed files and save progress after each successful implementation
                        completed_files.append(file_path)
                        try:
                            # Ensure the doc directory exists
                            os.makedirs(os.path.join(PROJECT_DIR, "doc"), exist_ok=True)
                            # Save progress
                            with open(progress_file, 'w') as f:
                                json.dump(completed_files, f)
                        except Exception as e:
                            print(f"Warning: Could not save progress: {e}")
                    
                    print(f"Progress: {len(completed_files)}/{len(implementation_files)} files completed")
                
                step_outputs[i] = f"Implemented {len(completed_files)} files"
                using_generated_structure = True
    
    # Store the overall implementation output
    output_summary = f"# Project Implementation Summary\n\n"
    output_summary += f"## Files Implemented\n\n"
    for file_path in sorted(file_map.keys()):
        output_summary += f"- {file_path}\n"
    
    with open(os.path.join(PROJECT_DIR, "doc", "IMPLEMENTATION.md"), "w", encoding="utf-8") as f:
        f.write(output_summary)
    
    print(f"\n=== Project Build Complete ===")
    print(f"Your project has been generated in the '{PROJECT_DIR}' directory.")
    print("You can find documentation in the 'doc' subdirectory.")
    
    # Perform final dependency check if tracking was enabled
    if track_dependencies and 'DEPENDENCY_TRACKING_AVAILABLE' in locals() and DEPENDENCY_TRACKING_AVAILABLE:
        try:
            print("\n=== Final Dependency Check ===")
            dependency_issues = perform_final_dependency_check(PROJECT_DIR, orchestrator, model_name)
            if not dependency_issues:
                print("✅ All dependencies successfully resolved!")
            else:
                print("⚠️ Some dependencies could not be fully resolved.")
        except Exception as e:
            print(f"⚠️ Error during final dependency check: {e}")
    
    # Clean up dependency tracking if it was initialized
    if track_dependencies and 'DEPENDENCY_TRACKING_AVAILABLE' in locals() and DEPENDENCY_TRACKING_AVAILABLE:
        try:
            restore_original_functions(sys.modules[__name__])
            print("✅ Dependency tracking resources cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up dependency tracking: {e}")

def implement_single_file(file_path: str, structure_content: str, step_outputs: Dict[int, str], 
                          orchestrator: AIOrchestrator, model_name: str, file_map: Dict[str, ProjectFile],
                          dependency_resolver=None) -> bool:
    """
    Implement a single file by prompting the LLM with specific context for that file.
    
    Args:
        file_path: The relative path of the file to implement
        structure_content: The project structure document content
        step_outputs: Outputs from previous steps for context
        orchestrator: The AI orchestrator instance
        model_name: The LLM model to use
        file_map: Map of already implemented files
        dependency_resolver: Dependency resolver instance (optional)
        
    Returns:
        bool: True if implementation was successful, False otherwise
    """
    print(f"\nImplementing file: {file_path}")
    
    # Check if the file already exists from a previous run - support for crash recovery
    full_file_path = os.path.join(PROJECT_DIR, file_path)
    if os.path.exists(full_file_path):
        file_size = os.path.getsize(full_file_path)
        if file_size > 0:  # Only consider non-empty files as successfully implemented
            print(f"File {file_path} already exists ({file_size} bytes). Skipping implementation.")
            
            # We need to add it to the file_map to ensure consistency
            with open(full_file_path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    file_map[file_path] = ProjectFile(file_path, content)
                    return True
                except Exception as e:
                    print(f"Error reading existing file {file_path}: {e}")
                    print("Will re-implement this file.")
                    # Continue with implementation if reading fails
    
    # Build focused context for this file
    file_prompt = f"""# File Implementation: {file_path}

IMPORTANT: You are implementing a production-ready source file that must be complete,
robust, and maintainable. Minimal or superficial implementations are not acceptable.

CRITICAL: DO NOT WRITE DEMONSTRATION CODE. Write REAL, FUNCTIONAL code that would
actually be used in a production environment. Your code will be saved directly to a file
and is expected to work without modification.

## Project Context
{step_outputs.get(0, '(No vision provided)')}

## Project Structure
The file is part of the following project structure:
```
{structure_content[:2000]}  # Include first 2000 chars of the structure
```

## Relevant Architecture & Design
{step_outputs.get(2, '(No architecture)')[:1000]}  # Only first 1000 chars of architecture
{step_outputs.get(3, '(No structure)')[:1000]}     # Only first 1000 chars of structure

## Implementation Task
Your task is to implement: {file_path}

## Implementation Requirements
1. Code Quality:
   - Production-ready, professional code
   - Comprehensive error handling
   - Complete input validation
   - Proper logging
   - Thorough documentation
   - Clear code organization

2. Technical Requirements:
   - Follow all architectural decisions
   - Implement complete functionality
   - Include ALL necessary imports
   - Handle ALL edge cases
   - Include proper error messages
   - Add debug logging where appropriate

3. Documentation Requirements:
   - File-level documentation
   - Function/class documentation
   - Important code block documentation
   - Usage examples in comments
   - Edge case documentation
   - Error handling documentation

4. Testing Considerations:
   - Make code testable
   - Document test scenarios
   - Handle boundary conditions
   - Consider error scenarios

5. Security & Robustness:
   - Implement security best practices
   - Handle resource cleanup
   - Prevent memory leaks
   - Secure error handling
   - Input sanitization

Remember: This code will be used in production. It must be complete, robust, and maintainable.
Avoid shortcuts or minimal implementations. Write code that you would confidently deploy to production.

Output your implementation in `=== File: {file_path} ===`"""
    
    try:
        # Call the LLM to implement this file
        system_prompt = """You are an expert software engineer implementing a critical file in a complex project.
CRITICAL INSTRUCTIONS:
1. Write COMPLETE, FUNCTIONAL code that can be used in a production environment
2. Do NOT write pseudo-code or example code
3. Do NOT include comments like "This is just a demonstration" or "This is a simplified version"
4. Implement FULL functionality according to requirements
5. Include ALL necessary imports, constants, error handling, and logic
6. Your code will be directly saved to a file and is expected to work without modifications"""
        
        # Drastically increase the max_tokens to ensure complete file generation
        # This doesn't affect input token limits, only allows more output
        max_output_tokens = 64000  # Maximum allowed for Claude models
        print(f"Setting max output tokens to {max_output_tokens} for file implementation")
        
        # Use temperature 0.0 for file implementation for deterministic code generation
        temperature = 0.0
        print(f"Using temperature {temperature} for precise code generation")
        
        ai_response = orchestrator.call_llm(system_prompt, file_prompt, max_tokens=max_output_tokens, temperature=temperature)
        
        if not ai_response or ai_response.startswith("ERROR"):
            print(f"Error implementing {file_path}: {ai_response}")
            return False
        
        # Process and store the file
        parse_ai_response_and_apply(ai_response, file_map)
        
        # Write all files after each implementation
        for rel_path, pf in file_map.items():
            write_project_file(PROJECT_DIR, pf)
        
        print(f"✅ Successfully implemented {file_path}")
        result = True
    except Exception as e:
        print(f"❌ Failed to implement {file_path}: {str(e)}")
        result = False
    
        # Check dependencies if dependency resolver is provided
    if result and dependency_resolver and file_path in file_map:
        try:
            # Use the resolver to check the file
            deps = dependency_resolver.check_file(file_path, file_map[file_path].content)
            if deps:
                print(f"⚠️ Found {len(deps)} dependencies to resolve in {file_path}")
            else:
                print(f"✅ No dependency issues found in {file_path}")
        except Exception as e:
            print(f"Error checking dependencies: {e}")
    
    return result

def run_syntax_check(directory: str) -> Dict[str, List[str]]:
    """
    Run syntax linters on the implemented project files based on their extensions.
    
    Args:
        directory: The root directory of the project
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping file paths to lists of syntax errors
    """
    print("\n=== Running Syntax Check ===")
    errors = {}
    
    # Get all files recursively
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip hidden files and directories
            if file.startswith('.') or '/.' in root or '\\.' in root:
                continue
            # Skip the doc directory
            if 'doc' in root.split(os.path.sep):
                continue
            
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    print(f"Found {len(all_files)} files to check for syntax errors")
    
    # Check Python files
    python_files = [f for f in all_files if f.endswith('.py')]
    if python_files:
        print(f"Checking {len(python_files)} Python files...")
        for py_file in python_files:
            rel_path = os.path.relpath(py_file, directory)
            try:
                # Use Python's built-in compile function to check syntax
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    compile(content, py_file, 'exec')
                except SyntaxError as e:
                    if rel_path not in errors:
                        errors[rel_path] = []
                    errors[rel_path].append(f"Line {e.lineno}: {e.msg}")
            except Exception as e:
                if rel_path not in errors:
                    errors[rel_path] = []
                errors[rel_path].append(f"Error reading file: {str(e)}")
    
    # Check JavaScript files
    js_files = [f for f in all_files if f.endswith('.js')]
    if js_files:
        print(f"Checking {len(js_files)} JavaScript files...")
        for js_file in js_files:
            rel_path = os.path.relpath(js_file, directory)
            try:
                # Use Node.js to check syntax if available
                result = subprocess.run(
                    ['node', '--check', js_file], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode != 0:
                    if rel_path not in errors:
                        errors[rel_path] = []
                    errors[rel_path].extend(result.stderr.splitlines())
            except Exception as e:
                # Node.js might not be available, skip with a warning
                print(f"Warning: Could not check JavaScript syntax for {rel_path}: {str(e)}")
    
    # Check JSON files
    json_files = [f for f in all_files if f.endswith('.json')]
    if json_files:
        print(f"Checking {len(json_files)} JSON files...")
        for json_file in json_files:
            rel_path = os.path.relpath(json_file, directory)
            try:
                # Use Python's json module to validate JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                if rel_path not in errors:
                    errors[rel_path] = []
                errors[rel_path].append(f"Line {e.lineno}: {e.msg}")
            except Exception as e:
                if rel_path not in errors:
                    errors[rel_path] = []
                errors[rel_path].append(f"Error reading file: {str(e)}")
    
    # Check HTML files (basic validation)
    html_files = [f for f in all_files if f.endswith(('.html', '.htm'))]
    if html_files:
        print(f"Checking {len(html_files)} HTML files...")
        try:
            # Try to import html.parser for basic validation
            from html.parser import HTMLParser
            
            class ValidatingHTMLParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.errors = []
                
                def error(self, message):
                    self.errors.append(message)
                    
            for html_file in html_files:
                rel_path = os.path.relpath(html_file, directory)
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    parser = ValidatingHTMLParser()
                    parser.feed(content)
                    
                    if parser.errors:
                        if rel_path not in errors:
                            errors[rel_path] = []
                        errors[rel_path].extend(parser.errors)
                except Exception as e:
                    if rel_path not in errors:
                        errors[rel_path] = []
                    errors[rel_path].append(f"Error checking HTML: {str(e)}")
        except ImportError:
            print("Warning: HTML parser not available, skipping HTML validation")
    
    # Check CSS files (basic validation)
    css_files = [f for f in all_files if f.endswith('.css')]
    if css_files:
        print(f"Checking {len(css_files)} CSS files...")
        for css_file in css_files:
            rel_path = os.path.relpath(css_file, directory)
            try:
                # Basic CSS syntax check (opening/closing braces)
                with open(css_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple check for balanced braces
                if content.count('{') != content.count('}'):
                    if rel_path not in errors:
                        errors[rel_path] = []
                    errors[rel_path].append("Unbalanced braces in CSS file")
            except Exception as e:
                if rel_path not in errors:
                    errors[rel_path] = []
                errors[rel_path].append(f"Error reading file: {str(e)}")
    
    # Print summary
    if errors:
        print(f"\n❌ Found syntax errors in {len(errors)} files:")
        for file_path, file_errors in errors.items():
            print(f"\n{file_path}:")
            for error in file_errors:
                print(f"  - {error}")
    else:
        print("\n✅ No syntax errors found in the implemented files")
    
    return errors 