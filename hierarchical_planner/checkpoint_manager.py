"""
Checkpoint manager for the Hierarchical Planner.

Provides functionality to save and load checkpoints of the planning process,
allowing the program to resume from where it left off after an interruption.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple

# Configure logger for this module
logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages checkpoints for the hierarchical planning process.
    
    Provides methods to save the current state of the planning process
    and load it back to resume from where it left off.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints, relative to the 
                            hierarchical_planner directory
        """
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = os.path.join(module_dir, checkpoint_dir)
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
                logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to create checkpoint directory: {e}")
                # Continue execution even if we can't create the directory
        
        # Standard filenames for each checkpoint type
        self.gen_checkpoint_filename = "generation_checkpoint.json"
        self.qa_checkpoint_filename = "qa_checkpoint.json"
    
    def save_generation_checkpoint(self, 
                                  goal: str, 
                                  current_state: Dict[str, Any], 
                                  last_processed_phase: Optional[str] = None, 
                                  last_processed_task: Optional[str] = None) -> str:
        """
        Save a checkpoint of the current generation progress.
        
        Args:
            goal: The high-level goal being processed
            current_state: The current reasoning tree with all generated content so far
            last_processed_phase: The last successfully processed phase, if any
            last_processed_task: The last successfully processed task within the phase, if any
            
        Returns:
            The path to the saved checkpoint file
        """
        # Create checkpoint data structure
        checkpoint_data = {
            "timestamp": time.time(),
            "goal": goal,
            "reasoning_tree": current_state,
            "last_processed_phase": last_processed_phase,
            "last_processed_task": last_processed_task
        }
        
        # Use a consistent filename based on the goal's hash
        safe_goal = "".join([c if c.isalnum() else "_" for c in goal[:20]])
        filename = f"gen_{safe_goal}.checkpoint.json"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save the checkpoint
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Updated generation checkpoint at {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""
    
    def save_qa_checkpoint(self,
                          input_path: str,
                          output_path: str, 
                          validated_data: Dict[str, Any],
                          last_phase: Optional[str] = None,
                          last_task: Optional[str] = None,
                          last_step_index: int = -1) -> str:
        """
        Save a checkpoint of the QA validation progress.
        
        Args:
            input_path: Path to the input plan file
            output_path: Path to the output validated plan file
            validated_data: Current state of the validation process
            last_phase: The last phase being processed
            last_task: The last task being processed
            last_step_index: Index of the last step processed, -1 if no steps processed yet
            
        Returns:
            The path to the saved checkpoint file
        """
        # Create checkpoint data structure
        checkpoint_data = {
            "timestamp": time.time(),
            "input_path": input_path,
            "output_path": output_path,
            "validated_data": validated_data,
            "last_phase": last_phase,
            "last_task": last_task,
            "last_step_index": last_step_index
        }
        
        # Create a filename based on the input file path
        input_file_base = os.path.basename(input_path)
        safe_input = "".join([c if c.isalnum() else "_" for c in input_file_base])
        filename = f"qa_{safe_input}.checkpoint.json"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save the checkpoint
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Updated QA checkpoint at {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save QA checkpoint: {e}")
            return ""
    
    def find_latest_generation_checkpoint(self, goal: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Find the latest generation checkpoint file, optionally filtered by goal.
        
        Args:
            goal: If provided, only consider checkpoints for this goal
            
        Returns:
            A tuple containing:
            - The loaded checkpoint data, or None if no valid checkpoint found
            - The path to the checkpoint file, or empty string if no valid checkpoint found
        """
        try:
            # If we have a goal, try to find a checkpoint specific to that goal first
            if goal:
                safe_goal = "".join([c if c.isalnum() else "_" for c in goal[:20]])
                specific_filename = f"gen_{safe_goal}.checkpoint.json"
                specific_path = os.path.join(self.checkpoint_dir, specific_filename)
                
                if os.path.exists(specific_path):
                    try:
                        with open(specific_path, 'r', encoding='utf-8') as f:
                            checkpoint_data = json.load(f)
                        logger.info(f"Found valid checkpoint for goal at {specific_path}")
                        return checkpoint_data, specific_path
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint {specific_path}: {e}")
            
            # List all checkpoint files as a fallback
            checkpoint_files = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith("gen_") and filename.endswith(".checkpoint.json"):
                    checkpoint_files.append(filename)
            
            if not checkpoint_files:
                logger.info("No generation checkpoints found")
                return None, ""
                
            # Sort by timestamp (newest first based on file modification time)
            checkpoint_files_with_time = [(f, os.path.getmtime(os.path.join(self.checkpoint_dir, f))) 
                                          for f in checkpoint_files]
            checkpoint_files_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Try each checkpoint until we find a valid one matching the goal
            for filename, _ in checkpoint_files_with_time:
                file_path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    # If goal is provided, check if this checkpoint matches
                    if goal is not None and checkpoint_data.get("goal") != goal:
                        continue
                        
                    logger.info(f"Found valid checkpoint at {file_path}")
                    return checkpoint_data, file_path
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {file_path}: {e}")
            
            logger.info(f"No valid generation checkpoints found for goal: {goal}")
            return None, ""
        except Exception as e:
            logger.error(f"Error finding checkpoints: {e}")
            return None, ""
    
    def find_latest_qa_checkpoint(self, input_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Find the latest QA checkpoint file.
        
        Args:
            input_path: If provided, only consider checkpoints for this input file
            
        Returns:
            A tuple containing:
            - The loaded checkpoint data, or None if no valid checkpoint found
            - The path to the checkpoint file, or empty string if no valid checkpoint found
        """
        try:
            # If we have an input path, try to find a checkpoint specific to that file first
            if input_path:
                input_file_base = os.path.basename(input_path)
                safe_input = "".join([c if c.isalnum() else "_" for c in input_file_base])
                specific_filename = f"qa_{safe_input}.checkpoint.json"
                specific_path = os.path.join(self.checkpoint_dir, specific_filename)
                
                if os.path.exists(specific_path):
                    try:
                        with open(specific_path, 'r', encoding='utf-8') as f:
                            checkpoint_data = json.load(f)
                        logger.info(f"Found valid checkpoint for input at {specific_path}")
                        return checkpoint_data, specific_path
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint {specific_path}: {e}")
            
            # List all checkpoint files as a fallback
            checkpoint_files = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith("qa_") and filename.endswith(".checkpoint.json"):
                    checkpoint_files.append(filename)
                    
            if not checkpoint_files:
                logger.info("No QA checkpoints found")
                return None, ""
                
            # Sort by timestamp (newest first based on file modification time)
            checkpoint_files_with_time = [(f, os.path.getmtime(os.path.join(self.checkpoint_dir, f))) 
                                          for f in checkpoint_files]
            checkpoint_files_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Try each checkpoint until we find a valid one
            for filename, _ in checkpoint_files_with_time:
                file_path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    # If input_path is provided, check if this checkpoint matches
                    if input_path is not None and checkpoint_data.get("input_path") != input_path:
                        continue
                        
                    logger.info(f"Found valid QA checkpoint at {file_path}")
                    return checkpoint_data, file_path
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {file_path}: {e}")
            
            logger.info(f"No valid QA checkpoints found for input: {input_path}")
            return None, ""
        except Exception as e:
            logger.error(f"Error finding QA checkpoints: {e}")
            return None, ""
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a checkpoint file after successful completion.
        
        Args:
            checkpoint_path: Path to the checkpoint file to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint file: {checkpoint_path}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")
            return False