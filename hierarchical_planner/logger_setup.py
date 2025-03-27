"""
Logging setup module for the Hierarchical Planner.

Configures Python's standard logging based on application configuration.
Supports console and rotating file logging.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]):
    """
    Configures logging based on the provided configuration dictionary.

    Sets up logging level, format, console handler, and optional file handler.

    Args:
        config: The loaded configuration dictionary, expected to contain a 'logging' section.
    """
    log_config = config.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()
    log_file_path = log_config.get('log_file') # Path is already absolute from config_loader
    log_to_console = log_config.get('log_to_console', True)

    # Get the numeric logging level
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers (e.g., from basicConfig in main) to avoid duplicates
    # Important if this function might be called after basicConfig
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(log_level) # Console logs at the configured level
        root_logger.addHandler(console_handler)
        logging.info(f"Console logging enabled at level {log_level_str}.") # Use logging here

    # Configure File Handler (Optional)
    if log_file_path:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                logging.info(f"Created log directory: {log_dir}")

            # Use RotatingFileHandler for log rotation
            # Rotate logs when they reach 5MB, keep 3 backup logs
            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setFormatter(log_format)
            file_handler.setLevel(log_level) # File logs at the configured level
            root_logger.addHandler(file_handler)
            logging.info(f"File logging enabled at level {log_level_str} to {log_file_path}.")
        except Exception as e:
            # Log error in setting up file handler, but don't crash the app
            logging.error(f"Failed to configure file logging to {log_file_path}: {e}", exc_info=True)
    else:
        logging.info("File logging is disabled.")

    # Example: Quieten overly verbose libraries if needed
    # logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)

    logging.info("Logging setup complete.")
