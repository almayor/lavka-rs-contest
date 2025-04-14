import logging
import sys
import os
from typing import Optional

from .config import Config

def setup_logging(config: Optional[Config] = None):
    """
    Configure logging to write to both file and console based on config.
    
    Parameters:
        config: Configuration object. If provided, logging settings will be read from config.
    """
    # Default values
    log_file = 'lavka_recsys.log'
    file_level = logging.DEBUG
    console_level = logging.INFO
    
    # Override with config values if available
    if config:
        log_config = config.get('logging', {})
        
        # Get log file path from config
        config_log_file = log_config.get('file')
        if config_log_file:
            log_file = config_log_file
        
        # Get log level from config
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }  
        config_console_level = log_config.get('console_level')
        if config_console_level:
            console_level = level_map.get(config_console_level.upper(), logging.INFO)
        config_file_level = log_config.get('file_level')
        if config_file_level:
            file_level = level_map.get(config_file_level.upper(), logging.DEBUG)
    
    # Create logger
    root_logger = logging.getLogger('lavka_recsys')
    root_logger.setLevel(min(file_level, console_level))
    root_logger.handlers = []  # Clear any existing handlers
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name=None):
    """
    Get a logger with the appropriate name that inherits settings from the root logger.
    If name is None, return the root logger.
    """
    if name:
        # This creates a child logger that inherits handlers from the root logger
        return logging.getLogger(f'lavka_recsys.{name}')
    return logging.getLogger('lavka_recsys')
