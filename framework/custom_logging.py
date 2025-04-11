import logging
import sys
import os

def setup_logging(log_file='lavka_recsys.log', file_level=logging.DEBUG, console_level=logging.INFO):
    """Configure logging to write to both file and console."""
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
    """
    if name:
        # This creates a child logger that inherits handlers from the root logger
        return logging.getLogger(f'lavka_recsys.{name}')
    return logging.getLogger('lavka_recsys')

# Create and configure logger
logger = setup_logging()