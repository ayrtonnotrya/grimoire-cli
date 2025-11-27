import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name="grimoire"):
    """Sets up a logger that writes to a daily log file in the logs/ directory."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate filename based on current date
    current_date = datetime.now().strftime("%d-%m-%Y")
    log_file = log_dir / f"{current_date}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Check if handler already exists to avoid duplicate logs
    if not logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False
        
    return logger

# Global logger instance
logger = setup_logger()
