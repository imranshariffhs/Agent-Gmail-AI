import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name='agent_ai', log_dir='logs'):
    """
    Set up a logger with both file and console handlers.
    The file handler uses RotatingFileHandler with a max size of 1GB and TimedRotatingFileHandler for daily rotation.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with size rotation (1GB)
    size_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / f'{name}.log',
        maxBytes=1024 * 1024 * 1024,  # 1GB
        backupCount=10,
        encoding='utf-8'
    )
    size_handler.setFormatter(file_formatter)
    size_handler.setLevel(logging.INFO)
    
    # File handler with time rotation (daily)
    time_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path / f'{name}_daily.log',
        when='midnight',
        interval=1,
        backupCount=30,  # Keep logs for 30 days
        encoding='utf-8'
    )
    time_handler.setFormatter(file_formatter)
    time_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(size_handler)
    logger.addHandler(time_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()

def get_logger(name='agent_ai'):
    """
    Get a logger instance. If it doesn't exist, create a new one.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name) 