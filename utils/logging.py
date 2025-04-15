import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import bittensor as bt
import colorlog

def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = None,
    level: int = logging.INFO,
    log_rotation_size: int = 10 * 1024 * 1024,  # 10MB
    log_rotation_count: int = 5,
    also_stdout: bool = True,
    file_log_level: int = None,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name (typically module name)
        log_dir: Directory to store logs
        log_file: Log filename (defaults to name.log)
        level: Logging level for console
        log_rotation_size: Max size per log file in bytes
        log_rotation_count: Number of backup logs to keep
        also_stdout: Whether to also log to stdout
        file_log_level: Separate logging level for file (defaults to level)
        
    Returns:
        Logger object
    """
    if file_log_level is None:
        file_log_level = level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(min(level, file_log_level))  # Set to lowest level of the two
    logger.propagate = False  # Don't propagate to root logger
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set up console handler if requested
    if also_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Set up file handler
    if log_file is None:
        log_file = f"{name}.log"
    
    try:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_rotation_size,
            backupCount=log_rotation_count
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file handler fails, log to console and continue
        logger.error(f"Failed to set up file logging: {e}")
    
    return logger

def integrate_with_bittensor_logging(clear_existing_handlers: bool = True):
    """
    Integrate our logging with Bittensor's logging system
    
    Args:
        clear_existing_handlers: Whether to clear existing bt handlers
    """
    if clear_existing_handlers:
        for handler in bt.logging.handlers:
            bt.logging.removeHandler(handler)
    
    # Set up main bittensor logger
    main_logger = setup_logger("bittensor", level=logging.INFO)
    
    # Add our handlers to bittensor's logger
    for handler in main_logger.handlers:
        bt.logging.addHandler(handler)

# Create default loggers for major components
miner_logger = setup_logger("telegraph.miner", level=logging.INFO)
validator_logger = setup_logger("telegraph.validator", level=logging.INFO)
protocol_logger = setup_logger("telegraph.protocol", level=logging.INFO)

# Default logger for general use
logger = setup_logger("telegraph", level=logging.INFO)

# Integrate with bittensor logging system
integrate_with_bittensor_logging()

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component
    
    Args:
        name: Component name
        
    Returns:
        Logger: Configured logger instance
    """
    return setup_logger(f"telegraph.{name}")