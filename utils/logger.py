
import logging
import os
import sys
from typing import Optional
from datetime import datetime

# Global logger configuration
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_LEVEL = logging.INFO
_LOG_DIR = "logs"
_LOGGERS = {}

def setup_logging(log_dir: str = _LOG_DIR,
                 log_level: int = _LOG_LEVEL,
                 log_to_console: bool = True) -> None:
    """
    Setup global logging configuration.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (e.g., logging.INFO)
        log_to_console: Whether to log to console
    """
    global _LOG_DIR, _LOG_LEVEL
    _LOG_DIR = log_dir
    _LOG_LEVEL = log_level

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(file_handler)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger.addHandler(console_handler)

    root_logger.info(f"Logging initialized. Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    global _LOGGERS

    if name in _LOGGERS:
        return _LOGGERS[name]

    # Create logger
    logger = logging.getLogger(name)

    # If no handlers exist yet, set up basic logging
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logging()

    _LOGGERS[name] = logger
    return logger