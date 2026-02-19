"""
Professional logging configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

from src.config.settings import LoggingConfig


def setup_logger(
    name: str,
    config: LoggingConfig,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure and return a logger instance
    
    Args:
        name: Logger name (typically __name__)
        config: Logging configuration
        log_file: Optional log file path (overrides config)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler
    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    file_path = log_file or config.file
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
