import logging
import os

import logging
import os


def setup_logger(
    name: str, log_file: str = None, level: str = "INFO", log_to_console: bool = True
) -> logging.Logger:
    """
    Sets up a logger with options to log to a file and/or the console.

    Console logging is filtered by the provided level, but all logs are saved to the file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, no file logging is set up.
        level (str): Logging level for console output. One of ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].
        log_to_console (bool): If True, log messages will be printed to the console.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Ensure logger captures all levels

    # Ensure no duplicate handlers are added
    if not logger.hasHandlers():
        # Formatter for logs
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (filters logs based on the provided level)
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level.upper())  # Console verbosity level
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler (captures all logs regardless of level)
        if log_file:
            os.makedirs(
                os.path.dirname(log_file), exist_ok=True
            )  # Ensure the directory exists
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always save all logs to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
