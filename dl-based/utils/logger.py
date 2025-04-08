"""
Custom logger module with colored output using colorama.
"""

import sys
import time
from enum import Enum
from typing import TextIO
from colorama import Fore, Back
from colorama import Style, init

# Initialize colorama
init(autoreset=True)


class LogLevel(Enum):
    """Log level enumeration with corresponding colors."""

    DEBUG = (Fore.CYAN, "DEBUG")
    INFO = (Fore.GREEN, "INFO")
    WARNING = (Fore.YELLOW, "WARNING")
    ERROR = (Fore.RED, "ERROR")
    CRITICAL = (Fore.WHITE + Back.RED, "CRITICAL")


class ColorLogger(object):
    """
    A custom logger class that provides colored output for different log levels.

    Attributes:
        name (str): The name of the logger.
        show_time (bool): Whether to show timestamps in log messages.
        time_format (str): The format for timestamps.
        output_stream (TextIO): The stream to write log messages to.
        min_level (LogLevel): The minimum log level to display.
    """

    def __init__(
        self,
        name: str = "ColorLogger",
        show_time: bool = True,
        time_format: str = "%Y-%m-%d %H:%M:%S",
        output_stream: TextIO = sys.stdout,
        min_level: LogLevel = LogLevel.DEBUG,
    ):
        """
        Initialize the ColorLogger.

        Args:
            name: The name of the logger.
            show_time: Whether to show timestamps in log messages.
            time_format: The format for timestamps.
            output_stream: The stream to write log messages to.
            min_level: The minimum log level to display.
        """
        self.name = name
        self.show_time = show_time
        self.time_format = time_format
        self.output_stream = output_stream
        self.min_level = min_level

    def _log(self, level: LogLevel, message: str, *args, **kwargs):
        """
        Internal method to handle logging with the specified level.

        Args:
            level: The log level.
            message: The message to log.
            *args: Format arguments for the message.
            **kwargs: Additional format arguments for the message.
        """
        if level.value[1] == self.min_level.value[1] or self._should_log(level):
            color, level_name = level.value

            # Format the message with args and kwargs if provided
            if args or kwargs:
                try:
                    formatted_message = message.format(*args, **kwargs)
                except Exception as e:
                    formatted_message = f"{message} (Error formatting message: {e})"
            else:
                formatted_message = message

            # Prepare timestamp if needed
            timestamp = ""
            if self.show_time:
                timestamp = f"{time.strftime(self.time_format)} "

            # Construct and print the log message
            log_message = f"{timestamp}{color}[{level_name}]{Style.RESET_ALL} [{self.name}] {formatted_message}"
            print(log_message, file=self.output_stream)
            self.output_stream.flush()

    def _should_log(self, level: LogLevel) -> bool:
        """
        Determine if a message with the given level should be logged.

        Args:
            level: The log level to check.

        Returns:
            bool: True if the message should be logged, False otherwise.
        """
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }

        return level_order[level] >= level_order[self.min_level]

    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self._log(LogLevel.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, *args, **kwargs)

    def warn(self, message: str, *args, **kwargs):
        """Alias for warning."""
        self.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self._log(LogLevel.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, *args, **kwargs)

    def set_level(self, level: LogLevel):
        """
        Set the minimum log level.

        Args:
            level: The new minimum log level.
        """
        self.min_level = level

    def set_name(self, name: str):
        """
        Set the logger name.

        Args:
            name: The new logger name.
        """
        self.name = name


# Create a default logger instance
default_logger = ColorLogger()


# Convenience functions using the default logger
def debug(message: str, *args, **kwargs):
    """Log a debug message using the default logger."""
    default_logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """Log an info message using the default logger."""
    default_logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Log a warning message using the default logger."""
    default_logger.warning(message, *args, **kwargs)


def warn(message: str, *args, **kwargs):
    """Alias for warning using the default logger."""
    default_logger.warn(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Log an error message using the default logger."""
    default_logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Log a critical message using the default logger."""
    default_logger.critical(message, *args, **kwargs)
