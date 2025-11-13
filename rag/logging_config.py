"""
Centralized logging configuration for the RAG application.

This module provides a consistent logging setup that ensures logs are
displayed in the console/terminal with proper formatting.
"""

import logging
import sys
from typing import Optional


# Track if logging has been configured to avoid overwriting
_LOGGING_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    enable_console: bool = True,
    force: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses default format.
        enable_console: Whether to enable console output (default: True)
        force: If True, reconfigure even if already configured (default: False)

    This function:
    - Sets up a StreamHandler to output logs to console/terminal
    - Configures log levels for specific modules
    - Ensures logs are visible even when used as a library
    - Is idempotent: safe to call multiple times (unless force=True)
    """
    global _LOGGING_CONFIGURED

    # If already configured and not forcing, skip
    if _LOGGING_CONFIGURED and not force:
        return

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Check if we already have a console handler
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in root_logger.handlers
    )

    # Only add console handler if needed
    if enable_console and not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    elif force and enable_console:
        # Force mode: remove old handlers and add new one
        root_logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Configure specific module log levels (always update these)
    logging.getLogger("rag.engine.retrievers").setLevel(logging.INFO)
    logging.getLogger("rag.indexer.orchestrator").setLevel(logging.INFO)
    logging.getLogger("rag.db.graph_db").setLevel(logging.INFO)
    logging.getLogger("rag.ingestion.data_loader").setLevel(logging.INFO)
    logging.getLogger("rag.ingestion.embedding_loader").setLevel(logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Note: This ensures logging is configured before returning the logger.
    """
    # Configure logging if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging()

    return logging.getLogger(name)
