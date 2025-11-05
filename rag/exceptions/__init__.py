from __future__ import annotations


class AppError(Exception):
    """Base exception for application errors."""

    pass


class QueryProcessingError(AppError):
    """Raised when query processing fails."""

    pass
