"""
LucusRAG - Code retrieval and understanding system.

A production-ready RAG system for code search and understanding using
semantic search, graph traversal, and hybrid retrieval strategies.
"""

__version__ = "0.1.0"

# Configure logging on import to ensure logs are visible
# This is idempotent - safe to call multiple times
from rag.logging_config import configure_logging

# Auto-configure logging when rag package is imported
# This ensures logs are visible even when used as a library
configure_logging()
