"""
Module for processing and indexing code files into Neo4j database.
Handles database population and code element processing.
"""

import os
from llama_index.core import (
    Response,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import Optional

from rag import engine
from rag.exceptions import QueryProcessingError
from rag.ingestion.data_loader import get_vector_index_config

import logging
import asyncio

logger = logging.getLogger(__name__)


_query_engine: Optional[RetrieverQueryEngine] = None
_query_engine_lock = asyncio.Lock()


async def initialize_query_engine(ast_cache_dir: str) -> RetrieverQueryEngine:
    """
    Initialize the query engine exactly once in an asyncio-safe way.
    """
    global _query_engine
    async with _query_engine_lock:
        if _query_engine is not None:
            logger.debug("Query engine already initialized by another coroutine.")
            return _query_engine

        logger.info(f"Creating new query engine with cache dir: {ast_cache_dir}")
        try:
            _query_engine = await engine.get_query_engine(
                ast_cache_dir, get_vector_index_config()
            )
            logger.info("Query engine created successfully.")
            return _query_engine
        except Exception as exc:
            msg = "Failed to initialize query engine."
            logger.exception(msg)
            raise QueryProcessingError(msg) from exc


async def get_query_engine() -> RetrieverQueryEngine:
    """
    Retrieve the cached query engine or initialize it once asynchronously.

    Async-safe and idempotent:
      • Uses `asyncio.Lock` to prevent concurrent initialization.
      • Raises `QueryProcessingError` if required env vars are missing.
      • Reuses cached engine after first creation.
    """
    ast_cache_dir = os.getenv("AST_CACHE_DIR")
    if not ast_cache_dir:
        msg = "AST_CACHE_DIR environment variable is not set"
        logger.error(msg)
        raise QueryProcessingError(msg)

    global _query_engine
    if _query_engine is not None:
        logger.info("Using existing query engine from cache")
        return _query_engine

    return await initialize_query_engine()


def log_chunks_retrieved(response: Response) -> None:
    # Log the number of source nodes/chunks used
    nodes = getattr(response, "source_nodes", None)
    if nodes is None:
        logger.warning("No source nodes retrieved")
        return

    logger.info(f"Retrieved {len(response.source_nodes)} source chunks")
    for idx, node in enumerate(nodes, start=1):
        score = getattr(node, "score", "N/A")
        logger.debug(f"Chunk {idx} score: {score}")


def validate_query_text(query_text: str) -> None:
    """Ensure query text is not empty or whitespace only."""
    if not query_text.strip():
        msg = "Empty text cannot be empty or whitespace only."
        logger.error(msg)
        raise QueryProcessingError(msg)


async def process_query(query_text: str, log_chunks: bool = False) -> str:
    """
        Process a query using the cached query engine.
        Args:
        - query_text: The query string to process.
        - log_chunks: Whether to log retrieved document chunks.
        Returns:
        - The query response as a string.
    Raises:
        QueryProcessingError: For validation or processing failures.
    """
    logger.info(f"Processing query: {query_text}")
    try:
        validate_query_text(query_text)

        logger.info("Initializing query engine...")
        query_engine: RetrieverQueryEngine = await get_query_engine()

        logger.info("Executing query...")
        logger.info("Starting document retrieval...")
        response = await query_engine.aquery(query_text)

        if log_chunks:
            log_chunks_retrieved(response)

        logger.info("Query processed successfully")
        return str(response)
    except QueryProcessingError:
        raise
    except Exception as e:
        # Capture unexpected failures.
        msg = f"Error in while processing query: {e}"
        logger.exception(msg)
        raise QueryProcessingError(msg) from e


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the current event loop or create a new one if none exists."""
    try:
        loop = asyncio.get_event_loop()
        logger.debug("Retrieved existing event loop")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug("Created new event loop")
    return loop


def process_query_sync(query_text: str) -> str:
    """
    Synchronous wrapper for `process_query`.

    - Safe to call when no event loop exists.
    - Raises `QueryProcessingError` on failure.
    """
    logger.info("Starting synchronous query processing...")
    try:
        loop = get_event_loop()
        if loop.is_running():
            msg = (
                "Event loop is already running. " "Cannot process query synchronously."
            )
            logger.error(msg)
            raise QueryProcessingError(msg)

        logger.debug("Executing process_query coroutine...")
        result = loop.run_until_complete(process_query(query_text))
        logger.info("Query processing completed successfully")
        return result
    except QueryProcessingError:
        raise
    except Exception as e:
        # Capture unexpected failures and chain context.
        msg = f"Error in sync wrapper in process_query_sync: {e}"
        logger.exception(msg)
        raise QueryProcessingError(msg) from e
