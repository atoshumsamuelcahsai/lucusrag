import os
from typing import Optional
import typing as t

from rag.exceptions import QueryProcessingError
from rag.indexer.orchestrator import CodeGraphIndexer
import asyncio
from rag.logging_config import get_logger

logger = get_logger(__name__)


_orchestrator: Optional[CodeGraphIndexer] = None
_orchestrator_lock = asyncio.Lock()


async def initialize_query_engine(
    ast_cache_dir: str, initialize_mode: str = "cold"
) -> CodeGraphIndexer:
    """
    Initialize the Orchestrator exactly once in an asyncio-safe way.
    """
    if initialize_mode not in {"cold", "refresh"}:
        raise QueryProcessingError("initialize_mode must be 'cold' or 'refresh'")

    global _orchestrator
    async with _orchestrator_lock:
        if _orchestrator is not None:
            logger.debug("Orchestrator already initialized by another coroutine.")
            return _orchestrator

        logger.info(f"Creating new Orchestrator with cache dir: {ast_cache_dir}")
        try:
            # Use top_k=15 to get more candidates for adaptive K selection
            _orchestrator = CodeGraphIndexer(ast_cache_dir, top_k=15)
            if initialize_mode == "cold":
                await _orchestrator.build()
            elif initialize_mode == "refresh":
                await _orchestrator.refresh()
            logger.info("Query engine created successfully.")
            return _orchestrator
        except Exception as exc:
            msg = "Failed to initialize Orchestrator."
            logger.exception(msg)
            raise QueryProcessingError(msg) from exc


async def get_orchestrator() -> CodeGraphIndexer:
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

    return await initialize_query_engine(ast_cache_dir)


def log_chunks_retrieved(response: t.Any) -> None:
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
        msg = "Query text cannot be empty or whitespace only."
        logger.error(msg)
        raise QueryProcessingError(msg)


async def process_query(query_text: str, log_chunks: bool = False) -> str:
    """
        Process a query using the cached Orchestrator.
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
        logger.info("Executing query...")
        logger.info("Starting document retrieval...")
        orch = await get_orchestrator()
        response = await orch.aquery(query_text)

        if log_chunks:
            log_chunks_retrieved(response)

        logger.info("Query processed successfully")
        return str(response)
    except QueryProcessingError:
        raise
    except Exception as e:
        # Capture unexpected failures.
        msg = f"Error while processing query: {e}"
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
