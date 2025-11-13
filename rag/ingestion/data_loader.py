from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from pathlib import Path
from time import perf_counter

import aiofiles
from llama_index.core import Document

from rag.db.graph_db import GraphDBManager
from rag.parser import process_code_element
from rag.schemas import CodeElement
from rag.schemas.vector_config import VectorIndexConfig, get_vector_index_config
from rag.ingestion.embedding_loader import populate_embeddings

logger = logging.getLogger(__name__)


def _filter_json_files(json_files: t.List[Path]) -> t.List[Path]:
    """Filter JSON files to only include code element files from results_* directories."""
    return [
        f
        for f in json_files
        if f.name not in [".rag_manifest.json", "progress.json"]
        and any(part.startswith("results_") for part in f.parts)
    ]


async def _process_single_file(
    json_file: Path, semaphore: asyncio.Semaphore
) -> t.Tuple[t.Optional[Document], t.Optional[CodeElement]]:
    """Process a single JSON file with semaphore control."""
    async with semaphore:
        data = None
        try:
            logger.debug(f"Processing file: {json_file}")
            async with aiofiles.open(json_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

                # Remove computed 'id' field if present
                # (it's a property, not a constructor arg)
                data.pop("id", None)

                # Robust defaults for optional fields
                data["parameters"] = data.get("parameters", [])
                data["dependencies"] = data.get("dependencies", [])
                data["base_classes"] = data.get("base_classes", [])
                data["calls"] = data.get("calls", [])

                # Ensure file_path is set
                data.setdefault("file_path", str(json_file))

                code_info = CodeElement(**data)
                llama_doc = process_code_element(code_info)

                return llama_doc, code_info

        except Exception as e:
            logger.error(f"Error processing {json_file}: {str(e)}")
            if data is None:
                logger.debug(
                    f"No data loaded for {json_file} (failed before Json parser)"
                )
            return None, None


async def _process_ast_files(
    ast_cache_dir: Path,
) -> t.Tuple[t.List[Document], t.List[CodeElement]]:
    """Process AST files into documents and code elements.

    Only processes code element JSON files from results_* subdirectories.
    """
    documents: t.List[Document] = []
    code_infos: t.List[CodeElement] = []

    logger.info(f"Scanning AST directory: {ast_cache_dir}")
    json_files = list(ast_cache_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files (before filtering)")

    # Filter out metadata files and only process files in results_* directories
    code_element_files = _filter_json_files(json_files)
    logger.info(f"Processing {len(code_element_files)} code element files")

    # Semaphore to limit concurrent file operations to 50
    semaphore = asyncio.Semaphore(50)

    # Process files concurrently with semaphore limiting
    results = await asyncio.gather(
        *[
            _process_single_file(json_file, semaphore)
            for json_file in code_element_files
        ],
        return_exceptions=True,
    )

    # Collect successful results
    for result in results:
        if isinstance(result, BaseException):
            logger.error(f"Exception during file processing: {result}")
            continue
        doc, code_info = result
        if doc is not None and code_info is not None:
            documents.append(doc)
            code_infos.append(code_info)

    logger.info(f"Successfully processed {len(documents)} documents")
    return documents, code_infos


async def _create_schema(
    db_manager: GraphDBManager, vector_config: VectorIndexConfig
) -> None:
    logger.info("Database empty. Creating schema and loading data...")
    await db_manager.create_schema(vector_config)


async def _build_code_db_graph(
    db_manager: GraphDBManager,
    code_infos: t.List[CodeElement],
    vector_config: VectorIndexConfig,
    max_concurrent: int = 20,
) -> None:
    """Populate empty database with nodes and relationships."""

    await _create_schema(db_manager, vector_config)

    # Semaphore to limit concurrent database operations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _create_node_with_semaphore(code_info: CodeElement) -> None:
        """Create a single node with semaphore control."""
        async with semaphore:
            try:
                await db_manager.create_node(code_info, vector_config)
            except Exception as e:
                logger.exception(f"Error creating nodes for {code_info.name}: {str(e)}")

    async def _create_relationships_with_semaphore(code_info: CodeElement) -> None:
        """Create relationships for a single node with semaphore control."""
        async with semaphore:
            try:
                await db_manager.create_relationships(code_info, vector_config)
            except Exception as e:
                logger.exception(
                    f"Error creating relationships for {code_info.name}: {str(e)}"
                )

    logger.info("Phase 1: Creating nodes...")
    # Process nodes concurrently with semaphore limiting
    await asyncio.gather(
        *[_create_node_with_semaphore(code_info) for code_info in code_infos],
        return_exceptions=True,
    )

    logger.info("\nPhase 2: Creating relationships...")
    # Process relationships concurrently with semaphore limiting
    await asyncio.gather(
        *[_create_relationships_with_semaphore(code_info) for code_info in code_infos],
        return_exceptions=True,
    )

    logger.info("Database created successfully")


async def _check_db_populated(
    db_manager: GraphDBManager, vector_config: VectorIndexConfig
) -> bool:
    """Check if Neo4j already contains nodes for the configured label."""
    query = f"MATCH (n:{vector_config.node_label}) RETURN count(n) > 0 as exists"
    try:
        driver = await db_manager.driver()
        async with driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            db_populated = bool(record and record.get("exists", False))
        label = vector_config.node_label
        logger.debug(f"Neo4j population check: label={label} populated={db_populated}")
        return db_populated
    except Exception as exc:
        logger.exception("Failed during Neo4j population check.")
        raise RuntimeError("Error checking Neo4j population status.") from exc


async def process_code_files(
    ast_cache_dir: str,
    db_manager: t.Optional[GraphDBManager] = None,
    vector_config: t.Optional[VectorIndexConfig] = None,
    force_rebuild_graph: bool = False,
) -> t.List[Document]:
    """
    Load AST-derived JSON files, optionally populate Neo4j, and return documents (llama_index.Document).

    Behaviour:
      • Validates the AST cache directory exists.
      • Checks if Neo4j already contains nodes for the configured label.
      • Processes all AST files into documents + code info.
      • Populates Neo4j only if database is empty.
      • Always closes the database driver.

    Returns:
        .

    Raises:
        FileNotFoundError: If the AST cache directory does not exist.
        RuntimeError: If database connectivity or query execution fails.
    """
    start_time = perf_counter()
    path = Path(ast_cache_dir)

    if not path.exists() or not path.is_dir():
        msg = f"AST cache directory does not exist or is not a directory: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    vector_config = vector_config or get_vector_index_config()  # Get config first
    db_manager = db_manager or GraphDBManager()
    try:
        # 1) DB state
        db_populated = await _check_db_populated(db_manager, vector_config)

        # 2) Parse files
        documents, code_infos = await _process_ast_files(Path(ast_cache_dir))

        # Populate database if empty or force rebuild
        # TODO: Implement partial updates for changed files only.
        if not db_populated or force_rebuild_graph:
            logger.info(
                f"Populating Neo4j with new code elements "
                f"(label={vector_config.node_label})..."
            )
            await _build_code_db_graph(db_manager, code_infos, vector_config)
            await populate_embeddings(db_manager, documents, vector_config)
        else:
            logger.info("Using existing database")

        elapsed = perf_counter() - start_time
        logger.info(
            f"Processed AST directory {path} in"
            f"{elapsed}s ({len(documents)} documents)."
        )
        return documents

    finally:
        await db_manager.close()
