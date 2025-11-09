from __future__ import annotations

import json
import logging
import typing as t
from contextlib import closing
from pathlib import Path
from time import perf_counter

from llama_index.core import Document

from rag.db.graph_db import GraphDBManager
from rag.parser import process_code_element
from rag.schemas import CodeElement
from rag.schemas.vector_config import VectorIndexConfig, get_vector_index_config
from rag.ingestion.embedding_loader import populate_embeddings

logger = logging.getLogger(__name__)


def _process_ast_files(
    ast_cache_dir: Path,
) -> t.Tuple[t.List[Document], t.List[CodeElement]]:
    """Process AST files into documents and code elements."""
    documents: t.List[Document] = []
    code_infos: t.List[CodeElement] = []

    logger.info(f"Scanning AST directory: {ast_cache_dir}")
    json_files = list(ast_cache_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        data = None
        try:
            logger.info(f"Processing file: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Robust defaults for optional fields
                data["parameters"] = data.get("parameters", [])
                data["dependencies"] = data.get("dependencies", [])
                data["base_classes"] = data.get("base_classes", [])
                data["calls"] = data.get("calls", [])

                # Ensure file_path is set
                data.setdefault("file_path", str(json_file))

                code_info = CodeElement(**data)
                llama_doc = process_code_element(code_info)

                documents.append(llama_doc)
                code_infos.append(code_info)

        except Exception as e:
            logger.exception(f"Error processing {json_file}: {str(e)}")
            if data is None:
                logger.debug(
                    f"No data loaded for {json_file} (failed before Json parser)"
                )
            continue

    logger.info(f"Successfully processed {len(documents)} documents")
    return documents, code_infos


def _create_schema(
    db_manager: GraphDBManager, vector_config: VectorIndexConfig
) -> None:
    logger.info("Database empty. Creating schema and loading data...")
    db_manager.create_schema(vector_config)


def _build_code_db_graph(
    db_manager: GraphDBManager,
    code_infos: t.List[CodeElement],
    vector_config: VectorIndexConfig,
) -> None:
    """Populate empty database with nodes and relationships."""

    _create_schema(db_manager, vector_config)

    logger.info("Phase 1: Creating nodes...")
    for code_info in code_infos:
        try:
            db_manager.create_node(code_info, vector_config)
        except Exception as e:
            logger.exception(f"Error creating nodes for {code_info.name}: {str(e)}")
            continue

    logger.info("\nPhase 2: Creating relationships...")
    for code_info in code_infos:
        try:
            db_manager.create_relationships(code_info, vector_config)
        except Exception as e:
            logger.exception(
                f"Error creating relationships for {code_info.name}: {str(e)}"
            )
            continue

    logger.info("Database created successfully")


def _check_db_populated(
    db_manager: GraphDBManager, vector_config: VectorIndexConfig
) -> bool:
    """Check if Neo4j already contains nodes for the configured label."""
    query = f"MATCH (n:{vector_config.node_label}) RETURN count(n) > 0 as exists"
    try:
        with closing(db_manager.driver.session()) as session:
            record = session.run(query).single()
            db_populated = bool(record and record.get("exists", False))
        label = vector_config.node_label
        logger.debug(f"Neo4j population check: label={label} populated={db_populated}")
        return db_populated
    except Exception as exc:
        logger.exception("Failed during Neo4j population check.")
        raise RuntimeError("Error checking Neo4j population status.") from exc


def process_code_files(
    ast_cache_dir: str,
    db_manager: t.Optional[GraphDBManager] = None,
    vector_config: t.Optional[VectorIndexConfig] = None,
    force_rebuild_graph: bool = False,
) -> int:
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
        db_populated = _check_db_populated(db_manager, vector_config)

        # 2) Parse files
        documents, code_infos = _process_ast_files(Path(ast_cache_dir))

        # Populate database if empty or force rebuild
        # TODO: Implement partial updates for changed files only.
        if not db_populated or force_rebuild_graph:
            logger.info(
                f"Populating Neo4j with new code elements "
                f"(label={vector_config.node_label})..."
            )
            _build_code_db_graph(db_manager, code_infos, vector_config)
            populate_embeddings(db_manager, documents, vector_config)
        else:
            logger.info("Using existing database")

        elapsed = perf_counter() - start_time
        logger.info(
            f"Processed AST directory {path} in"
            f"{elapsed}s ({len(documents)} documents)."
        )
        return len(documents)

    finally:
        db_manager.close()
