"""
Indexer module for creating vector search infrastructure.

This module handles Phase 3 of the pipeline:
- Creating vector indexes from existing Neo4j nodes with embeddings
- Configuring LlamaIndex settings (LLM, embeddings, parsers)
- Does NOT modify graph data (read-only for indexing)
"""

import logging
import asyncio
import os
import typing as t
from dotenv import load_dotenv

from rag.providers import get_llm, get_embeddings
from rag.parser import parse_documents_to_nodes
from rag.schemas.vector_config import VectorIndexConfig
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


_settings_configured = False
_settings_lock = asyncio.Lock()


async def graph_configure_settings() -> None:
    """Configure the settings for the graph.

    All configuration comes from environment variables:
    - LLM_PROVIDER: Which LLM provider to use (anthropic/openai, default: anthropic)
    - EMBEDDING_PROVIDER: Which embedding provider to use (default: voyage)
    - LLM_MAX_OUTPUT_TOKENS: Maximum output tokens (default: 768)
    - LLM_CONTEXT_WINDOW: Context window size (default: 4200)
    """
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "voyage")
    num_output = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "768"))
    context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "4200"))

    global _settings_configured
    if _settings_configured:
        return
    async with _settings_lock:
        if _settings_configured:
            return
        logger.info("Configuring LlamaIndex Settings....")

        embedding_model = get_embeddings(provider=embedding_provider)
        Settings.llm = get_llm(llm_provider)
        Settings.embed_model = embedding_model
        Settings.node_parser = parse_documents_to_nodes  # type: ignore[assignment]
        Settings.num_output = num_output
        Settings.context_window = context_window

        _settings_configured = True
        logger.info(
            f"LlamaIndex Settings configured. Provider: {llm_provider}, Model: {os.getenv('LLM_MODEL', 'default')}"
        )


def _graph_configure_settings_blocking() -> None:
    """Call the async settings config from sync code safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(graph_configure_settings())
    else:
        # If caller is already async, they should await the async variant.
        # We don't change the public API here, just log.
        logger.debug("Settings called from running loop; assume configured elsewhere.")


def create_vector_index_from_existing_nodes(
    vector_config: t.Optional[VectorIndexConfig] = None,
    docs: list[Document] | None = None,
) -> VectorStoreIndex:
    """
    Create vector index from existing Neo4j nodes with embeddings.

    This function:
    - Does NOT create or update nodes
    - Does NOT generate embeddings
    - Only creates a VectorStoreIndex that points to existing nodes
    - Uses existing embeddings and relationships
    - Hydrates in-memory docstore for BM25/keyword retrievers

    Args:
        vector_config: Vector index configuration (loads from env if None)
        ast_cache_dir: Path to AST cache (currently unused, kept for backward compat)
        docs: List of Documents from process_code_files (REQUIRED for BM25 support)

    Returns:
        VectorStoreIndex configured to use existing Neo4j data with populated docstore

    Raises:
        ValueError: If docs is None or empty
    """
    if vector_config is None:
        vector_config = VectorIndexConfig.from_env()

    _graph_configure_settings_blocking()

    # Validate docs parameter
    if not docs:
        raise ValueError(
            "docs parameter is required for BM25 support. "
            "Pass the documents returned from process_code_files()."
        )

    logger.info("Creating vector index from existing nodes...")

    neo4j_url = vector_config.neo4j_url
    neo4j_user = vector_config.neo4j_user
    neo4j_password = vector_config.neo4j_password

    # Create vector store - points to existing nodes with embeddings
    vector_store = Neo4jVectorStore(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=vector_config.name,
        node_label=vector_config.node_label,
        store_nodes=True,
        embedding_dimension=vector_config.dimension,
        similarity_metric=vector_config.similarity_metric,
        text_node_property="text",
        embedding_node_property=vector_config.vector_property,
    )

    # Create graph store - uses existing relationships
    graph_store = Neo4jGraphStore(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        node_label=vector_config.node_label,
        edge_labels=["INHERITS_FROM", "CALLS", "DEPENDS_ON"],
        node_id_property="id",
        text_node_property="text",
        embedding_node_property=vector_config.vector_property,
        edge_relation_property="RELATES_TO",
    )

    # Hydrate docstore with nodes for BM25 and keyword retrievers
    logger.info(f"Parsing {len(docs)} documents into nodes for docstore...")
    nodes_to_store = parse_documents_to_nodes(docs)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes_to_store)

    # Also query Neo4j to get ALL existing nodes and add them to docstore
    # This ensures graph expansion can find nodes from previous runs
    logger.info("Fetching all nodes from Neo4j to populate docstore...")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            result = session.run(
                f"MATCH (n:{vector_config.node_label}) RETURN n.id AS id, n.text AS text, n"
            )
            neo4j_node_count = 0
            for record in result:
                node_id = record["id"]
                # Only add if not already in docstore
                if node_id not in docstore.docs:
                    # Create a TextNode from Neo4j data
                    from llama_index.core.schema import TextNode

                    node_data = dict(record["n"])
                    text_node = TextNode(
                        id_=node_id,
                        text=node_data.get("text", ""),
                        metadata={
                            k: v
                            for k, v in node_data.items()
                            if k not in ["id", "text", "embedding"]
                        },
                    )
                    docstore.add_documents([text_node])
                    neo4j_node_count += 1
        driver.close()
        logger.info(f"Added {neo4j_node_count} additional nodes from Neo4j to docstore")
    except Exception as e:
        logger.warning(f"Could not fetch additional nodes from Neo4j: {e}")

    logger.info(
        f"Hydrated docstore with {len(docstore.docs)} total nodes for keyword retrievers."
    )

    # Create index from existing Neo4j vector store; then attach docstore + graph_store for hybrid use
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=False,
    )
    # Manually attach stores needed by BM25 and graph operations
    setattr(index, "_docstore", docstore)
    setattr(index, "_graph_store", graph_store)
    logger.info(
        f"Index created. Docstore manually attached with {len(index.docstore.docs)} nodes for BM25."
    )

    logger.info("Vector index created successfully from existing nodes")

    return index
