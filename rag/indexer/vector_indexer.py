"""
Indexer module for creating vector search infrastructure.

This module handles Phase 3 of the pipeline:
- Creating vector indexes from existing Neo4j nodes with embeddings
- Configuring LlamaIndex settings (LLM, embeddings, parsers)
- Does NOT modify graph data (read-only for indexing)
"""

import logging
import asyncio

from rag.providers import get_llm, get_embeddings, EmbeddingProvider
from rag.parser import parse_documents_to_nodes
from rag.schemas.vector_config import VectorIndexConfig
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

import typing as t

logger = logging.getLogger(__name__)


_settings_configured = False
_settings_lock = asyncio.Lock()


async def graph_configure_settings(
    num_output: int = 420,
    context_window: int = 4200,
    llm_provider: str = "anthropic",
    embedding_provider: str = "voyage",
) -> None:
    """Configure the settings for the graph."""
    global _settings_configured
    if _settings_configured:
        return
    async with _settings_lock:
        if _settings_configured:
            return
        logger.info("Configuring LlamaIndex Settings....")

        embedding_model = get_embeddings(
            provider=EmbeddingProvider(embedding_provider).value
        )
        Settings.llm = get_llm(llm_provider)
        Settings.embed_model = embedding_model
        Settings.node_parser = parse_documents_to_nodes  # type: ignore[assignment]
        Settings.num_output = num_output
        Settings.context_window = context_window

        _settings_configured = True
        logger.info("LlamaIndex Settings configured successfully.")


def _graph_configure_settings_blocking(**kw) -> None:  # type: ignore
    """Call the async settings config from sync code safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(graph_configure_settings(**kw))
    else:
        # If caller is already async, they should await the async variant.
        # We don't change the public API here, just log.
        logger.debug("Settings called from running loop; assume configured elsewhere.")


def create_vector_index_from_existing_nodes(
    vector_config: t.Optional[VectorIndexConfig] = None,
) -> VectorStoreIndex:
    """
    Create vector index from existing Neo4j nodes with embeddings.

    This function:
    - Does NOT create or update nodes
    - Does NOT generate embeddings
    - Only creates a VectorStoreIndex that points to existing nodes
    - Uses existing embeddings and relationships

    Args:
        vector_config: Vector index configuration (loads from env if None)

    Returns:
        VectorStoreIndex configured to use existing Neo4j data
    """
    if vector_config is None:
        vector_config = VectorIndexConfig.from_env()

    _graph_configure_settings_blocking()

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

    # Create index from existing nodes (no document processing needed)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    logger.info("Vector index created successfully from existing nodes")

    return index
