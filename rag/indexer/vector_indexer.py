import logging
import asyncio

from rag import embedding
from rag.ingestion.data_loader import get_vector_index_config, process_code_files
from rag.providers.llm import get_llm, EmbeddingProvider
from rag.parser import CodeElementGraphParser
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from rag.db.graph_db import VectorIndexConfig

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

        custom_parser = CodeElementGraphParser()
        embedding_model = embedding.get_embeddings(
            provider=EmbeddingProvider(embedding_provider).value
        )
        Settings.llm = get_llm(llm_provider)
        Settings.embed_model = embedding_model
        Settings.node_parser = custom_parser
        Settings.num_output = num_output
        Settings.context_window = context_window

        _settings_configured = True
        logger.info("LlamaIndex Settings configured successfully.")


def _graph_configure_settings_blocking(**kw) -> None:
    # TODO: Check for better way to do this.
    """Call the async settings config from sync code safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(graph_configure_settings(**kw))
    else:
        # If caller is already async, they should await the async variant.
        # We don’t change the public API here, just log.
        logger.debug("Settings called from running loop; assume configured elsewhere.")


def get_vector_index(documents, vector_config: VectorIndexConfig):
    """Create a vector store index with Neo4j backend
    using existing GraphDBManager."""
    _graph_configure_settings_blocking()

    if not documents:
        logger.warning(
            "get_vector_index called with \
            empty documents; index will be empty."
        )

    # Initialize GraphDBManager with existing config
    # db_manager = GraphDBManager()

    neo4j_url = vector_config.neo4j_url
    neo4j_user = vector_config.neo4j_user
    neo4j_password = vector_config.neo4j_password

    # Create vector store using URL and credentials instead of driver
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

    # Create graph store using existing relationships
    graph_store = Neo4jGraphStore(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        node_label=vector_config.node_label,
        edge_labels=["HAS_METHOD", "INHERITS_FROM", "CALLS", "DEPENDS_ON"],
        node_id_property="id",
        text_node_property="text",
        embedding_node_property="embedding",
        edge_relation_property="RELATES_TO",
    )

    # Create the index
    # transformations creates embedding
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        graph_store=graph_store,
        transformations=[Settings.node_parser],
    )

    return index


def get_vector_store_index(ast_cache_dir: str):
    documents = process_code_files(ast_cache_dir)
    vector_config = get_vector_index_config()
    return get_vector_index(documents, vector_config)
