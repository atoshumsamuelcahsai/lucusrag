import typing as t

from llama_index.core import Document, Settings

from rag.db.graph_db import GraphDBManager
from rag.schemas.vector_config import VectorIndexConfig
from rag.parser import create_text_representation
from rag.logging_config import get_logger

logger = get_logger(__name__)


async def populate_embeddings(
    db: GraphDBManager,
    documents: t.List[Document],
    vector_config: t.Optional[VectorIndexConfig] = None,
) -> int:
    """
    Generate embeddings for the given documents and upsert them to Neo4j nodes.
    Returns the number of nodes successfully updated.
    """
    if not documents:
        logger.info("No documents to embed.")
        return 0

    if vector_config is None:
        from rag.schemas.vector_config import get_vector_index_config

        vector_config = get_vector_index_config()

    if not Settings.embed_model:
        raise RuntimeError(
            "Embedding model not configured; call graph_configure_settings()."
        )

    # Prepare payload (compute text + embedding in Python, write once)
    embed_model = Settings.embed_model
    payload = []
    for doc in documents:
        node_id = doc.metadata.get("id")
        if not node_id:
            logger.debug("Skipping document without id: %s", doc.metadata)
            continue
        text = create_text_representation(doc.metadata)
        try:
            vec = embed_model.get_text_embedding(text)
            # Include metadata for _node_content field that LlamaIndex needs
            payload.append(
                {
                    "id": node_id,
                    "text": text,
                    "vec": vec,
                    "metadata": doc.metadata,  # Store full metadata for node reconstruction
                }
            )
        except Exception:
            logger.exception("Embedding failed for id=%s", node_id)

    if not payload:
        logger.info("Nothing to write (no successful embeddings).")
        return 0

    updated = await db.upsert_embeddings(payload, vector_config)
    logger.info("Upserted embeddings for %d nodes.", updated)
    return updated
