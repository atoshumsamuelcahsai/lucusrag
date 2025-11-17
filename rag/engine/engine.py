import typing as t
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from rag.engine import retrievers
from rag.logging_config import get_logger
import os
from rag.providers.llms import get_llm


# Load environment variables (search parent directories to find .env in project root)
load_dotenv(override=True)  # override=True ensures env vars take precedence

logger = get_logger(__name__)


def make_query_engine(
    index: VectorStoreIndex,
    k: int,
    retriever_type: str | None = None,
    **retriever_kwargs: t.Any,
) -> RetrieverQueryEngine:
    """Create query engine with configurable retriever type.

    Args:
        index: VectorStoreIndex instance
        k: Number of documents to retrieve
        retriever_type: Type of retriever to use (default: from RETRIEVER_TYPE env var or "vector")
        **retriever_kwargs: Additional arguments passed to retriever factory

    Returns:
        RetrieverQueryEngine instance

    Raises:
        ValueError: If retriever_type is unknown
        RuntimeError: If reranker is requested but not available
    """
    if retriever_type is None:
        retriever_type = retrievers.get_retriever_type()

    logger.info(
        f"Using retriever type: {retriever_type} ==============================="
    )
    logger.info(f"Top-K documents for retrieval: {k}")

    factory = retrievers.get_retriever(retriever_type)
    spec = factory(index, k, **retriever_kwargs)

    # Get LLM with logging
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model = os.getenv("LLM_MODEL", "default")
    llm = get_llm(llm_provider)
    logger.info(f"Using LLM: {llm_provider} / {llm_model}")

    # Get response mode from environment variable (default: COMPACT for speed)
    # Options: COMPACT (fast, 1 LLM call), TREE_SUMMARIZE (slow, multiple calls),
    #          REFINE (iterative), SIMPLE_SUMMARIZE (basic)
    response_mode_str = os.getenv("RESPONSE_MODE", "COMPACT").upper()
    try:
        response_mode = ResponseMode[response_mode_str]
    except KeyError:
        logger.warning(
            f"Invalid RESPONSE_MODE '{response_mode_str}', falling back to COMPACT. "
            f"Valid options: {[mode.name for mode in ResponseMode]}"
        )
        response_mode = ResponseMode.COMPACT

    logger.info(f"Using response mode: {response_mode.name}")

    synth = get_response_synthesizer(
        response_mode=response_mode, verbose=True, use_async=True, llm=llm
    )
    return RetrieverQueryEngine(
        retriever=spec.retriever,
        response_synthesizer=synth,
        node_postprocessors=spec.postprocessors,
    )


async def retrieve_documents_from_engine(
    engine: RetrieverQueryEngine, query: str, k: int = 10
) -> t.List[t.Dict[str, t.Any]]:
    """Retrieve top-k documents for a query without LLM processing.

    Applies the full retrieval pipeline including any configured postprocessors
    (e.g., reranking, similarity filtering). This respects the RETRIEVER_TYPE
    that was used to create the engine.

    Args:
        engine: RetrieverQueryEngine instance
        query: Query string to search for
        k: Number of documents to retrieve (default: 10)

    Returns:
        List of dicts with:
        - node_id: unique identifier
        - file_path: source file
        - element_type: type of code element
        - element_name: name of code element
        - score: similarity score
        - text: document content preview (first 200 chars)
    """
    # Get the retriever from the query engine
    retriever = engine.retriever

    # Retrieve initial candidates from vector search
    nodes = await retriever.aretrieve(query)

    # Apply postprocessors (reranking, filtering) if configured
    # This is what makes vector_reranked different from vector
    if hasattr(engine, "_node_postprocessors") and engine._node_postprocessors:
        for postprocessor in engine._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=query)

    # Limit to requested k after all postprocessing
    nodes = nodes[:k]

    # Extract relevant information
    results = []
    for node in nodes:
        metadata = node.node.metadata
        results.append(
            {
                "node_id": node.node.node_id,
                "file_path": metadata.get("file_path", "unknown"),
                "element_type": metadata.get("type", "unknown"),
                "element_name": metadata.get("name", "unknown"),
                "score": node.score if hasattr(node, "score") else 0.0,
                "text": node.node.get_content()[:200],  # Preview
            }
        )

    return results
