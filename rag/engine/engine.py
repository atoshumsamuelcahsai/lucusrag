from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
import typing as t
from llama_index.core.postprocessor.types import BaseNodePostprocessor


def make_query_engine(index: VectorStoreIndex, k: int) -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
    post: t.Sequence[BaseNodePostprocessor] = [
        SimilarityPostprocessor(similarity_cutoff=0.3)
    ]
    synth = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE, verbose=True, use_async=True
    )
    return RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=synth, node_postprocessors=list(post)
    )


async def retrieve_documents_from_engine(
    engine: RetrieverQueryEngine, query: str, k: int = 10
) -> t.List[t.Dict[str, t.Any]]:
    """Retrieve top-k documents for a query without LLM processing.

    Uses the retriever from the query engine to get raw retrieval results.
    This is useful for evaluation purposes.

    Args:
        engine: RetrieverQueryEngine instance
        query: Query string to search for
        k: Number of documents to retrieve (default: 20)

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

    # Temporarily update similarity_top_k if needed
    original_k = getattr(retriever, "similarity_top_k", None)
    if k != original_k:
        setattr(retriever, "similarity_top_k", k)

    # Retrieve nodes using the engine's retriever
    nodes = await retriever.aretrieve(query)

    # Restore original k if we changed it
    if original_k is not None and k != original_k:
        setattr(retriever, "similarity_top_k", original_k)

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
