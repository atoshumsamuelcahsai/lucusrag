from rag.indexer.vector_indexer import get_index
from rag.schemas.vector_config import VectorIndexConfig
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex


def get_query_engine(ast_cache_dir: str, vector_config: VectorIndexConfig, k: int = 5):
    """Get graph-based index."""

    index = get_index(ast_cache_dir)

    # Create retriever with higher top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,  # Increased to get more context
    )

    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.3)]
    synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", verbose=True, use_async=True
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=node_postprocessors,
    )


def make_query_engine(index: VectorStoreIndex, k: int) -> RetrieverQueryEngine:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
    post = [SimilarityPostprocessor(similarity_cutoff=0.3)]
    synth = get_response_synthesizer(
        response_mode="tree_summarize", verbose=True, use_async=True
    )
    return RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=synth, node_postprocessors=post
    )
