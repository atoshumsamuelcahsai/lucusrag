import os
import logging
import typing as t
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
import asyncio


# Load environment variables (search parent directories to find .env in project root)
load_dotenv(override=True)  # override=True ensures env vars take precedence

logger = logging.getLogger(__name__)

# Registry for retriever factories
_RETRIEVER_REGISTRY: dict[str, t.Callable[..., "RetrieverSpec"]] = {}


def register_retriever(
    name: str,
) -> t.Callable[[t.Callable[..., "RetrieverSpec"]], t.Callable[..., "RetrieverSpec"]]:
    """Decorator to register retriever factory functions."""

    def decorator(
        factory: t.Callable[..., "RetrieverSpec"],
    ) -> t.Callable[..., "RetrieverSpec"]:
        _RETRIEVER_REGISTRY[name] = factory
        return factory

    return decorator


@dataclass
class RetrieverSpec:
    """Specification for a retriever with its postprocessors."""

    retriever: BaseRetriever
    postprocessors: list[BaseNodePostprocessor]


@register_retriever("bm25")
def _make_bm25_retriever(
    index: VectorStoreIndex,
    k: int,
    **kwargs: t.Any,
) -> RetrieverSpec:
    retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=k)
    return RetrieverSpec(retriever=retriever, postprocessors=[])


@register_retriever("bm25_reranked")
def _make_bm25_reranked_retriever(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    pre_k = initial_k or max(k * 3, 30)
    retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k)

    postprocessors: list[BaseNodePostprocessor] = [
        # Take the best k after rerank
        SentenceTransformerRerank(model=reranker_model, top_n=k)
    ]
    return RetrieverSpec(retriever=retriever, postprocessors=postprocessors)


@register_retriever("vector")
def _make_vector_retriever(
    index: VectorStoreIndex,
    k: int,
    similarity_cutoff: float | None = None,
    **kwargs: t.Any,
) -> RetrieverSpec:
    """Basic vector retriever with optional similarity filtering."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
    postprocessors: list[BaseNodePostprocessor] = []
    if similarity_cutoff is not None:
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        )
    return RetrieverSpec(retriever=retriever, postprocessors=postprocessors)


@register_retriever("vector_reranked")
def _make_vector_reranked_retriever(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    similarity_cutoff: float | None = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    """Vector retriever with cross-encoder reranking using SentenceTransformerRerank."""
    # Retrieve more candidates, then rerank to top k
    top_k = initial_k or max(k * 3, 10)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    postprocessors: list[BaseNodePostprocessor] = []

    # Add reranker FIRST (rerank all candidates, take top k)
    postprocessors.append(SentenceTransformerRerank(model=reranker_model, top_n=k))

    # Add similarity filter SECOND (filter low-confidence from top k)
    if similarity_cutoff is not None:
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        )

    return RetrieverSpec(retriever=retriever, postprocessors=postprocessors)


def _rrf_fuse(
    lists: list[list[NodeWithScore]],
    k: int,
    k_rrf: int = 60,
) -> list[NodeWithScore]:
    rank_maps: list[dict[str, int]] = [
        {n.node.node_id: i for i, n in enumerate(lst)} for lst in lists
    ]
    all_ids: set[str] = set().union(*[set(m.keys()) for m in rank_maps])

    scores: dict[str, float] = {}
    exemplar: dict[str, NodeWithScore] = {}
    for nid in all_ids:
        s = 0.0
        for rm in rank_maps:
            if nid in rm:
                rank = rm[nid]  # 0-based
                s += 1.0 / (k_rrf + rank + 1)
        scores[nid] = s

    # keep one exemplar per id
    for lst in lists:
        for n in lst:
            if n.node.node_id not in exemplar:
                exemplar[n.node.node_id] = n

    fused_ids = sorted(all_ids, key=lambda nid: scores[nid], reverse=True)[:k]
    return [
        NodeWithScore(node=exemplar[nid].node, score=scores[nid]) for nid in fused_ids
    ]


class HybridRRFRetriever(BaseRetriever):
    """Fuse Vector and BM25 via RRF and return a fused candidate set."""

    def __init__(
        self,
        vector: BaseRetriever,
        bm25: BaseRetriever,
        pre_k_each: int,
        pre_k_fused: int,
    ):
        super().__init__()
        self.vector = vector
        self.bm25 = bm25
        self.pre_k_each = pre_k_each
        self.pre_k_fused = pre_k_fused

    # ---- required sync path (BaseRetriever calls this in some code paths)
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Call child retrievers' sync API with the same bundle
        vec = self.vector.retrieve(query_bundle)
        bm = self.bm25.retrieve(query_bundle)
        vec = vec[: self.pre_k_each]
        bm = bm[: self.pre_k_each]
        return _rrf_fuse([vec, bm], k=self.pre_k_fused)

    # ---- async path used by BaseRetriever.aretrieve(...)
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Call child retrievers' async API with the same bundle
        vec, bm = await asyncio.gather(
            self.vector.aretrieve(query_bundle),
            self.bm25.aretrieve(query_bundle),
        )
        vec = vec[: self.pre_k_each]
        bm = bm[: self.pre_k_each]
        return _rrf_fuse([vec, bm], k=self.pre_k_fused)


@register_retriever("hybrid_rrf")
def _make_hybrid_rrf(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    **kwargs: t.Any,
) -> RetrieverSpec:
    """Hybrid = Vector + BM25 fused with RRF. Returns fused top-k directly (no reranker)."""
    pre_k_each = initial_k or max(k * 3, 30)  # candidates per retriever
    pre_k_fused = k  # fused list size to return

    vec_ret = VectorIndexRetriever(index=index, similarity_top_k=pre_k_each)
    bm25_ret = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k_each)

    retriever = HybridRRFRetriever(
        vector=vec_ret,
        bm25=bm25_ret,
        pre_k_each=pre_k_each,
        pre_k_fused=pre_k_fused,
    )
    return RetrieverSpec(retriever=retriever, postprocessors=[])


@register_retriever("hybrid_rrf_reranked")
def _make_hybrid_rrf_reranked(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,  # candidates per retriever
    fused_k: int | None = None,  # size of fused list before rerank
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    """
    Hybrid (BM25 + Vector) with RRF, then cross-encoder reranking.
    - Each retriever returns initial_k
    - RRF fuses to fused_k
    - Reranker picks final top-k
    """
    pre_k_each = initial_k or max(k * 3, 30)  # e.g., k=10 -> 50 each
    pre_k_fused = fused_k or max(k * 3, 20)  # e.g., k=10 -> 30 fused for reranker

    vec_ret = VectorIndexRetriever(index=index, similarity_top_k=pre_k_each)
    bm25_ret = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k_each)

    retriever = HybridRRFRetriever(
        vector=vec_ret,
        bm25=bm25_ret,
        pre_k_each=pre_k_each,
        pre_k_fused=pre_k_fused,
    )

    postprocessors: list[BaseNodePostprocessor] = [
        SentenceTransformerRerank(model=reranker_model, top_n=k)
    ]
    return RetrieverSpec(retriever=retriever, postprocessors=postprocessors)


def get_retriever_type() -> str:
    """Get retriever type from environment variable."""
    return os.getenv("RETRIEVER_TYPE", "vector")


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
        retriever_type = get_retriever_type()

    logger.info(
        f"Using retriever type: {retriever_type} ==============================="
    )

    factory = _RETRIEVER_REGISTRY.get(retriever_type)
    if not factory:
        valid = ", ".join(sorted(_RETRIEVER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown retriever type '{retriever_type}'. Valid options: {valid}"
        )

    spec = factory(index, k, **retriever_kwargs)

    synth = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE, verbose=True, use_async=True
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
