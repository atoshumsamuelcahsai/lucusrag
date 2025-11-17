import typing as t
import time
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
import asyncio
import os
import threading

from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from rag.schemas.vector_config import VectorIndexConfig
from dotenv import load_dotenv
from rag.logging_config import get_logger


logger = get_logger(__name__)
load_dotenv(override=True)  # override=True ensures env vars take precedence

# Thread-local storage for timing information
_timing_storage = threading.local()


def get_timing_info() -> dict:
    """Get timing information from thread-local storage."""
    if hasattr(_timing_storage, "timing_info"):
        return _timing_storage.timing_info.copy()
    return {}


def clear_timing_info() -> None:
    """Clear timing information from thread-local storage."""
    if hasattr(_timing_storage, "timing_info"):
        _timing_storage.timing_info = {}


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


def get_retriever_type() -> str:
    """Get retriever type from environment variable."""
    return os.getenv("RETRIEVER_TYPE", "vector")


def get_retriever(retriever_type: str) -> t.Callable[..., "RetrieverSpec"]:
    """Get retriever factory by type."""
    retriever_type = retriever_type.strip()
    factory = _RETRIEVER_REGISTRY.get(retriever_type)
    if factory is None:
        valid = ", ".join(sorted(_RETRIEVER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown retriever type '{retriever_type}'. Valid options: {valid}"
        )
    return factory


@dataclass
class RetrieverSpec:
    """Specification for a retriever with its postprocessors."""

    retriever: BaseRetriever
    postprocessors: list[BaseNodePostprocessor]


def _rrf_fuse(
    lists: list[list[NodeWithScore]],
    k: int,
    k_rrf: int = 60,
) -> list[NodeWithScore]:
    rrf_start = time.perf_counter()

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
    result = [
        NodeWithScore(node=exemplar[nid].node, score=scores[nid]) for nid in fused_ids
    ]

    rrf_time = time.perf_counter() - rrf_start
    # Store timing in thread-local storage
    if not hasattr(_timing_storage, "timing_info"):
        _timing_storage.timing_info = {}
    _timing_storage.timing_info["rrf_fusion"] = rrf_time

    return result


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
    def _retrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # Call child retrievers' sync API with the same bundle
        vec = self.vector.retrieve(query_bundle)
        bm = self.bm25.retrieve(query_bundle)
        vec = vec[: self.pre_k_each]
        bm = bm[: self.pre_k_each]
        return _rrf_fuse([vec, bm], k=self.pre_k_fused)

    # ---- async path used by BaseRetriever.aretrieve(...)
    async def _aretrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # Call child retrievers' async API with the same bundle
        vec, bm = await asyncio.gather(
            self.vector.aretrieve(query_bundle),
            self.bm25.aretrieve(query_bundle),
        )
        vec = vec[: self.pre_k_each]
        bm = bm[: self.pre_k_each]
        return _rrf_fuse([vec, bm], k=self.pre_k_fused)


def _expand_nodes_via_graph(
    nodes: t.List[NodeWithScore],
    index: VectorStoreIndex,
    expansion_depth: int = 1,
    max_expansion_nodes: int = 50,
) -> t.List[NodeWithScore]:
    """Expand nodes by following graph relationships.

    Uses multiple strategies to find related nodes:
    1. Traverse by node ID (UUID5 based on type, name, file_path)
    2. Traverse by node name (since relationships connect by name)
    3. Lookup targets from metadata (calls, dependencies fields)

    Args:
        nodes: Seed nodes to expand from
        index: VectorStoreIndex with graph_store and docstore attached
        expansion_depth: How many hops to follow in graph (default: 1)
        max_expansion_nodes: Max nodes to return after expansion (default: 50)

    Returns:
        Expanded list of nodes including originals and related nodes
    """
    graph_start = time.perf_counter()

    if not nodes or not hasattr(index, "_graph_store"):
        logger.debug("No nodes to expand or graph_store not available")
        graph_time = time.perf_counter() - graph_start
        if not hasattr(_timing_storage, "timing_info"):
            _timing_storage.timing_info = {}
        _timing_storage.timing_info["graph_expansion"] = graph_time
        return nodes

    graph_store = index._graph_store
    docstore = index.docstore

    # Get Neo4j connection from graph store
    if not hasattr(graph_store, "_driver") or graph_store._driver is None:
        logger.warning("Neo4j driver not available for graph expansion")
        graph_time = time.perf_counter() - graph_start
        if not hasattr(_timing_storage, "timing_info"):
            _timing_storage.timing_info = {}
        _timing_storage.timing_info["graph_expansion"] = graph_time
        return nodes

    try:
        config = VectorIndexConfig.from_env()
        depth = int(expansion_depth)  # Must be literal in Cypher
        label = config.node_label

        # Collect seed IDs and names
        seed_ids = [node.node.node_id for node in nodes]
        seed_names = [node.node.metadata.get("name", "") for node in nodes]

        logger.info(f"Expanding {len(seed_ids)} seed nodes: {seed_names[:3]}...")

        all_records = []

        # Strategy 1: Traverse by node ID
        q_by_id = f"""
            MATCH (seed:{label})
            WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[:CALLS|DEPENDS_ON*1..{depth}]-(related:{label})
            WHERE related.id <> seed.id
            WITH DISTINCT related, relationships(path) AS rels
            RETURN related.id AS id, related.name AS name, 
                   [r IN rels | type(r)] AS rel_types
            LIMIT $max_nodes
        """

        with graph_store._driver.session() as session:
            result = session.run(
                q_by_id, {"seed_ids": seed_ids, "max_nodes": max_expansion_nodes}
            )
            all_records.extend(list(result))

        # Strategy 2: Traverse by node name (relationships connect by name)
        q_by_name_traverse = f"""
            MATCH (seed:{label})
            WHERE seed.name IN $seed_names
            MATCH path = (seed)-[:CALLS|DEPENDS_ON*1..{depth}]-(related:{label})
            WHERE related.name <> seed.name
            WITH DISTINCT related, relationships(path) AS rels
            RETURN related.id AS id, related.name AS name, 
                   [r IN rels | type(r)] AS rel_types
            LIMIT $max_nodes
        """

        with graph_store._driver.session() as session:
            result = session.run(
                q_by_name_traverse,
                {"seed_names": seed_names, "max_nodes": max_expansion_nodes},
            )
            all_records.extend(list(result))

        # Strategy 3: Lookup targets from metadata (calls, dependencies)
        by_name_targets: set[str] = set()
        for node in nodes:
            metadata = node.node.metadata or {}
            for key in ("calls", "dependencies"):
                values = metadata.get(key, []) or []
                for val in values:
                    if isinstance(val, str) and val:
                        by_name_targets.add(val)

        if by_name_targets:
            q_by_name_lookup = f"""
                MATCH (related:{label})
                WHERE related.name IN $names
                RETURN DISTINCT related.id AS id, related.name AS name, 
                       ['BY_NAME'] AS rel_types
                LIMIT $max_nodes
            """

            with graph_store._driver.session() as session:
                result = session.run(
                    q_by_name_lookup,
                    {"names": list(by_name_targets), "max_nodes": max_expansion_nodes},
                )
                all_records.extend(list(result))

        # Deduplicate and collect related node info
        seen: set[str] = set()
        related_info: list[tuple[str, str, str]] = []

        for record in all_records:
            rid = record["id"]
            rname = record["name"]
            rtypes = record.get("rel_types") or []

            if rid and rid not in seen:
                seen.add(rid)
                # Use first relationship type or 'UNKNOWN'
                rtype = rtypes[0] if rtypes else "UNKNOWN"
                related_info.append((rid, rname, rtype))

        if not related_info:
            logger.info(
                f"No related nodes found for seeds: {seed_names[:3]}. "
                "This may mean no graph relationships exist (CALLS/DEPENDS_ON)."
            )
            return nodes

        # Retrieve related nodes from docstore
        expanded_nodes = list(nodes)  # Start with original seed nodes
        found_count = 0
        missing_count = 0

        for rel_id, rel_name, rel_type in related_info:
            if rel_id in docstore.docs:
                rel_node = docstore.docs[rel_id]
                # Add with a slightly lower score to preserve original ranking
                expanded_nodes.append(NodeWithScore(node=rel_node, score=0.5))
                found_count += 1
            else:
                missing_count += 1
                logger.debug(
                    f"Related node '{rel_name}' (ID: {rel_id[:8]}...) found in graph "
                    f"but missing from docstore"
                )

        # Collect unique relationship types for logging
        unique_rel_types = list(set(r[2] for r in related_info))

        logger.info(
            f"Graph expansion: {len(nodes)} seeds → {len(expanded_nodes)} total nodes. "
            f"Added {found_count} related nodes (from {len(related_info)} graph matches) "
            f"via {unique_rel_types} relationships. "
            f"({missing_count} found in graph but missing from docstore)"
        )

        result = expanded_nodes[:max_expansion_nodes]
        graph_time = time.perf_counter() - graph_start

        # Store timing in thread-local storage
        if not hasattr(_timing_storage, "timing_info"):
            _timing_storage.timing_info = {}
        _timing_storage.timing_info["graph_expansion"] = graph_time

        return result

    except Exception as e:
        logger.error(f"Graph expansion failed: {e}", exc_info=True)
        graph_time = time.perf_counter() - graph_start
        if not hasattr(_timing_storage, "timing_info"):
            _timing_storage.timing_info = {}
        _timing_storage.timing_info["graph_expansion"] = graph_time
        return nodes


class BM25GraphExpandedRetriever(BaseRetriever):
    """BM25 retriever with graph expansion (no vector, no hybrid)."""

    def __init__(
        self,
        bm25: BaseRetriever,
        index: VectorStoreIndex,
        top_k: int,
        expansion_depth: int = 1,
        max_expansion_nodes: int = 50,
    ):
        super().__init__()
        self.bm25 = bm25
        self.index = index
        self.top_k = top_k
        self.expansion_depth = expansion_depth
        self.max_expansion_nodes = max_expansion_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # BM25 retrieval
        bm_nodes = self.bm25.retrieve(query_bundle)[: self.top_k]

        # Expand via graph relationships
        expanded = _expand_nodes_via_graph(
            bm_nodes, self.index, self.expansion_depth, self.max_expansion_nodes
        )
        return expanded

    async def _aretrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # BM25 retrieval
        bm_nodes = await self.bm25.aretrieve(query_bundle)
        bm_nodes = bm_nodes[: self.top_k]

        # Expand via graph relationships (sync operation)
        expanded = _expand_nodes_via_graph(
            bm_nodes, self.index, self.expansion_depth, self.max_expansion_nodes
        )
        return expanded


class GraphExpandedHybridRetriever(BaseRetriever):
    """
    Hybrid retriever that expands BM25 results using graph relationships.

    Flow:
    1. BM25 retrieval → initial keyword matches
    2. Expand BM25 nodes via graph relationships (CALLS, DEPENDS_ON)
    3. Vector retrieval → semantic matches
    4. RRF fusion of expanded BM25 + vector results
    """

    def __init__(
        self,
        vector: BaseRetriever,
        bm25: BaseRetriever,
        index: VectorStoreIndex,
        pre_k_each: int,
        pre_k_fused: int,
        expansion_depth: int = 1,
        max_expansion_nodes: int = 50,
    ):
        super().__init__()
        self.vector = vector
        self.bm25 = bm25
        self.index = index
        self.pre_k_each = pre_k_each
        self.pre_k_fused = pre_k_fused
        self.expansion_depth = expansion_depth
        self.max_expansion_nodes = max_expansion_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # BM25 retrieval
        bm_nodes = self.bm25.retrieve(query_bundle)[: self.pre_k_each]

        # Expand BM25 results using graph
        bm_expanded = _expand_nodes_via_graph(
            bm_nodes, self.index, self.expansion_depth, self.max_expansion_nodes
        )

        # Vector retrieval
        vec_nodes = self.vector.retrieve(query_bundle)[: self.pre_k_each]

        # RRF fusion
        return _rrf_fuse([bm_expanded, vec_nodes], k=self.pre_k_fused)

    async def _aretrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        # BM25 and vector retrieval in parallel
        bm_nodes, vec_nodes = await asyncio.gather(
            self.bm25.aretrieve(query_bundle),
            self.vector.aretrieve(query_bundle),
        )

        bm_nodes = bm_nodes[: self.pre_k_each]
        vec_nodes = vec_nodes[: self.pre_k_each]

        # Expand BM25 results using graph (sync operation)
        bm_expanded = _expand_nodes_via_graph(
            bm_nodes, self.index, self.expansion_depth, self.max_expansion_nodes
        )

        # RRF fusion
        return _rrf_fuse([bm_expanded, vec_nodes], k=self.pre_k_fused)


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


@register_retriever("bm25_graph_expanded")
def _make_bm25_graph_expanded(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    expansion_depth: int = 1,
    max_expansion_nodes: int = 50,
    **kwargs: t.Any,
) -> RetrieverSpec:
    """
    BM25 retriever with graph expansion (no vector, no reranking).

    Workflow:
    1. BM25 retrieves initial keyword matches
    2. Expands results by following graph relationships (CALLS, DEPENDS_ON)
    3. Returns expanded results

    Args:
        index: VectorStoreIndex with graph_store attached
        k: Final number of results to return
        initial_k: Initial BM25 candidates before expansion (default: k*2)
        expansion_depth: How many hops to follow in graph (default: 1)
        max_expansion_nodes: Max nodes after expansion (default: 50)

    Returns:
        RetrieverSpec with BM25 graph-expanded retriever
    """
    pre_k = initial_k or max(k * 2, 30)
    bm25_ret = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k)

    retriever = BM25GraphExpandedRetriever(
        bm25=bm25_ret,
        index=index,
        top_k=k,
        expansion_depth=expansion_depth,
        max_expansion_nodes=max_expansion_nodes,
    )

    return RetrieverSpec(retriever=retriever, postprocessors=[])


@register_retriever("bm25_graph_expanded_reranked")
def _make_bm25_graph_expanded_reranked(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    expansion_depth: int = 1,
    max_expansion_nodes: int = 50,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    """
    BM25 retriever with graph expansion and cross-encoder reranking (no vector).

    Workflow:
    1. BM25 retrieves initial keyword matches
    2. Expands results by following graph relationships (CALLS, DEPENDS_ON)
    3. Cross-encoder reranks expanded candidates
    4. Returns top-k reranked results

    Args:
        index: VectorStoreIndex with graph_store attached
        k: Final number of results to return
        initial_k: Initial BM25 candidates before expansion (default: k*3)
        expansion_depth: How many hops to follow in graph (default: 1)
        max_expansion_nodes: Max nodes after expansion (default: 50)
        reranker_model: Cross-encoder model for reranking

    Returns:
        RetrieverSpec with BM25 graph-expanded retriever and reranker
    """
    pre_k = initial_k or max(k * 3, 30)
    bm25_ret = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k)

    retriever = BM25GraphExpandedRetriever(
        bm25=bm25_ret,
        index=index,
        top_k=max_expansion_nodes,  # Let expansion get more nodes
        expansion_depth=expansion_depth,
        max_expansion_nodes=max_expansion_nodes,
    )

    postprocessors: list[BaseNodePostprocessor] = [
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
    pre_k_each = initial_k or max(k * 3, 30)  # e.g., k=10 -> 30 each
    pre_k_fused = fused_k or max(k * 3, 20)  # e.g., k=10 -> 20 fused for reranker

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


@register_retriever("bm25_graph_vector_rrf_reranked")
def _make_bm25_graph_vector_rrf_reranked(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    fused_k: int | None = None,
    expansion_depth: int = 1,
    max_expansion_nodes: int = 50,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    """
    Full-featured hybrid retriever: BM25 + graph expansion + Vector + RRF + reranking.

    Workflow:
    1. BM25 retrieves initial keyword matches
    2. Expands BM25 results by following graph relationships (CALLS, DEPENDS_ON)
    3. Vector retrieves semantic matches
    4. RRF fuses expanded BM25 + vector results
    5. Cross-encoder reranks final candidates

    This is the most comprehensive retrieval strategy, combining:
    - Keyword matching (BM25)
    - Graph-aware expansion (code structure understanding)
    - Semantic search (Vector)
    - Intelligent fusion (RRF)
    - Quality reranking (Cross-encoder)

    Args:
        index: VectorStoreIndex with graph_store attached
        k: Final number of results to return
        initial_k: Candidates per retriever before expansion (default: k*3)
        fused_k: Size of fused list before reranking (default: k*2)
        expansion_depth: How many hops to follow in graph (default: 1)
        max_expansion_nodes: Max nodes to add via expansion (default: 50)
        reranker_model: Cross-encoder model for reranking

    Returns:
        RetrieverSpec with full-featured hybrid retriever and reranker
    """
    pre_k_each = initial_k or max(k * 3, 30)
    pre_k_fused = fused_k or max(k * 2, 20)

    vec_ret = VectorIndexRetriever(index=index, similarity_top_k=pre_k_each)
    bm25_ret = BM25Retriever.from_defaults(index=index, similarity_top_k=pre_k_each)

    retriever = GraphExpandedHybridRetriever(
        vector=vec_ret,
        bm25=bm25_ret,
        index=index,
        pre_k_each=pre_k_each,
        pre_k_fused=pre_k_fused,
        expansion_depth=expansion_depth,
        max_expansion_nodes=max_expansion_nodes,
    )

    postprocessors: list[BaseNodePostprocessor] = [
        SentenceTransformerRerank(model=reranker_model, top_n=k)
    ]
    return RetrieverSpec(retriever=retriever, postprocessors=postprocessors)


@register_retriever("hybrid_graph_expanded")
def _make_hybrid_graph_expanded(
    index: VectorStoreIndex,
    k: int,
    initial_k: int | None = None,
    fused_k: int | None = None,
    expansion_depth: int = 1,
    max_expansion_nodes: int = 50,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: t.Any,
) -> RetrieverSpec:
    """
    Alias for bm25_graph_vector_rrf_reranked for backward compatibility.

    Graph-expanded hybrid retriever with BM25, vector, RRF fusion, and reranking.
    """
    return _make_bm25_graph_vector_rrf_reranked(
        index=index,
        k=k,
        initial_k=initial_k,
        fused_k=fused_k,
        expansion_depth=expansion_depth,
        max_expansion_nodes=max_expansion_nodes,
        reranker_model=reranker_model,
        **kwargs,
    )
