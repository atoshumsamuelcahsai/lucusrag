import time
import typing as t
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from rag.logging_config import get_logger
from rag.engine.retrievers import get_timing_info, clear_timing_info
from rag.engine.adaptive_k import adaptive_k_selection

logger = get_logger(__name__)


class TimedRetriever(BaseRetriever):
    """Wrapper retriever that tracks timing for embedding and retrieval steps."""

    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self._retriever = retriever
        self._timing_info: dict[str, float] = {}

    async def _aretrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        """Retrieve with timing tracking."""
        # Track embedding time (happens in the retriever's internal _get_query_embedding or similar)
        # For vector retrievers, embedding happens during retrieval
        retrieval_start = time.perf_counter()

        # Check if this is a vector retriever to track embedding separately
        if hasattr(self._retriever, "_get_query_embedding"):
            # This would require accessing the embedding model, which may not be directly accessible
            # For now, we'll track it as part of retrieval
            pass

        nodes = await self._retriever.aretrieve(query_bundle)
        retrieval_time = time.perf_counter() - retrieval_start

        self._timing_info["retrieval"] = retrieval_time
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> t.List[NodeWithScore]:
        """Sync retrieve with timing."""
        retrieval_start = time.perf_counter()
        nodes = self._retriever.retrieve(query_bundle)
        retrieval_time = time.perf_counter() - retrieval_start
        self._timing_info["retrieval"] = retrieval_time
        return nodes

    def get_timing_info(self) -> dict:
        """Get timing information."""
        return self._timing_info.copy()


class TimedQueryEngine:
    """
    Wrapper around RetrieverQueryEngine that tracks granular timing for each step:
    - Query embedding
    - Vector/BM25 retrieval
    - Graph expansion (if applicable)
    - RRF fusion (if applicable)
    - Reranking (if applicable)
    - LLM summarization
    """

    def __init__(self, engine: RetrieverQueryEngine):
        self._engine = engine
        self._retriever = engine.retriever
        # Try different ways to access response_synthesizer
        # RetrieverQueryEngine stores it in different ways depending on version
        self._response_synthesizer = (
            getattr(engine, "response_synthesizer", None)
            or getattr(engine, "_response_synthesizer", None)
            or getattr(engine, "_synthesizer", None)
            or getattr(engine, "synthesizer", None)
        )
        if self._response_synthesizer is None:
            logger.warning(
                "Could not access response_synthesizer directly. "
                "Will use fallback method for LLM timing."
            )
        self._postprocessors = (
            engine._node_postprocessors
            if hasattr(engine, "_node_postprocessors")
            else []
        )

    async def aquery(self, query_str: str) -> t.Any:
        """Execute query with detailed timing breakdown."""
        total_start = time.perf_counter()
        timing_info = {}

        # Clear any previous timing info
        clear_timing_info()

        try:
            query_bundle = QueryBundle(query_str=query_str)

            # Step 1 & 2: Query embedding + Document retrieval
            # Embedding happens inside the retriever, so we measure retrieval and estimate embedding
            retrieval_start = time.perf_counter()
            nodes = await self._retriever.aretrieve(query_bundle)
            retrieval_time = time.perf_counter() - retrieval_start

            # Get timing info from retrievers (graph expansion, RRF)
            retriever_timing = get_timing_info()

            # Estimate embedding time (typically 10-20% of retrieval for vector retrievers)
            # This is an estimate since embedding happens inside the retriever
            if self._is_vector_retriever():
                embed_time = (
                    retrieval_time * 0.15
                )  # Rough estimate (15% of retrieval time)
                actual_retrieval_time = retrieval_time - embed_time
            else:
                embed_time = 0.0
                actual_retrieval_time = retrieval_time

            # Subtract graph expansion and RRF time from retrieval time if they were measured
            if "graph_expansion" in retriever_timing:
                actual_retrieval_time -= retriever_timing["graph_expansion"]
            if "rrf_fusion" in retriever_timing:
                actual_retrieval_time -= retriever_timing["rrf_fusion"]

            timing_info["query_embedding"] = embed_time
            timing_info["document_retrieval"] = max(
                0, actual_retrieval_time
            )  # Ensure non-negative

            logger.info(f"⏱️  Query embedding: {embed_time*1000:.2f}ms")
            logger.info(
                f"⏱️  Document retrieval (vector/BM25): {max(0, actual_retrieval_time)*1000:.2f}ms "
                f"({len(nodes)} nodes retrieved)"
            )

            # Step 3: Graph expansion (if applicable)
            graph_expansion_time = retriever_timing.get("graph_expansion", 0.0)
            if graph_expansion_time > 0:
                timing_info["graph_expansion"] = graph_expansion_time
                logger.info(f"⏱️  Graph expansion: {graph_expansion_time*1000:.2f}ms")

            # Step 4: RRF fusion (if applicable)
            rrf_time = retriever_timing.get("rrf_fusion", 0.0)
            if rrf_time > 0:
                timing_info["rrf_fusion"] = rrf_time
                logger.info(f"⏱️  RRF fusion: {rrf_time*1000:.2f}ms")

            # Step 5: Reranking (postprocessors)
            rerank_time = 0.0
            if self._postprocessors:
                rerank_start = time.perf_counter()
                nodes_before_rerank = len(nodes)
                for postprocessor in self._postprocessors:
                    nodes = postprocessor.postprocess_nodes(nodes, query_str=query_str)
                rerank_time = time.perf_counter() - rerank_start
                timing_info["reranking"] = rerank_time
                logger.info(
                    f"⏱️  Reranking: {rerank_time*1000:.2f}ms "
                    f"({nodes_before_rerank} nodes → {len(nodes)} nodes after reranking)"
                )

            # Step 5.5: Adaptive K Selection (after reranking, before LLM)
            adaptive_start = time.perf_counter()
            nodes_before_adaptive = len(nodes)
            selected_nodes, adaptive_metadata = adaptive_k_selection(
                nodes,
                k_min=2,
                k_max=10,
                probability_target=0.70,  # Stop when cumulative probability >= 95%
                temperature=1.0,  # Softmax temperature (1.0 = standard softmax)
                max_cost_per_query=0.01,
                price_per_1k_tokens=0.001,  # $0.001 per 1k tokens = $1 per 1M tokens
            )
            nodes = selected_nodes
            adaptive_time = time.perf_counter() - adaptive_start
            timing_info["adaptive_k_selection"] = adaptive_time
            logger.info(
                f"⏱️  Adaptive K selection: {adaptive_time*1000:.2f}ms "
                f"({nodes_before_adaptive} candidates → {adaptive_metadata['selected_k']} selected, "
                f"cost: ${adaptive_metadata['total_cost']:.4f})"
            )

            # Step 6: LLM summarization
            total_chars = sum(len(node.node.get_content()) for node in nodes)
            logger.info(
                f"📄 Sending {len(nodes)} documents to LLM ({total_chars} chars total)"
            )
            llm_start = time.perf_counter()
            if self._response_synthesizer is not None:
                response = await self._response_synthesizer.asynthesize(
                    query=query_bundle,
                    nodes=nodes,
                )
            else:
                # Fallback: use the engine's synthesize method or aquery
                # We'll need to reconstruct the query bundle and call synthesize
                # For now, let's use the engine's internal method

                # Try to get the synthesizer from the engine
                synth = getattr(self._engine, "_synthesizer", None) or getattr(
                    self._engine, "synthesizer", None
                )
                if synth:
                    response = await synth.asynthesize(
                        query=query_bundle,
                        nodes=nodes,
                    )
                else:
                    # Last resort: call the engine's aquery but we've already done retrieval
                    # This is not ideal but will work
                    response = await self._engine.aquery(query_str)
                    # We can't measure LLM time separately in this case
                    llm_time = time.perf_counter() - llm_start
                    timing_info["llm_summarization"] = llm_time
                    logger.warning(
                        "Could not access synthesizer directly, LLM time includes retrieval"
                    )
                    logger.info(
                        f"⏱️  LLM summarization (estimated): {llm_time*1000:.2f}ms"
                    )
                    # Skip to total time calculation
                    total_time = time.perf_counter() - total_start
                    timing_info["total"] = total_time
                    logger.info("=" * 60)
                    logger.info("📊 QUERY PROCESSING TIMING SUMMARY")
                    logger.info("=" * 60)
                    logger.info(
                        f"  Query embedding:        {timing_info['query_embedding']*1000:>8.2f}ms"
                    )
                    logger.info(
                        f"  Document retrieval:    {timing_info['document_retrieval']*1000:>8.2f}ms"
                    )
                    if timing_info.get("graph_expansion", 0) > 0:
                        logger.info(
                            f"  Graph expansion:        {timing_info['graph_expansion']*1000:>8.2f}ms"
                        )
                    if timing_info.get("rrf_fusion", 0) > 0:
                        logger.info(
                            f"  RRF fusion:            {timing_info['rrf_fusion']*1000:>8.2f}ms"
                        )
                    if rerank_time > 0:
                        logger.info(
                            f"  Reranking:              {timing_info['reranking']*1000:>8.2f}ms"
                        )
                    if timing_info.get("adaptive_k_selection", 0) > 0:
                        cumulative_masses = adaptive_metadata.get(
                            "cumulative_masses", []
                        )
                        selected_k = adaptive_metadata.get("selected_k", 0)
                        if selected_k > 0 and len(cumulative_masses) >= selected_k:
                            final_cumulative = cumulative_masses[selected_k - 1]
                        else:
                            final_cumulative = 0.0
                        logger.info(
                            f"  Adaptive K selection:    {timing_info['adaptive_k_selection']*1000:>8.2f}ms "
                            f"(selected: {adaptive_metadata.get('selected_k', 'N/A')}, "
                            f"cumulative: {final_cumulative:.2f}, "
                            f"cost: ${adaptive_metadata.get('total_cost', 0):.4f})"
                        )
                    logger.info(
                        f"  LLM summarization:      {timing_info['llm_summarization']*1000:>8.2f}ms"
                    )
                    logger.info("-" * 60)
                    logger.info(
                        f"  TOTAL QUERY TIME:        {timing_info['total']*1000:>8.2f}ms ({timing_info['total']:.3f}s)"
                    )
                    logger.info("=" * 60)
                    return response

            llm_time = time.perf_counter() - llm_start
            timing_info["llm_summarization"] = llm_time
            logger.info(f"⏱️  LLM summarization: {llm_time*1000:.2f}ms")

            # Total time
            total_time = time.perf_counter() - total_start
            timing_info["total"] = total_time

            # Log summary
            logger.info("=" * 60)
            logger.info("📊 QUERY PROCESSING TIMING SUMMARY")
            logger.info("=" * 60)
            logger.info(
                f"  Query embedding:        {timing_info['query_embedding']*1000:>8.2f}ms"
            )
            logger.info(
                f"  Document retrieval:    {timing_info['document_retrieval']*1000:>8.2f}ms"
            )
            if timing_info.get("graph_expansion", 0) > 0:
                logger.info(
                    f"  Graph expansion:        {timing_info['graph_expansion']*1000:>8.2f}ms"
                )
            if timing_info.get("rrf_fusion", 0) > 0:
                logger.info(
                    f"  RRF fusion:            {timing_info['rrf_fusion']*1000:>8.2f}ms"
                )
            if rerank_time > 0:
                logger.info(
                    f"  Reranking:              {timing_info['reranking']*1000:>8.2f}ms"
                )
            if timing_info.get("adaptive_k_selection", 0) > 0:
                cumulative_masses = adaptive_metadata.get("cumulative_masses", [])
                selected_k = adaptive_metadata.get("selected_k", 0)
                if selected_k > 0 and len(cumulative_masses) >= selected_k:
                    final_cumulative = cumulative_masses[selected_k - 1]
                else:
                    final_cumulative = 0.0
                logger.info(
                    f"  Adaptive K selection:    {timing_info['adaptive_k_selection']*1000:>8.2f}ms "
                    f"(selected: {adaptive_metadata['selected_k']}, "
                    f"cumulative: {final_cumulative:.2f}, cost: ${adaptive_metadata['total_cost']:.4f})"
                )
            logger.info(
                f"  LLM summarization:      {timing_info['llm_summarization']*1000:>8.2f}ms"
            )
            logger.info("-" * 60)
            logger.info(
                f"  TOTAL QUERY TIME:        {timing_info['total']*1000:>8.2f}ms ({timing_info['total']:.3f}s)"
            )
            logger.info("=" * 60)

            return response

        except Exception as e:
            total_time = time.perf_counter() - total_start
            logger.error(f"Query processing failed after {total_time*1000:.2f}ms: {e}")
            raise

    def _has_graph_expansion(self) -> bool:
        """Check if retriever uses graph expansion."""
        retriever_type = type(self._retriever).__name__
        return "GraphExpanded" in retriever_type or "graph" in retriever_type.lower()

    def _has_rrf(self) -> bool:
        """Check if retriever uses RRF fusion."""
        retriever_type = type(self._retriever).__name__
        return "RRF" in retriever_type or "Hybrid" in retriever_type

    def _is_vector_retriever(self) -> bool:
        """Check if retriever is a vector retriever."""
        retriever_type = type(self._retriever).__name__
        return "Vector" in retriever_type or "vector" in retriever_type.lower()
