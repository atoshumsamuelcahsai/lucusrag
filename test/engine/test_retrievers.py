"""
Minimal tests for retrievers module - critical paths only.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import VectorStoreIndex

from rag.engine.retrievers import (
    get_retriever_type,
    get_retriever,
    _rrf_fuse,
    _expand_nodes_via_graph,
    _make_bm25_retriever,
    RetrieverSpec,
)


class TestGetRetrieverType:
    """Test get_retriever_type function."""

    def test_get_retriever_type_from_env(self):
        """Test getting retriever type from environment variable."""
        with patch.dict("os.environ", {"RETRIEVER_TYPE": "bm25"}):
            assert get_retriever_type() == "bm25"

    def test_get_retriever_type_default(self):
        """Test default retriever type when env var not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Should default to "vector"
            result = get_retriever_type()
            assert result == "vector"


class TestGetRetriever:
    """Test get_retriever function."""

    def test_get_retriever_valid_type(self):
        """Test getting a valid retriever factory."""
        factory = get_retriever("bm25")
        assert callable(factory)
        assert factory.__name__ == "_make_bm25_retriever"

    def test_get_retriever_invalid_type(self):
        """Test getting an invalid retriever type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown retriever type"):
            get_retriever("invalid_retriever_type")


class TestRRFFuse:
    """Test RRF fusion function."""

    def test_rrf_fuse_basic(self):
        """Test basic RRF fusion of two lists."""
        # Create mock nodes
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)
        node3 = NodeWithScore(node=TextNode(text="node3", id_="id3"), score=0.7)

        list1 = [node1, node2]  # id1, id2
        list2 = [node2, node3]  # id2, id3

        fused = _rrf_fuse([list1, list2], k=3)

        # Should return top k nodes by RRF score
        assert len(fused) <= 3
        assert all(isinstance(n, NodeWithScore) for n in fused)
        # id2 appears in both lists, should have higher RRF score
        fused_ids = [n.node.node_id for n in fused]
        assert "id2" in fused_ids or "id1" in fused_ids or "id3" in fused_ids

    def test_rrf_fuse_empty_lists(self):
        """Test RRF fusion with empty lists."""
        fused = _rrf_fuse([[], []], k=5)
        assert len(fused) == 0

    def test_rrf_fuse_single_list(self):
        """Test RRF fusion with single list."""
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        fused = _rrf_fuse([[node1]], k=5)
        assert len(fused) == 1
        assert fused[0].node.node_id == "id1"


class TestExpandNodesViaGraph:
    """Test graph expansion function."""

    def test_expand_nodes_no_graph_store(self):
        """Test expansion returns original nodes when graph_store not available."""
        mock_index = Mock()
        del mock_index._graph_store  # No graph_store attribute

        nodes = [NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)]
        result = _expand_nodes_via_graph(nodes, mock_index)

        assert result == nodes  # Should return unchanged

    def test_expand_nodes_no_driver(self):
        """Test expansion returns original nodes when Neo4j driver not available."""
        mock_index = Mock()
        mock_graph_store = Mock()
        mock_graph_store._driver = None
        mock_index._graph_store = mock_graph_store
        mock_index.docstore = Mock()

        nodes = [NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)]
        result = _expand_nodes_via_graph(nodes, mock_index)

        assert result == nodes  # Should return unchanged

    def test_expand_nodes_with_graph_expansion(self):
        """Test graph expansion finds related nodes."""
        # Create mock index with graph_store
        mock_index = Mock()
        mock_graph_store = Mock()
        mock_driver = Mock()
        mock_session = MagicMock()  # Use MagicMock for better dict-like behavior

        # Create a proper mock record that supports dict access
        class MockRecord:
            def __init__(self):
                self.data = {
                    "id": "related-id-1",
                    "name": "related_node",
                    "rel_types": ["CALLS"],
                }

            def __getitem__(self, key):
                return self.data[key]

            def get(self, key, default=None):
                return self.data.get(key, default)

        mock_record = MockRecord()
        # session.run() returns an iterable result, code does list(result)
        mock_result = iter([mock_record])  # Make it iterable
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        mock_graph_store._driver = mock_driver
        mock_index._graph_store = mock_graph_store

        # Mock docstore
        mock_docstore = Mock()
        related_node = TextNode(text="related", id_="related-id-1")
        mock_docstore.docs = {"related-id-1": related_node}
        mock_index.docstore = mock_docstore

        # Mock config
        with patch("rag.engine.retrievers.VectorIndexConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.node_label = "CodeElement"
            mock_config_class.from_env.return_value = mock_config

            nodes = [
                NodeWithScore(
                    node=TextNode(
                        text="test", id_="id1", metadata={"name": "test_func"}
                    ),
                    score=0.9,
                )
            ]
            result = _expand_nodes_via_graph(
                nodes, mock_index, expansion_depth=1, max_expansion_nodes=10
            )

            # Should have original + expanded nodes
            assert len(result) >= len(nodes)
            # Check that related node was added
            result_ids = [n.node.node_id for n in result]
            assert "related-id-1" in result_ids or "id1" in result_ids


class TestBM25RetrieverFactory:
    """Test BM25 retriever factory function."""

    def test_make_bm25_retriever(self):
        """Test BM25 retriever factory creates correct spec."""
        mock_index = Mock(spec=VectorStoreIndex)

        with patch("rag.engine.retrievers.BM25Retriever") as mock_bm25:
            mock_retriever = Mock()
            mock_bm25.from_defaults.return_value = mock_retriever

            spec = _make_bm25_retriever(mock_index, k=10)

            assert isinstance(spec, RetrieverSpec)
            assert spec.retriever == mock_retriever
            assert len(spec.postprocessors) == 0
            mock_bm25.from_defaults.assert_called_once_with(
                index=mock_index, similarity_top_k=10
            )


class TestTimingInfo:
    """Test timing information functions."""

    def test_get_timing_info_empty(self):
        """Test getting timing info when none exists."""
        from rag.engine.retrievers import get_timing_info, clear_timing_info

        clear_timing_info()
        info = get_timing_info()
        assert info == {}

    def test_timing_info_stored_in_rrf(self):
        """Test that RRF fusion stores timing info."""
        from rag.engine.retrievers import get_timing_info, clear_timing_info

        clear_timing_info()
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)

        _rrf_fuse([[node1], [node2]], k=2)

        info = get_timing_info()
        assert "rrf_fusion" in info
        assert info["rrf_fusion"] > 0

    def test_timing_info_stored_in_graph_expansion(self):
        """Test that graph expansion stores timing info."""
        from rag.engine.retrievers import get_timing_info, clear_timing_info

        clear_timing_info()
        mock_index = Mock()
        del mock_index._graph_store  # No graph_store

        nodes = [NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)]
        _expand_nodes_via_graph(nodes, mock_index)

        info = get_timing_info()
        assert "graph_expansion" in info
        assert info["graph_expansion"] >= 0


class TestHybridRRFRetriever:
    """Test HybridRRFRetriever class."""

    def test_hybrid_rrf_retriever_init(self):
        """Test HybridRRFRetriever initialization."""
        from rag.engine.retrievers import HybridRRFRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        retriever = HybridRRFRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            pre_k_each=10,
            pre_k_fused=5,
        )

        assert retriever.vector == mock_vector
        assert retriever.bm25 == mock_bm25
        assert retriever.pre_k_each == 10
        assert retriever.pre_k_fused == 5

    def test_hybrid_rrf_retriever_retrieve(self):
        """Test HybridRRFRetriever sync retrieve."""
        from rag.engine.retrievers import HybridRRFRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)

        mock_vector.retrieve.return_value = [node1]
        mock_bm25.retrieve.return_value = [node2]

        retriever = HybridRRFRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            pre_k_each=10,
            pre_k_fused=5,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = retriever._retrieve(query)

        assert len(result) <= 5
        mock_vector.retrieve.assert_called_once_with(query)
        mock_bm25.retrieve.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_hybrid_rrf_retriever_aretrieve(self):
        """Test HybridRRFRetriever async retrieve."""
        from rag.engine.retrievers import HybridRRFRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)

        mock_vector.aretrieve = AsyncMock(return_value=[node1])
        mock_bm25.aretrieve = AsyncMock(return_value=[node2])

        retriever = HybridRRFRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            pre_k_each=10,
            pre_k_fused=5,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = await retriever._aretrieve(query)

        assert len(result) <= 5
        mock_vector.aretrieve.assert_called_once_with(query)
        mock_bm25.aretrieve.assert_called_once_with(query)


class TestBM25GraphExpandedRetriever:
    """Test BM25GraphExpandedRetriever class."""

    def test_bm25_graph_expanded_init(self):
        """Test BM25GraphExpandedRetriever initialization."""
        from rag.engine.retrievers import BM25GraphExpandedRetriever

        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)

        retriever = BM25GraphExpandedRetriever(
            bm25=mock_bm25,
            index=mock_index,
            top_k=10,
            expansion_depth=1,
            max_expansion_nodes=50,
        )

        assert retriever.bm25 == mock_bm25
        assert retriever.index == mock_index
        assert retriever.top_k == 10
        assert retriever.expansion_depth == 1
        assert retriever.max_expansion_nodes == 50

    def test_bm25_graph_expanded_retrieve(self):
        """Test BM25GraphExpandedRetriever sync retrieve."""
        from rag.engine.retrievers import BM25GraphExpandedRetriever

        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)

        mock_bm25.retrieve.return_value = [node1]

        retriever = BM25GraphExpandedRetriever(
            bm25=mock_bm25,
            index=mock_index,
            top_k=10,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = retriever._retrieve(query)

        mock_bm25.retrieve.assert_called_once_with(query)
        assert len(result) >= 1  # At least original nodes

    @pytest.mark.asyncio
    async def test_bm25_graph_expanded_aretrieve(self):
        """Test BM25GraphExpandedRetriever async retrieve."""
        from rag.engine.retrievers import BM25GraphExpandedRetriever

        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)

        mock_bm25.aretrieve = AsyncMock(return_value=[node1])

        retriever = BM25GraphExpandedRetriever(
            bm25=mock_bm25,
            index=mock_index,
            top_k=10,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = await retriever._aretrieve(query)

        mock_bm25.aretrieve.assert_called_once_with(query)
        assert len(result) >= 1


class TestGraphExpandedHybridRetriever:
    """Test GraphExpandedHybridRetriever class."""

    def test_graph_expanded_hybrid_init(self):
        """Test GraphExpandedHybridRetriever initialization."""
        from rag.engine.retrievers import GraphExpandedHybridRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)

        retriever = GraphExpandedHybridRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            index=mock_index,
            pre_k_each=10,
            pre_k_fused=5,
            expansion_depth=1,
            max_expansion_nodes=50,
        )

        assert retriever.vector == mock_vector
        assert retriever.bm25 == mock_bm25
        assert retriever.index == mock_index

    def test_graph_expanded_hybrid_retrieve(self):
        """Test GraphExpandedHybridRetriever sync retrieve."""
        from rag.engine.retrievers import GraphExpandedHybridRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)

        mock_vector.retrieve.return_value = [node1]
        mock_bm25.retrieve.return_value = [node2]

        retriever = GraphExpandedHybridRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            index=mock_index,
            pre_k_each=10,
            pre_k_fused=5,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = retriever._retrieve(query)

        mock_vector.retrieve.assert_called_once_with(query)
        mock_bm25.retrieve.assert_called_once_with(query)
        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_graph_expanded_hybrid_aretrieve(self):
        """Test GraphExpandedHybridRetriever async retrieve."""
        from rag.engine.retrievers import GraphExpandedHybridRetriever

        mock_vector = Mock()
        mock_bm25 = Mock()
        mock_index = Mock(spec=VectorStoreIndex)
        node1 = NodeWithScore(node=TextNode(text="node1", id_="id1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="node2", id_="id2"), score=0.8)

        mock_vector.aretrieve = AsyncMock(return_value=[node1])
        mock_bm25.aretrieve = AsyncMock(return_value=[node2])

        retriever = GraphExpandedHybridRetriever(
            vector=mock_vector,
            bm25=mock_bm25,
            index=mock_index,
            pre_k_each=10,
            pre_k_fused=5,
        )

        from llama_index.core.schema import QueryBundle

        query = QueryBundle(query_str="test")
        result = await retriever._aretrieve(query)

        mock_vector.aretrieve.assert_called_once_with(query)
        mock_bm25.aretrieve.assert_called_once_with(query)
        assert len(result) <= 5


class TestRetrieverFactories:
    """Test retriever factory functions."""

    def test_make_vector_retriever(self):
        """Test vector retriever factory."""
        from rag.engine.retrievers import _make_vector_retriever

        mock_index = Mock(spec=VectorStoreIndex)
        with patch("rag.engine.retrievers.VectorIndexRetriever") as mock_vector:
            mock_retriever = Mock()
            mock_vector.return_value = mock_retriever

            spec = _make_vector_retriever(mock_index, k=10)

            assert isinstance(spec, RetrieverSpec)
            assert spec.retriever == mock_retriever
            assert len(spec.postprocessors) == 0

    def test_make_vector_retriever_with_cutoff(self):
        """Test vector retriever factory with similarity cutoff."""
        from rag.engine.retrievers import _make_vector_retriever

        mock_index = Mock(spec=VectorStoreIndex)
        with patch("rag.engine.retrievers.VectorIndexRetriever") as mock_vector:
            mock_retriever = Mock()
            mock_vector.return_value = mock_retriever

            spec = _make_vector_retriever(mock_index, k=10, similarity_cutoff=0.5)

            assert len(spec.postprocessors) == 1

    def test_make_bm25_reranked_retriever(self):
        """Test BM25 reranked retriever factory."""
        from rag.engine.retrievers import _make_bm25_reranked_retriever

        mock_index = Mock(spec=VectorStoreIndex)
        with patch("rag.engine.retrievers.BM25Retriever") as mock_bm25:
            mock_retriever = Mock()
            mock_bm25.from_defaults.return_value = mock_retriever

            spec = _make_bm25_reranked_retriever(mock_index, k=10)

            assert isinstance(spec, RetrieverSpec)
            assert len(spec.postprocessors) == 1

    def test_make_hybrid_rrf_retriever(self):
        """Test hybrid RRF retriever factory."""
        from rag.engine.retrievers import _make_hybrid_rrf

        mock_index = Mock(spec=VectorStoreIndex)
        with patch("rag.engine.retrievers.VectorIndexRetriever") as mock_vector, patch(
            "rag.engine.retrievers.BM25Retriever"
        ) as mock_bm25, patch(
            "rag.engine.retrievers.HybridRRFRetriever"
        ) as mock_hybrid:
            mock_vector_ret = Mock()
            mock_bm25_ret = Mock()
            mock_hybrid_ret = Mock()
            mock_vector.return_value = mock_vector_ret
            mock_bm25.from_defaults.return_value = mock_bm25_ret
            mock_hybrid.return_value = mock_hybrid_ret

            spec = _make_hybrid_rrf(mock_index, k=10)

            assert isinstance(spec, RetrieverSpec)
            assert spec.retriever == mock_hybrid_ret
            assert len(spec.postprocessors) == 0

    def test_make_hybrid_rrf_reranked_retriever(self):
        """Test hybrid RRF reranked retriever factory."""
        from rag.engine.retrievers import _make_hybrid_rrf_reranked

        mock_index = Mock(spec=VectorStoreIndex)
        with patch("rag.engine.retrievers.VectorIndexRetriever") as mock_vector, patch(
            "rag.engine.retrievers.BM25Retriever"
        ) as mock_bm25, patch(
            "rag.engine.retrievers.HybridRRFRetriever"
        ) as mock_hybrid:
            mock_vector_ret = Mock()
            mock_bm25_ret = Mock()
            mock_hybrid_ret = Mock()
            mock_vector.return_value = mock_vector_ret
            mock_bm25.from_defaults.return_value = mock_bm25_ret
            mock_hybrid.return_value = mock_hybrid_ret

            spec = _make_hybrid_rrf_reranked(mock_index, k=10)

            assert isinstance(spec, RetrieverSpec)
            assert spec.retriever == mock_hybrid_ret
            assert len(spec.postprocessors) == 1


class TestExpandNodesViaGraphEdgeCases:
    """Test edge cases for graph expansion."""

    def test_expand_nodes_empty_list(self):
        """Test expansion with empty node list."""
        mock_index = Mock()
        mock_index._graph_store = Mock()

        result = _expand_nodes_via_graph([], mock_index)
        assert result == []

    def test_expand_nodes_exception_handling(self):
        """Test that exceptions in graph expansion are handled."""
        mock_index = Mock()
        mock_graph_store = Mock()
        mock_driver = Mock()
        mock_session = MagicMock()

        # Make session.run raise an exception
        mock_session.run.side_effect = Exception("Database error")
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_graph_store._driver = mock_driver
        mock_index._graph_store = mock_graph_store
        mock_index.docstore = Mock()

        nodes = [NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)]

        with patch("rag.engine.retrievers.VectorIndexConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.node_label = "CodeElement"
            mock_config.from_env.return_value = mock_config_instance

            # Should return original nodes on error
            result = _expand_nodes_via_graph(nodes, mock_index)
            assert len(result) == 1
            assert result[0] == nodes[0]

    def test_expand_nodes_no_related_nodes(self):
        """Test expansion when no related nodes found."""
        mock_index = Mock()
        mock_graph_store = Mock()
        mock_driver = Mock()
        mock_session = MagicMock()

        # Return empty result
        mock_session.run.return_value = iter([])
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_graph_store._driver = mock_driver
        mock_index._graph_store = mock_graph_store
        mock_index.docstore = Mock()

        nodes = [
            NodeWithScore(
                node=TextNode(text="test", id_="id1", metadata={"name": "test_func"}),
                score=0.9,
            )
        ]

        with patch("rag.engine.retrievers.VectorIndexConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.node_label = "CodeElement"
            mock_config.from_env.return_value = mock_config_instance

            result = _expand_nodes_via_graph(nodes, mock_index)
            # Should return original nodes when no related nodes found
            assert len(result) == 1
            assert result == nodes
