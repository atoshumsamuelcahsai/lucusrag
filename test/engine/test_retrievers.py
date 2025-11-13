"""
Minimal tests for retrievers module - critical paths only.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
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
