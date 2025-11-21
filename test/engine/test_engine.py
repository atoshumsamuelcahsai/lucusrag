"""
Minimal tests for engine module - critical paths only.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from rag.engine.engine import make_query_engine, retrieve_documents_from_engine


class TestMakeQueryEngine:
    """Test make_query_engine function."""

    def test_make_query_engine_default_retriever(self):
        """Test creating query engine with default retriever type."""
        mock_index = Mock(spec=VectorStoreIndex)

        with patch("rag.engine.engine.retrievers") as mock_retrievers:
            mock_retrievers.get_retriever_type.return_value = "vector"
            mock_factory = Mock()
            mock_factory.return_value = Mock(retriever=Mock(), postprocessors=[])
            mock_retrievers.get_retriever.return_value = mock_factory

            with patch("rag.engine.engine.get_llm") as mock_get_llm:
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm

                with patch("rag.engine.engine.get_response_synthesizer") as mock_synth:
                    mock_synth.return_value = Mock()
                    engine = make_query_engine(mock_index, k=10)

                    assert isinstance(engine, RetrieverQueryEngine)
                    mock_retrievers.get_retriever.assert_called_once_with("vector")
                    mock_factory.assert_called_once_with(mock_index, 10)
                    mock_get_llm.assert_called_once()

    def test_make_query_engine_custom_retriever_type(self):
        """Test creating query engine with custom retriever type."""
        mock_index = Mock(spec=VectorStoreIndex)

        with patch("rag.engine.engine.retrievers") as mock_retrievers:
            mock_factory = Mock()
            mock_factory.return_value = Mock(retriever=Mock(), postprocessors=[])
            mock_retrievers.get_retriever.return_value = mock_factory

            with patch("rag.engine.engine.get_llm") as mock_get_llm:
                mock_llm = Mock()
                mock_get_llm.return_value = mock_llm

                with patch("rag.engine.engine.get_response_synthesizer") as mock_synth:
                    mock_synth.return_value = Mock()
                    engine = make_query_engine(mock_index, k=5, retriever_type="bm25")

                    assert isinstance(engine, RetrieverQueryEngine)
                    mock_retrievers.get_retriever.assert_called_once_with("bm25")
                    mock_factory.assert_called_once_with(mock_index, 5)
                    mock_get_llm.assert_called_once()

    def test_make_query_engine_invalid_retriever_type(self):
        """Test that invalid retriever type raises ValueError."""
        mock_index = Mock(spec=VectorStoreIndex)

        with patch("rag.engine.engine.retrievers") as mock_retrievers:
            mock_retrievers.get_retriever.side_effect = ValueError(
                "Unknown retriever type"
            )

            with pytest.raises(ValueError, match="Unknown retriever type"):
                make_query_engine(mock_index, k=10, retriever_type="invalid")


class TestRetrieveDocumentsFromEngine:
    """Test retrieve_documents_from_engine function."""

    @pytest.mark.asyncio
    async def test_retrieve_documents_basic(self):
        """Test basic document retrieval without postprocessors."""
        # Create mock engine
        mock_engine = Mock(spec=RetrieverQueryEngine)
        mock_retriever = AsyncMock()

        # Create mock nodes
        mock_node1 = NodeWithScore(
            node=TextNode(
                text="test content 1",
                id_="id1",
                metadata={"name": "func1", "type": "function", "file_path": "file1.py"},
            ),
            score=0.9,
        )
        mock_node2 = NodeWithScore(
            node=TextNode(
                text="test content 2",
                id_="id2",
                metadata={"name": "func2", "type": "function", "file_path": "file2.py"},
            ),
            score=0.8,
        )

        mock_retriever.aretrieve.return_value = [mock_node1, mock_node2]
        mock_engine.retriever = mock_retriever
        mock_engine._node_postprocessors = None  # No postprocessors

        results = await retrieve_documents_from_engine(mock_engine, "test query", k=2)

        assert len(results) == 2
        assert results[0]["node_id"] == "id1"
        assert results[0]["element_name"] == "func1"
        assert results[0]["file_path"] == "file1.py"
        assert results[0]["score"] == 0.9
        assert "test content" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_retrieve_documents_with_postprocessors(self):
        """Test document retrieval with postprocessors applied."""
        # Create mock engine
        mock_engine = Mock(spec=RetrieverQueryEngine)
        mock_retriever = AsyncMock()

        # Create mock nodes
        mock_node1 = NodeWithScore(
            node=TextNode(
                text="content 1",
                id_="id1",
                metadata={"name": "func1", "type": "function", "file_path": "file1.py"},
            ),
            score=0.9,
        )
        mock_node2 = NodeWithScore(
            node=TextNode(
                text="content 2",
                id_="id2",
                metadata={"name": "func2", "type": "function", "file_path": "file2.py"},
            ),
            score=0.8,
        )

        mock_retriever.aretrieve.return_value = [mock_node1, mock_node2]
        mock_engine.retriever = mock_retriever

        # Mock postprocessor that reorders nodes
        mock_postprocessor = Mock()
        mock_postprocessor.postprocess_nodes.return_value = [
            mock_node2,
            mock_node1,
        ]  # Reversed
        mock_engine._node_postprocessors = [mock_postprocessor]

        results = await retrieve_documents_from_engine(mock_engine, "test query", k=2)

        # Postprocessor should be called
        mock_postprocessor.postprocess_nodes.assert_called_once()
        assert len(results) == 2
        # Results should be in postprocessor order
        assert results[0]["node_id"] == "id2"

    @pytest.mark.asyncio
    async def test_retrieve_documents_limits_k(self):
        """Test that retrieval respects k parameter."""
        mock_engine = Mock(spec=RetrieverQueryEngine)
        mock_retriever = AsyncMock()

        # Create 5 nodes but only request 2
        mock_nodes = [
            NodeWithScore(
                node=TextNode(
                    text=f"content {i}",
                    id_=f"id{i}",
                    metadata={
                        "name": f"func{i}",
                        "type": "function",
                        "file_path": f"file{i}.py",
                    },
                ),
                score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        mock_retriever.aretrieve.return_value = mock_nodes
        mock_engine.retriever = mock_retriever
        mock_engine._node_postprocessors = None

        results = await retrieve_documents_from_engine(mock_engine, "test query", k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_documents_handles_missing_metadata(self):
        """Test that retrieval handles missing metadata gracefully."""
        mock_engine = Mock(spec=RetrieverQueryEngine)
        mock_retriever = AsyncMock()

        # Node with minimal metadata
        mock_node = NodeWithScore(
            node=TextNode(text="content", id_="id1", metadata={}), score=0.9
        )

        mock_retriever.aretrieve.return_value = [mock_node]
        mock_engine.retriever = mock_retriever
        mock_engine._node_postprocessors = None

        results = await retrieve_documents_from_engine(mock_engine, "test query", k=1)

        assert len(results) == 1
        assert results[0]["element_name"] == "unknown"
        assert results[0]["element_type"] == "unknown"
        assert results[0]["file_path"] == "unknown"
