"""
Tests for timed_query_engine module - query execution with timing.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.response_synthesizers import BaseSynthesizer

from rag.engine.timed_query_engine import TimedQueryEngine, TimedRetriever
from rag.engine.retrievers import clear_timing_info


class TestTimedRetriever:
    """Test TimedRetriever wrapper."""

    @pytest.mark.asyncio
    async def test_timed_retriever_aretrieve(self):
        """Test async retrieval with timing."""
        mock_retriever = AsyncMock()
        mock_node = NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)
        mock_retriever.aretrieve.return_value = [mock_node]

        timed_retriever = TimedRetriever(mock_retriever)
        result = await timed_retriever.aretrieve(QueryBundle(query_str="test"))

        assert len(result) == 1
        assert result[0] == mock_node
        timing_info = timed_retriever.get_timing_info()
        assert "retrieval" in timing_info
        assert timing_info["retrieval"] > 0

    def test_timed_retriever_retrieve(self):
        """Test sync retrieval with timing."""
        mock_retriever = Mock()
        mock_node = NodeWithScore(node=TextNode(text="test", id_="id1"), score=0.9)
        mock_retriever.retrieve.return_value = [mock_node]

        timed_retriever = TimedRetriever(mock_retriever)
        result = timed_retriever.retrieve(QueryBundle(query_str="test"))

        assert len(result) == 1
        timing_info = timed_retriever.get_timing_info()
        assert "retrieval" in timing_info


class TestTimedQueryEngine:
    """Test TimedQueryEngine wrapper."""

    def _create_mock_engine(self):
        """Helper to create mock query engine."""
        mock_engine = Mock(spec=RetrieverQueryEngine)
        mock_retriever = AsyncMock()
        mock_engine.retriever = mock_retriever
        mock_engine._node_postprocessors = []
        return mock_engine, mock_retriever

    def _create_mock_synthesizer(self):
        """Helper to create mock response synthesizer."""
        mock_synthesizer = AsyncMock(spec=BaseSynthesizer)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test response")
        mock_synthesizer.asynthesize = AsyncMock(return_value=mock_response)
        return mock_synthesizer, mock_response

    @pytest.mark.asyncio
    async def test_timed_query_engine_basic_query(self):
        """Test basic query execution with timing."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create node with enough text to avoid index errors
        mock_node = NodeWithScore(
            node=TextNode(text="test content " * 10, id_="id1"), score=0.9
        )
        mock_retriever.aretrieve.return_value = [mock_node]

        # Set synthesizer
        mock_engine.response_synthesizer = mock_synthesizer

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        result = await timed_engine.aquery("test query")

        assert str(result) == "Test response"
        mock_retriever.aretrieve.assert_called_once()
        mock_synthesizer.asynthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_timed_query_engine_with_reranking(self):
        """Test query execution with reranking postprocessor."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create nodes with enough text
        mock_node1 = NodeWithScore(
            node=TextNode(text="content1 " * 10, id_="id1"), score=0.9
        )
        mock_node2 = NodeWithScore(
            node=TextNode(text="content2 " * 10, id_="id2"), score=0.8
        )
        mock_retriever.aretrieve.return_value = [mock_node1, mock_node2]

        # Create mock postprocessor (reranker)
        mock_postprocessor = Mock()
        mock_postprocessor.postprocess_nodes.return_value = [mock_node2, mock_node1]

        mock_engine._node_postprocessors = [mock_postprocessor]
        mock_engine.response_synthesizer = mock_synthesizer

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        await timed_engine.aquery("test query")

        mock_postprocessor.postprocess_nodes.assert_called_once()
        mock_synthesizer.asynthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_timed_query_engine_with_adaptive_k(self):
        """Test query execution with adaptive K selection."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create multiple nodes for adaptive K
        mock_nodes = [
            NodeWithScore(
                node=TextNode(text=f"content{i}", id_=f"id{i}"), score=0.9 - i * 0.1
            )
            for i in range(5)
        ]
        mock_retriever.aretrieve.return_value = mock_nodes

        mock_engine._node_postprocessors = []
        mock_engine.response_synthesizer = mock_synthesizer

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        await timed_engine.aquery("test query")

        # Adaptive K should select some subset
        call_args = mock_synthesizer.asynthesize.call_args
        nodes_passed = call_args[1]["nodes"]
        # Should have selected some nodes (adaptive K will filter)
        assert len(nodes_passed) <= len(mock_nodes)

    @pytest.mark.asyncio
    async def test_timed_query_engine_timing_info(self):
        """Test that timing information is tracked."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create node with enough text
        mock_node = NodeWithScore(
            node=TextNode(text="test content " * 10, id_="id1"), score=0.9
        )
        mock_retriever.aretrieve.return_value = [mock_node]

        mock_engine._node_postprocessors = []
        mock_engine.response_synthesizer = mock_synthesizer

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        with patch("rag.engine.timed_query_engine.logger") as mock_logger:
            await timed_engine.aquery("test query")

            # Should log timing information
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_timed_query_engine_handles_exception(self):
        """Test that exceptions are handled and logged."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_retriever.aretrieve.side_effect = Exception("Retrieval failed")

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        with pytest.raises(Exception, match="Retrieval failed"):
            await timed_engine.aquery("test query")

    @pytest.mark.asyncio
    async def test_timed_query_engine_fallback_synthesizer(self):
        """Test fallback when synthesizer is not directly accessible."""
        mock_engine, mock_retriever = self._create_mock_engine()
        # Create node with enough text to generate cumulative_masses
        mock_node = NodeWithScore(
            node=TextNode(text="test content " * 20, id_="id1"), score=0.9
        )
        mock_retriever.aretrieve.return_value = [mock_node]

        # Don't set response_synthesizer directly
        mock_engine.response_synthesizer = None
        mock_engine._node_postprocessors = []
        # Also clear _synthesizer and synthesizer attributes
        mock_engine._synthesizer = None
        mock_engine.synthesizer = None

        # Mock the engine's aquery method as fallback
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Fallback response")
        mock_engine.aquery = AsyncMock(return_value=mock_response)

        timed_engine = TimedQueryEngine(mock_engine)
        clear_timing_info()

        result = await timed_engine.aquery("test query")

        # Should use fallback
        assert str(result) == "Fallback response"

    def test_timed_query_engine_has_graph_expansion(self):
        """Test graph expansion detection."""
        mock_engine = Mock(spec=RetrieverQueryEngine)

        # Mock retriever with graph expansion in name
        class GraphExpandedRetriever:
            pass

        mock_engine.retriever = GraphExpandedRetriever()
        timed_engine = TimedQueryEngine(mock_engine)

        assert timed_engine._has_graph_expansion() is True

    def test_timed_query_engine_has_rrf(self):
        """Test RRF detection."""
        mock_engine = Mock(spec=RetrieverQueryEngine)

        # Mock retriever with RRF in name
        class HybridRRFRetriever:
            pass

        mock_engine.retriever = HybridRRFRetriever()
        timed_engine = TimedQueryEngine(mock_engine)

        assert timed_engine._has_rrf() is True

    def test_timed_query_engine_is_vector_retriever(self):
        """Test vector retriever detection."""
        mock_engine = Mock(spec=RetrieverQueryEngine)

        # Mock vector retriever
        class VectorIndexRetriever:
            pass

        mock_engine.retriever = VectorIndexRetriever()
        timed_engine = TimedQueryEngine(mock_engine)

        assert timed_engine._is_vector_retriever() is True

    @pytest.mark.asyncio
    async def test_timed_query_engine_with_graph_expansion_timing(self):
        """Test timing includes graph expansion."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create node with enough text
        mock_node = NodeWithScore(
            node=TextNode(text="test content " * 10, id_="id1"), score=0.9
        )
        mock_retriever.aretrieve.return_value = [mock_node]

        mock_engine._node_postprocessors = []
        mock_engine.response_synthesizer = mock_synthesizer

        # Set up timing info to simulate graph expansion
        clear_timing_info()
        with patch("rag.engine.timed_query_engine.get_timing_info") as mock_get_timing:
            mock_get_timing.return_value = {"graph_expansion": 0.05}

            timed_engine = TimedQueryEngine(mock_engine)
            result = await timed_engine.aquery("test query")

            # Should handle graph expansion timing
            assert result is not None

    @pytest.mark.asyncio
    async def test_timed_query_engine_with_rrf_timing(self):
        """Test timing includes RRF fusion."""
        mock_engine, mock_retriever = self._create_mock_engine()
        mock_synthesizer, mock_response = self._create_mock_synthesizer()

        # Create node with enough text
        mock_node = NodeWithScore(
            node=TextNode(text="test content " * 10, id_="id1"), score=0.9
        )
        mock_retriever.aretrieve.return_value = [mock_node]

        mock_engine._node_postprocessors = []
        mock_engine.response_synthesizer = mock_synthesizer

        # Set up timing info to simulate RRF fusion
        clear_timing_info()
        with patch("rag.engine.timed_query_engine.get_timing_info") as mock_get_timing:
            mock_get_timing.return_value = {"rrf_fusion": 0.02}

            timed_engine = TimedQueryEngine(mock_engine)
            result = await timed_engine.aquery("test query")

            # Should handle RRF timing
            assert result is not None
