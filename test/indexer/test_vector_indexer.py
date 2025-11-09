"""
Tests for the vector_indexer module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from llama_index.core import Settings
from rag.indexer.vector_indexer import (
    graph_configure_settings,
    create_vector_index_from_existing_nodes,
)
import rag.indexer.vector_indexer as vi
from rag.schemas.vector_config import VectorIndexConfig
from rag.indexer.vector_indexer import _graph_configure_settings_blocking


class TestGraphConfigureSettings:
    """Test suite for graph_configure_settings function."""

    @pytest.mark.skip(reason="LlamaIndex Settings validation - needs real objects")
    @pytest.mark.asyncio
    async def test_configure_settings_first_time(self):
        """Test settings configuration on first call."""
        # Reset global state

        vi._settings_configured = False

        with patch("rag.indexer.vector_indexer.get_embeddings") as mock_embed, patch(
            "rag.indexer.vector_indexer.get_llm"
        ) as mock_llm:

            mock_embed.return_value = Mock()
            mock_llm.return_value = Mock()

            await graph_configure_settings(
                num_output=512,
                context_window=2048,
                llm_provider="anthropic",
                embedding_provider="voyage",
            )

            # Verify LLM was configured
            mock_llm.assert_called_once_with("anthropic")

            # Verify embeddings were configured
            mock_embed.assert_called_once()

            # Verify Settings were updated
            assert Settings.num_output == 512
            assert Settings.context_window == 2048

    @pytest.mark.asyncio
    async def test_configure_settings_idempotent(self):
        """Test that settings are only configured once."""
        import rag.indexer.vector_indexer as vi

        vi._settings_configured = True  # Already configured

        with patch("rag.indexer.vector_indexer.get_embeddings") as mock_embed, patch(
            "rag.indexer.vector_indexer.get_llm"
        ) as mock_llm:

            await graph_configure_settings()

            # Should not call providers if already configured
            mock_embed.assert_not_called()
            mock_llm.assert_not_called()

    @pytest.mark.skip(reason="LlamaIndex Settings validation - needs real objects")
    @pytest.mark.asyncio
    async def test_configure_settings_thread_safe(self):
        """Test that concurrent calls don't race."""
        vi._settings_configured = False

        call_count = 0

        async def mock_setup():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work

        with patch("rag.indexer.vector_indexer.get_embeddings") as mock_embed, patch(
            "rag.indexer.vector_indexer.get_llm"
        ) as mock_llm:

            mock_embed.return_value = Mock()
            mock_llm.return_value = Mock()

            # Call multiple times concurrently
            await asyncio.gather(
                graph_configure_settings(),
                graph_configure_settings(),
                graph_configure_settings(),
            )

            # Should only configure once due to lock
            assert mock_llm.call_count == 1


class TestCreateVectorIndexFromExistingNodes:
    """Test suite for create_vector_index_from_existing_nodes function."""

    def test_create_vector_index_loads_config_from_env(self):
        """Test that config is loaded from env if not provided."""
        with patch(
            "rag.indexer.vector_indexer._graph_configure_settings_blocking"
        ), patch(
            "rag.indexer.vector_indexer.VectorIndexConfig.from_env"
        ) as mock_from_env, patch(
            "rag.indexer.vector_indexer.Neo4jVectorStore"
        ), patch(
            "rag.indexer.vector_indexer.Neo4jGraphStore"
        ), patch(
            "rag.indexer.vector_indexer.VectorStoreIndex.from_vector_store"
        ) as mock_index:

            mock_config = Mock()
            mock_config.neo4j_url = "bolt://localhost:7687"
            mock_config.neo4j_user = "neo4j"
            mock_config.neo4j_password = "password"
            mock_config.name = "test"
            mock_config.node_label = "Test"
            mock_config.dimension = 1536
            mock_config.similarity_metric = "cosine"
            mock_config.vector_property = "embedding"

            mock_from_env.return_value = mock_config
            mock_index.return_value = Mock()

            create_vector_index_from_existing_nodes(vector_config=None)

            # Should call from_env when config is None
            mock_from_env.assert_called_once()

    def test_create_vector_index_creates_stores(self):
        """Test that vector and graph stores are created correctly."""
        config = VectorIndexConfig(
            name="test_index",
            dimension=1536,
            node_label="TestNode",
            vector_property="embedding",
            similarity_metric="cosine",
            neo4j_url="bolt://test:7687",
            neo4j_user="testuser",
            neo4j_password="testpass",
        )

        with patch(
            "rag.indexer.vector_indexer._graph_configure_settings_blocking"
        ), patch("rag.indexer.vector_indexer.Neo4jVectorStore") as mock_vector, patch(
            "rag.indexer.vector_indexer.Neo4jGraphStore"
        ) as mock_graph, patch(
            "rag.indexer.vector_indexer.VectorStoreIndex.from_vector_store"
        ) as mock_index:

            mock_index.return_value = Mock()

            create_vector_index_from_existing_nodes(config)

            # Verify Neo4jVectorStore was created with correct params
            mock_vector.assert_called_once()
            call_kwargs = mock_vector.call_args.kwargs
            assert call_kwargs["url"] == "bolt://test:7687"
            assert call_kwargs["username"] == "testuser"
            assert call_kwargs["password"] == "testpass"
            assert call_kwargs["index_name"] == "test_index"
            assert call_kwargs["embedding_dimension"] == 1536

            # Verify Neo4jGraphStore was created
            mock_graph.assert_called_once()
            graph_kwargs = mock_graph.call_args.kwargs
            assert graph_kwargs["url"] == "bolt://test:7687"
            assert "INHERITS_FROM" in graph_kwargs["edge_labels"]
            assert "CALLS" in graph_kwargs["edge_labels"]
            assert "DEPENDS_ON" in graph_kwargs["edge_labels"]

            # Verify VectorStoreIndex.from_vector_store was called
            mock_index.assert_called_once()
            index_call_args = mock_index.call_args.kwargs
            assert "vector_store" in index_call_args
            assert "graph_store" in index_call_args


class TestBlockingConfigureSettings:
    """Test suite for _graph_configure_settings_blocking."""

    def test_blocking_outside_event_loop(self):
        """Test that blocking version works outside async context."""

        with patch(
            "rag.indexer.vector_indexer.asyncio.get_running_loop"
        ) as mock_loop, patch("rag.indexer.vector_indexer.asyncio.run") as mock_run:

            # Simulate no running loop
            mock_loop.side_effect = RuntimeError("No running loop")

            _graph_configure_settings_blocking()

            # Should call asyncio.run
            mock_run.assert_called_once()

    # def test_blocking_inside_event_loop(self):
    #     """Test that blocking version logs when inside async context."""
    #     with patch(
    #         "rag.indexer.vector_indexer.asyncio.get_running_loop"
    #     ) as mock_loop, patch(
    #         "rag.indexer.vector_indexer.asyncio.run"
    #     ) as mock_run, patch(
    #         "rag.indexer.vector_indexer.logger"
    #     ) as mock_logger:

    #         # Simulate running loop exists
    #         mock_loop.return_value = Mock()

    #         _graph_configure_settings_blocking()

    #         # Should NOT call asyncio.run
    #         mock_run.assert_not_called()

    #         # Should log debug message
    #         mock_logger.debug.assert_called_once()


# Integration test placeholder
@pytest.mark.skip(reason="Requires Neo4j, API keys - run manually for E2E testing")
def test_full_vector_index_creation():
    """
    Full integration test for vector index creation.

    NOTE: Requires:
    - Running Neo4j instance
    - Valid API keys in .env
    - Sample AST cache directory

    Run with: pytest -v -s test/indexer/test_vector_indexer.py -k full_vector
    """
    from rag.indexer.vector_indexer import create_vector_index_from_existing_nodes

    index = create_vector_index_from_existing_nodes()

    assert index is not None
    assert hasattr(index, "as_retriever")
