"""
Tests for the vector_indexer module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from llama_index.core import Settings, Document
from llama_index.core.schema import TextNode
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

            # Set environment variables for test
            import os

            os.environ["LLM_PROVIDER"] = "anthropic"
            os.environ["EMBEDDING_PROVIDER"] = "voyage"
            os.environ["LLM_MAX_OUTPUT_TOKENS"] = "512"
            os.environ["LLM_CONTEXT_WINDOW"] = "2048"

            await graph_configure_settings()

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

    @pytest.mark.asyncio
    async def test_create_vector_index_loads_config_from_env(self):
        """Test that config is loaded from env if not provided."""
        # Create mock documents
        mock_docs = [Document(text="test document", metadata={"name": "test"})]
        # Create proper TextNode objects
        mock_nodes = [TextNode(text="test node", id_="test-id-1")]

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
        ) as mock_index, patch(
            "rag.indexer.vector_indexer.parse_documents_to_nodes"
        ) as mock_parse, patch(
            "rag.indexer.vector_indexer.SimpleDocumentStore"
        ) as mock_docstore_class, patch(
            "rag.indexer.vector_indexer.logger"
        ), patch(
            "neo4j.AsyncGraphDatabase"
        ) as mock_graph_db:

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
            mock_index_instance = Mock()
            # Make docstore.docs return a dict-like object with len
            mock_docstore_attr = MagicMock()
            mock_docstore_attr.docs = {"test-id-1": mock_nodes[0]}
            mock_index_instance.docstore = mock_docstore_attr
            mock_index.return_value = mock_index_instance
            mock_parse.return_value = mock_nodes

            # Create a proper mock docstore instance with docs attribute that supports len()
            mock_docstore_instance = MagicMock()
            mock_docstore_instance.docs = {
                "test-id-1": mock_nodes[0]
            }  # Real dict that supports len()
            mock_docstore_class.return_value = mock_docstore_instance

            # Mock Neo4j async driver to avoid actual connection attempts
            mock_driver = AsyncMock()
            mock_session = AsyncMock()
            mock_result = AsyncMock()

            # Make result async iterable but empty (no additional nodes from Neo4j)
            async def async_iter():
                return
                yield  # Make it an async generator

            mock_result.__aiter__ = AsyncMock(return_value=async_iter())
            mock_session.run = AsyncMock(return_value=mock_result)
            # Make session work as an async context manager
            mock_driver.session = AsyncMock(return_value=mock_session)
            mock_driver.close = AsyncMock()
            mock_graph_db.driver.return_value = mock_driver

            await create_vector_index_from_existing_nodes(
                vector_config=None, docs=mock_docs
            )

            # Should call from_env when config is None
            mock_from_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_vector_index_creates_stores(self):
        """Test that vector and graph stores are created correctly."""
        # Create mock documents
        mock_docs = [Document(text="test document", metadata={"name": "test"})]
        # Create proper TextNode objects
        mock_nodes = [TextNode(text="test node", id_="test-id-1")]

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
        ) as mock_index, patch(
            "rag.indexer.vector_indexer.parse_documents_to_nodes"
        ) as mock_parse, patch(
            "rag.indexer.vector_indexer.SimpleDocumentStore"
        ) as mock_docstore, patch(
            "rag.indexer.vector_indexer.logger"
        ), patch(
            "neo4j.AsyncGraphDatabase"
        ) as mock_graph_db:

            mock_index_instance = Mock()
            # Make docstore.docs return a dict-like object with len
            mock_docstore_attr = MagicMock()
            mock_docstore_attr.docs = {"test-id-1": mock_nodes[0]}
            mock_index_instance.docstore = mock_docstore_attr
            mock_index.return_value = mock_index_instance
            mock_parse.return_value = mock_nodes

            # Create a proper mock docstore instance with docs attribute that supports len()
            mock_docstore_instance = MagicMock()
            mock_docstore_instance.docs = {
                "test-id-1": mock_nodes[0]
            }  # Real dict that supports len()
            mock_docstore.return_value = mock_docstore_instance

            # Mock Neo4j async driver to avoid actual connection attempts
            mock_driver = AsyncMock()
            mock_session = AsyncMock()
            mock_result = AsyncMock()

            # Make result async iterable but empty (no additional nodes from Neo4j)
            async def async_iter():
                return
                yield  # Make it an async generator

            mock_result.__aiter__ = AsyncMock(return_value=async_iter())
            mock_session.run = AsyncMock(return_value=mock_result)
            # Make session work as an async context manager
            mock_driver.session = AsyncMock(return_value=mock_session)
            mock_driver.close = AsyncMock()
            mock_graph_db.driver.return_value = mock_driver

            await create_vector_index_from_existing_nodes(config, docs=mock_docs)

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

            # Verify VectorStoreIndex.from_vector_store was called with vector_store only
            mock_index.assert_called_once()
            index_call_args = mock_index.call_args.kwargs
            assert "vector_store" in index_call_args
            assert index_call_args.get("show_progress") is False

            # Verify docstore was created and nodes were added
            mock_docstore.assert_called_once()
            mock_docstore_instance.add_documents.assert_called_once()

            # Verify parse_documents_to_nodes was called
            mock_parse.assert_called_once_with(mock_docs)


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
@pytest.mark.asyncio
async def test_full_vector_index_creation():
    """
    Full integration test for vector index creation.

    NOTE: Requires:
    - Running Neo4j instance
    - Valid API keys in .env
    - Sample AST cache directory

    Run with: pytest -v -s test/indexer/test_vector_indexer.py -k full_vector
    """
    from rag.indexer.vector_indexer import create_vector_index_from_existing_nodes

    index = await create_vector_index_from_existing_nodes()

    assert index is not None
    assert hasattr(index, "as_retriever")
