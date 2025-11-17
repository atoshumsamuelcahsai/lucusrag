"""
Tests for embedding_loader module - embedding generation and upsertion.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from llama_index.core import Document

from rag.ingestion.embedding_loader import populate_embeddings
from rag.db.graph_db import GraphDBManager
from rag.schemas.vector_config import VectorIndexConfig


class TestPopulateEmbeddings:
    """Test populate_embeddings function."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock GraphDBManager."""
        manager = AsyncMock(spec=GraphDBManager)
        manager.upsert_embeddings = AsyncMock(return_value=5)
        return manager

    @pytest.fixture
    def mock_embed_model(self):
        """Create a mock embedding model."""
        model = Mock()
        model.get_text_embedding = Mock(return_value=[0.1] * 1536)
        return model

    @pytest.fixture
    def vector_config(self):
        """Create a vector config."""
        return VectorIndexConfig(
            name="test_index",
            dimension=1536,
            node_label="CodeElement",
            vector_property="embedding",
            similarity_metric="cosine",
            neo4j_url="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(
                text="test content 1",
                metadata={"id": "node-1", "name": "func1", "type": "function"},
            ),
            Document(
                text="test content 2",
                metadata={"id": "node-2", "name": "func2", "type": "function"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_populate_embeddings_empty_documents(self, mock_db_manager):
        """Test with empty documents list."""
        result = await populate_embeddings(mock_db_manager, [])
        assert result == 0
        mock_db_manager.upsert_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_populate_embeddings_success(
        self, mock_db_manager, mock_embed_model, vector_config, sample_documents
    ):
        """Test successful embedding population."""
        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "text representation"

                result = await populate_embeddings(
                    mock_db_manager, sample_documents, vector_config
                )

                assert result == 5
                assert mock_db_manager.upsert_embeddings.call_count == 1
                call_args = mock_db_manager.upsert_embeddings.call_args
                payload = call_args[0][0]
                assert len(payload) == 2
                assert payload[0]["id"] == "node-1"
                assert payload[1]["id"] == "node-2"
                assert "vec" in payload[0]
                assert "text" in payload[0]
                assert "metadata" in payload[0]

    @pytest.mark.asyncio
    async def test_populate_embeddings_no_config(
        self, mock_db_manager, mock_embed_model, sample_documents
    ):
        """Test with no vector_config provided (should load from env)."""
        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            with patch(
                "rag.schemas.vector_config.get_vector_index_config"
            ) as mock_get_config:
                mock_config = VectorIndexConfig(
                    name="test",
                    dimension=1536,
                    node_label="CodeElement",
                    vector_property="embedding",
                    similarity_metric="cosine",
                    neo4j_url="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="test",
                )
                mock_get_config.return_value = mock_config
                with patch(
                    "rag.ingestion.embedding_loader.create_text_representation"
                ) as mock_create_text:
                    mock_create_text.return_value = "text representation"

                    result = await populate_embeddings(
                        mock_db_manager, sample_documents
                    )

                    assert result == 5
                    mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_populate_embeddings_no_embed_model(
        self, mock_db_manager, sample_documents
    ):
        """Test when embedding model is not configured."""
        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = None

            with pytest.raises(RuntimeError, match="Embedding model not configured"):
                await populate_embeddings(mock_db_manager, sample_documents)

    @pytest.mark.asyncio
    async def test_populate_embeddings_document_without_id(
        self, mock_db_manager, mock_embed_model, vector_config
    ):
        """Test documents without id are skipped."""
        documents = [
            Document(text="test", metadata={"name": "func1"}),  # No id
            Document(
                text="test2",
                metadata={"id": "node-2", "name": "func2"},
            ),
        ]

        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "text representation"

                await populate_embeddings(mock_db_manager, documents, vector_config)

                # Only one document should be processed
                call_args = mock_db_manager.upsert_embeddings.call_args
                payload = call_args[0][0]
                assert len(payload) == 1
                assert payload[0]["id"] == "node-2"

    @pytest.mark.asyncio
    async def test_populate_embeddings_embedding_failure(
        self, mock_db_manager, mock_embed_model, vector_config, sample_documents
    ):
        """Test when embedding generation fails for some documents."""
        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            # First call succeeds, second fails
            mock_embed_model.get_text_embedding.side_effect = [
                [0.1] * 1536,
                Exception("Embedding failed"),
            ]
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "text representation"

                await populate_embeddings(
                    mock_db_manager, sample_documents, vector_config
                )

                # Only successful embedding should be in payload
                call_args = mock_db_manager.upsert_embeddings.call_args
                payload = call_args[0][0]
                assert len(payload) == 1
                assert payload[0]["id"] == "node-1"

    @pytest.mark.asyncio
    async def test_populate_embeddings_all_fail(
        self, mock_db_manager, mock_embed_model, vector_config, sample_documents
    ):
        """Test when all embeddings fail."""
        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            mock_embed_model.get_text_embedding.side_effect = Exception("All failed")
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "text representation"

                result = await populate_embeddings(
                    mock_db_manager, sample_documents, vector_config
                )

                # Should return 0 and not call upsert
                assert result == 0
                mock_db_manager.upsert_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_populate_embeddings_payload_structure(
        self, mock_db_manager, mock_embed_model, vector_config
    ):
        """Test that payload has correct structure."""
        documents = [
            Document(
                text="test",
                metadata={
                    "id": "node-1",
                    "name": "func1",
                    "type": "function",
                    "file_path": "test.py",
                },
            )
        ]

        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "func1 function in test.py"

                await populate_embeddings(mock_db_manager, documents, vector_config)

                call_args = mock_db_manager.upsert_embeddings.call_args
                payload = call_args[0][0]
                assert len(payload) == 1
                item = payload[0]
                assert item["id"] == "node-1"
                assert item["text"] == "func1 function in test.py"
                assert isinstance(item["vec"], list)
                assert len(item["vec"]) == 1536
                assert item["metadata"]["name"] == "func1"
                assert item["metadata"]["type"] == "function"

    @pytest.mark.asyncio
    async def test_populate_embeddings_multiple_documents(
        self, mock_db_manager, mock_embed_model, vector_config
    ):
        """Test with multiple documents."""
        documents = [
            Document(
                text=f"content {i}",
                metadata={"id": f"node-{i}", "name": f"func{i}"},
            )
            for i in range(10)
        ]

        with patch("rag.ingestion.embedding_loader.Settings") as mock_settings:
            mock_settings.embed_model = mock_embed_model
            with patch(
                "rag.ingestion.embedding_loader.create_text_representation"
            ) as mock_create_text:
                mock_create_text.return_value = "text representation"

                result = await populate_embeddings(
                    mock_db_manager, documents, vector_config
                )

                assert result == 5  # Mock returns 5
                call_args = mock_db_manager.upsert_embeddings.call_args
                payload = call_args[0][0]
                assert len(payload) == 10
