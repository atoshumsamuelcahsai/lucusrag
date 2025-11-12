from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from llama_index.core import Document

from rag.ingestion.data_loader import (
    _process_ast_files,
    _create_schema,
    _build_code_db_graph,
    _check_db_populated,
    process_code_files,
)
from rag.db.graph_db import GraphDBManager
from rag.schemas import CodeElement
from rag.schemas.vector_config import VectorIndexConfig, get_vector_index_config


@pytest.fixture
def vector_config():
    """Fixture for vector index configuration."""
    return VectorIndexConfig(
        name="test_index",
        dimension=1536,
        node_label="TestNode",
        vector_property="embedding",
        similarity_metric="cosine",
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
    )


@pytest.fixture
def mock_db_manager():
    """Fixture for mocked GraphDBManager."""
    manager = MagicMock(spec=GraphDBManager)
    manager.driver = MagicMock()
    manager.driver.session = MagicMock()
    manager.create_schema = Mock()
    manager.create_node = Mock()
    manager.create_relationships = Mock()
    manager.close = Mock()
    return manager


@pytest.fixture
def sample_code_element():
    """Fixture for sample CodeElement."""
    return CodeElement(
        type="function",
        name="test_function",
        docstring="Test function",
        code="def test_function(): pass",
        file_path="/test/path.py",
        parameters=[{"name": "x", "type": "int"}],
        return_type="str",
    )


@pytest.fixture
def sample_ast_data():
    """Fixture for sample AST JSON data."""
    return {
        "type": "function",
        "name": "sample_function",
        "docstring": "Sample function",
        "code": "def sample_function(x): return x",
        "file_path": "/sample/path.py",
        "parameters": [{"name": "x", "type": "int"}],
        "return_type": "int",
        "decorators": ["@decorator"],
        "dependencies": ["module.dep"],
        "base_classes": [],
        "calls": ["helper"],
        "assignments": [],
    }


class TestGetVectorIndexConfig:
    """Tests for get_vector_index_config function."""

    @patch.dict(
        "os.environ",
        {
            "VECTOR_INDEX_NAME": "test_index",
            "VECTOR_DIMENSION": "512",
            "NODE_LABEL": "TestLabel",
            "VECTOR_PROPERTY": "vec",
            "SIMILARITY_METRIC": "euclidean",
            "NEO4J_URL": "bolt://test:7687",
            "NEO4J_USER": "test_user",
            "NEO4J_PASSWORD": "test_pass",
        },
    )
    @pytest.mark.asyncio
    async def test_get_vector_index_config_from_env(self) -> None:
        """Test getting config from environment variables."""
        config = get_vector_index_config()

        assert config.name == "test_index"
        assert config.dimension == 512
        assert config.node_label == "TestLabel"
        assert config.vector_property == "vec"
        assert config.similarity_metric == "euclidean"


class TestProcessASTFiles:
    """Tests for _process_ast_files function."""

    def test_process_ast_files_empty_directory(self, tmp_path):
        """Test processing an empty directory."""
        documents, code_infos = _process_ast_files(tmp_path)

        assert len(documents) == 0
        assert len(code_infos) == 0

    @patch("rag.ingestion.data_loader.process_code_element")
    def test_process_ast_files_with_valid_json(
        self, mock_process, tmp_path, sample_ast_data
    ):
        """Test processing directory with valid JSON files."""
        # Create results_* subdirectory (required by _process_ast_files)
        results_dir = tmp_path / "results_test"
        results_dir.mkdir()

        # Create test JSON file in results_* directory
        json_file = results_dir / "test.json"
        json_file.write_text(json.dumps(sample_ast_data))

        # Mock process_code_element to return a Document
        mock_doc = Document(text="test", metadata={"name": "test"})
        mock_process.return_value = mock_doc

        documents, code_infos = _process_ast_files(tmp_path)

        assert len(documents) == 1
        assert len(code_infos) == 1
        assert documents[0] == mock_doc
        assert code_infos[0].name == "sample_function"
        mock_process.assert_called_once()

    @patch("rag.ingestion.data_loader.process_code_element")
    def test_process_ast_files_multiple_files(self, mock_process, tmp_path):
        """Test processing multiple JSON files."""
        # Create results_* subdirectory (required by _process_ast_files)
        results_dir = tmp_path / "results_test"
        results_dir.mkdir()

        # Create multiple JSON files in results_* directory
        for i in range(3):
            data = {
                "type": "function",
                "name": f"function_{i}",
                "docstring": f"Function {i}",
                "code": f"def function_{i}(): pass",
                "file_path": f"/test/file_{i}.py",
            }
            json_file = results_dir / f"test_{i}.json"
            json_file.write_text(json.dumps(data))

        mock_process.return_value = Document(text="test", metadata={})

        documents, code_infos = _process_ast_files(tmp_path)

        assert len(documents) == 3
        assert len(code_infos) == 3
        assert mock_process.call_count == 3

    @patch("rag.ingestion.data_loader.process_code_element")
    def test_process_ast_files_with_invalid_json(self, mock_process, tmp_path):
        """Test processing directory with invalid JSON (should skip and continue)."""
        # Create results_* subdirectory (required by _process_ast_files)
        results_dir = tmp_path / "results_test"
        results_dir.mkdir()

        # Create invalid JSON file
        bad_file = results_dir / "bad.json"
        bad_file.write_text("{ invalid json }")

        # Create valid JSON file
        good_data = {
            "type": "function",
            "name": "good_function",
            "docstring": "Good function",
            "code": "def good_function(): pass",
            "file_path": "/test/good.py",
        }
        good_file = results_dir / "good.json"
        good_file.write_text(json.dumps(good_data))

        mock_process.return_value = Document(text="test", metadata={})

        documents, code_infos = _process_ast_files(tmp_path)

        # Should process only the valid file
        assert len(documents) == 1
        assert len(code_infos) == 1
        assert code_infos[0].name == "good_function"

    @patch("rag.ingestion.data_loader.process_code_element")
    def test_process_ast_files_sets_defaults(self, mock_process, tmp_path):
        """Test that missing optional fields get default values."""
        # Create results_* subdirectory (required by _process_ast_files)
        results_dir = tmp_path / "results_test"
        results_dir.mkdir()

        # Create JSON with minimal fields
        minimal_data = {
            "type": "function",
            "name": "minimal_function",
            "docstring": "Minimal",
            "code": "def minimal_function(): pass",
        }
        json_file = results_dir / "minimal.json"
        json_file.write_text(json.dumps(minimal_data))

        mock_process.return_value = Document(text="test", metadata={})

        documents, code_infos = _process_ast_files(tmp_path)

        assert len(code_infos) == 1
        code_info = code_infos[0]
        # Check that defaults were set
        assert code_info.parameters == []
        assert code_info.dependencies == []
        assert code_info.base_classes == []
        assert code_info.calls == []


class TestCreateSchema:
    """Tests for _create_schema function."""

    def test_create_schema_success(self, mock_db_manager, vector_config):
        """Test successful schema creation."""
        _create_schema(mock_db_manager, vector_config)

        mock_db_manager.create_schema.assert_called_once_with(vector_config)

    def test_create_schema_failure(self, mock_db_manager, vector_config):
        """Test schema creation failure propagates exception."""
        mock_db_manager.create_schema.side_effect = Exception("Schema error")

        with pytest.raises(Exception, match="Schema error"):
            _create_schema(mock_db_manager, vector_config)


class TestBuildCodeDBGraph:
    """Tests for _build_code_db_graph function."""

    def test_build_code_db_graph_empty_list(self, mock_db_manager, vector_config):
        """Test building database graph with empty list."""
        _build_code_db_graph(mock_db_manager, [], vector_config)

        mock_db_manager.create_schema.assert_called_once()
        mock_db_manager.create_node.assert_not_called()
        mock_db_manager.create_relationships.assert_not_called()

    def test_build_code_db_graph_with_code_elements(
        self, mock_db_manager, vector_config, sample_code_element
    ):
        """Test building database graph with code elements."""
        code_infos = [sample_code_element]

        _build_code_db_graph(mock_db_manager, code_infos, vector_config)

        mock_db_manager.create_schema.assert_called_once()
        mock_db_manager.create_node.assert_called_once_with(
            sample_code_element, vector_config
        )
        mock_db_manager.create_relationships.assert_called_once_with(
            sample_code_element, vector_config
        )

    @pytest.mark.asyncio
    async def test_build_code_db_graph_continues_on_node_error(
        self, mock_db_manager, vector_config, sample_code_element
    ):
        """Test that building continues when node creation fails."""
        code_element = CodeElement(
            type="class",
            name="TestClass",
            docstring="Test",
            code="class TestClass: pass",
            file_path="/test.py",
        )
        code_infos = [sample_code_element, code_element]

        # First call fails, second succeeds
        mock_db_manager.create_node.side_effect = [Exception("Node error"), None]

        _build_code_db_graph(mock_db_manager, code_infos, vector_config)

        # Should attempt both
        assert mock_db_manager.create_node.call_count == 2
        # Should still try to create relationships for the second one
        assert mock_db_manager.create_relationships.call_count == 2

    def test_build_code_db_graph_continues_on_relationship_error(
        self, mock_db_manager, vector_config, sample_code_element
    ):
        """Test that build database graph continues when relationship creation fails."""
        code_element = CodeElement(
            type="class",
            name="TestClass",
            docstring="Test",
            code="class TestClass: pass",
            file_path="/test.py",
        )
        code_infos = [sample_code_element, code_element]

        # First relationship fails, second succeeds
        mock_db_manager.create_relationships.side_effect = [
            Exception("Relationship error"),
            None,
        ]

        _build_code_db_graph(mock_db_manager, code_infos, vector_config)

        assert mock_db_manager.create_relationships.call_count == 2


class TestCheckDBBuilt:
    """Tests for _check_db_populated function (checks if graph is built)."""

    def test_check_db_built_true(self, mock_db_manager, vector_config):
        """Test checking if database graph is built."""
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.get.return_value = True
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_db_manager.driver.session.return_value = mock_session

        result = _check_db_populated(mock_db_manager, vector_config)

        assert result is True
        mock_session.run.assert_called_once()

    def test_check_db_built_false(self, mock_db_manager, vector_config):
        """Test checking if database graph is not built (empty)."""
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.get.return_value = False
        mock_session.run.return_value.single.return_value = mock_record
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_db_manager.driver.session.return_value = mock_session

        result = _check_db_populated(mock_db_manager, vector_config)

        assert result is False

    def test_check_db_built_error(self, mock_db_manager, vector_config):
        """Test error handling when checking database."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Connection error")
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_db_manager.driver.session.return_value = mock_session

        with pytest.raises(
            RuntimeError, match="Error checking Neo4j population status"
        ):
            _check_db_populated(mock_db_manager, vector_config)


class TestProcessCodeFiles:
    """Tests for process_code_files main function."""

    def test_process_code_files_directory_not_exists(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            process_code_files("/nonexistent/path")

    def test_process_code_files_path_is_file(self, tmp_path):
        """Test error when path is a file, not a directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(FileNotFoundError, match="is not a directory"):
            process_code_files(str(test_file))

    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader._process_ast_files")
    @patch("rag.ingestion.data_loader._build_code_db_graph")
    @patch("rag.ingestion.data_loader.populate_embeddings")
    @patch("rag.ingestion.data_loader.GraphDBManager")
    @patch("rag.ingestion.data_loader.get_vector_index_config")
    def test_process_code_files_db_empty(
        self,
        mock_get_config,
        mock_db_class,
        mock_populate_embeddings,
        mock_build_graph,
        mock_process_files,
        mock_check_db,
        tmp_path,
        vector_config,
        sample_code_element,
    ):
        """Test processing when database is empty (should build graph)."""
        mock_get_config.return_value = vector_config
        mock_db_manager = MagicMock()
        mock_db_class.return_value = mock_db_manager
        mock_check_db.return_value = False  # DB graph not built

        mock_doc = Document(text="test", metadata={})
        mock_process_files.return_value = ([mock_doc], [sample_code_element])

        result = process_code_files(str(tmp_path))

        assert len(result) == 1  # Returns list of documents
        assert result[0] == mock_doc
        mock_check_db.assert_called_once()
        mock_build_graph.assert_called_once()
        # Check that populate_embeddings was called with correct arguments
        mock_populate_embeddings.assert_called_once()
        call_args = mock_populate_embeddings.call_args
        assert call_args[0][0] == mock_db_manager
        assert len(call_args[0][1]) == 1
        assert call_args[0][1][0] == mock_doc
        # Compare vector_config fields instead of object identity
        # The config passed might be a different instance but should have same values
        actual_config = call_args[0][2]
        assert isinstance(actual_config, VectorIndexConfig)
        assert actual_config.name == vector_config.name
        assert actual_config.dimension == vector_config.dimension
        assert actual_config.node_label == vector_config.node_label
        assert actual_config.vector_property == vector_config.vector_property
        assert actual_config.similarity_metric == vector_config.similarity_metric
        assert actual_config.neo4j_url == vector_config.neo4j_url
        assert actual_config.neo4j_user == vector_config.neo4j_user
        assert actual_config.neo4j_password == vector_config.neo4j_password
        mock_db_manager.close.assert_called_once()

    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader._process_ast_files")
    @patch("rag.ingestion.data_loader._build_code_db_graph")
    @patch("rag.ingestion.data_loader.populate_embeddings")
    @patch("rag.ingestion.data_loader.GraphDBManager")
    @patch("rag.ingestion.data_loader.get_vector_index_config")
    @pytest.mark.asyncio
    async def test_process_code_files_db_built(
        self,
        mock_get_config,
        mock_db_class,
        mock_populate_embeddings,
        mock_build_graph,
        mock_process_files,
        mock_check_db,
        tmp_path,
        vector_config,
        sample_code_element,
    ):
        """Test processing when database graph is already built (should skip)."""
        mock_get_config.return_value = vector_config
        mock_db_manager = MagicMock()
        mock_db_class.return_value = mock_db_manager
        mock_check_db.return_value = True  # DB graph is built

        mock_doc = Document(text="test", metadata={})
        mock_process_files.return_value = ([mock_doc], [sample_code_element])

        result = process_code_files(str(tmp_path))

        assert len(result) == 1  # Returns list of documents
        assert result[0] == mock_doc
        mock_check_db.assert_called_once()
        mock_build_graph.assert_not_called()  # Should not build graph
        mock_populate_embeddings.assert_not_called()  # Should not populate embeddings
        mock_db_manager.close.assert_called_once()

    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader._process_ast_files")
    def test_process_code_files_with_dependency_injection(
        self,
        mock_process_files,
        mock_check_db,
        tmp_path,
        mock_db_manager,
        vector_config,
        sample_code_element,
    ):
        """Test process_code_files with injected dependencies."""
        mock_check_db.return_value = True
        mock_doc = Document(text="test", metadata={})
        mock_process_files.return_value = ([mock_doc], [sample_code_element])

        result = process_code_files(
            str(tmp_path), db_manager=mock_db_manager, vector_config=vector_config
        )

        assert len(result) == 1  # Returns list of documents
        assert result[0] == mock_doc
        mock_db_manager.close.assert_called_once()

    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader._process_ast_files")
    @patch("rag.ingestion.data_loader.GraphDBManager")
    @patch("rag.ingestion.data_loader.get_vector_index_config")
    def test_process_code_files_closes_manager_on_error(
        self,
        mock_get_config,
        mock_db_class,
        mock_process_files,
        mock_check_db,
        tmp_path,
        vector_config,
    ):
        """Test that database manager is closed even when error occurs."""
        mock_get_config.return_value = vector_config
        mock_db_manager = MagicMock()
        mock_db_class.return_value = mock_db_manager
        mock_check_db.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            process_code_files(str(tmp_path))

        # Should still close the manager
        mock_db_manager.close.assert_called_once()

    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader._process_ast_files")
    @patch("rag.ingestion.data_loader.GraphDBManager")
    @patch("rag.ingestion.data_loader.get_vector_index_config")
    def test_process_code_files_returns_empty_on_no_files(
        self,
        mock_get_config,
        mock_db_class,
        mock_process_files,
        mock_check_db,
        tmp_path,
        vector_config,
    ):
        """Test processing directory with no JSON files."""
        mock_get_config.return_value = vector_config
        mock_db_manager = MagicMock()
        mock_db_class.return_value = mock_db_manager
        mock_check_db.return_value = True
        mock_process_files.return_value = ([], [])  # No files

        result = process_code_files(str(tmp_path))

        assert result == []  # Returns empty list of documents
        mock_db_manager.close.assert_called_once()


class TestProcessCodeFilesIntegration:
    """Integration tests for process_code_files."""

    @patch("rag.ingestion.data_loader.populate_embeddings")
    @patch("rag.ingestion.data_loader.process_code_element")
    @patch("rag.ingestion.data_loader._check_db_populated")
    @patch("rag.ingestion.data_loader.GraphDBManager")
    @patch("rag.ingestion.data_loader.get_vector_index_config")
    def test_full_workflow_new_database(
        self,
        mock_get_config,
        mock_db_class,
        mock_check_db,
        mock_process_element,
        mock_populate_embeddings,
        tmp_path,
        vector_config,
        sample_ast_data,
    ):
        """Test complete workflow with new database."""
        # Setup
        mock_get_config.return_value = vector_config
        mock_db_manager = MagicMock()
        mock_db_class.return_value = mock_db_manager
        mock_check_db.return_value = False  # New database
        mock_process_element.return_value = Document(text="test", metadata={})
        mock_populate_embeddings.return_value = 1

        # Create results_* subdirectory (required by _process_ast_files)
        results_dir = tmp_path / "results_test"
        results_dir.mkdir()

        # Create test JSON file in results_* directory
        json_file = results_dir / "test.json"
        json_file.write_text(json.dumps(sample_ast_data))

        # Execute
        result = process_code_files(str(tmp_path))

        # Verify
        assert len(result) == 1  # Returns list of documents
        mock_db_manager.create_schema.assert_called_once()
        mock_db_manager.create_node.assert_called_once()
        mock_db_manager.create_relationships.assert_called_once()
        mock_populate_embeddings.assert_called_once()
        mock_db_manager.close.assert_called_once()
