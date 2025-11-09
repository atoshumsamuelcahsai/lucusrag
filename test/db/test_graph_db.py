from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from neo4j import Driver

from rag.db.graph_db import GraphDBManager
from rag.schemas import CodeElement
from rag.schemas.vector_config import (
    Neo4jConfig,
    VectorIndexConfig,
    get_vector_index_config,
)


@pytest.fixture
def neo4j_config():
    """Fixture for Neo4j configuration."""
    return Neo4jConfig(
        url="bolt://localhost:7687", user="neo4j", password="test_password"
    )


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
def code_element():
    """Fixture for CodeElement."""
    element = CodeElement(
        type="function",
        name="test_function",
        docstring="Test function",
        code="def test_function(): pass",
        file_path="/test/path.py",
        parameters=[{"name": "x", "type": "int"}],
        return_type="str",
        decorators=["@decorator"],
        dependencies=["module.dependency"],
        base_classes=None,
        methods=None,
        calls=["other_function"],
        assignments=None,
        explanation="Test explanation",
    )
    return element


@pytest.fixture
def mock_driver():
    """Fixture for mocked Neo4j driver."""
    driver = MagicMock(spec=Driver)
    driver.verify_connectivity = Mock()
    driver.close = Mock()
    return driver


@pytest.fixture
def mock_session():
    """Fixture for mocked Neo4j session."""
    session = MagicMock()
    session.run = Mock(return_value=MagicMock())
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    return session


class TestGraphDBManagerInit:
    """Tests for GraphDBManager initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        manager = GraphDBManager()
        assert manager.config is not None
        assert manager._driver is None
        assert manager._max_retries == 3
        assert manager._delay == 1.0

    def test_init_with_custom_config(self, neo4j_config):
        """Test initialization with custom config."""
        manager = GraphDBManager(config=neo4j_config, max_retries=5, delay=2.0)
        assert manager.config == neo4j_config
        assert manager._max_retries == 5
        assert manager._delay == 2.0

    def test_init_lazy_driver(self):
        """Test that driver is not initialized on init."""
        manager = GraphDBManager()
        assert manager._driver is None


class TestGraphDBManagerDriver:
    """Tests for driver property and connection management."""

    @patch("rag.db.graph_db.GraphDatabase")
    def test_driver_successful_connection(
        self, mock_graph_db, neo4j_config, mock_driver
    ):
        """Test successful driver connection on first attempt."""
        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)
        driver = manager.driver

        assert driver == mock_driver
        mock_graph_db.driver.assert_called_once_with(
            neo4j_config.url,
            auth=(neo4j_config.user, neo4j_config.password),
            max_connection_lifetime=30 * 60,
        )
        mock_driver.verify_connectivity.assert_called_once()

    @patch("rag.db.graph_db.GraphDatabase")
    @patch("rag.db.graph_db.time.sleep")
    def test_driver_retry_logic(
        self, mock_sleep, mock_graph_db, neo4j_config, mock_driver
    ):
        """Test retry logic on connection failure."""
        # Fail twice, succeed on third attempt
        mock_graph_db.driver.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            mock_driver,
        ]

        manager = GraphDBManager(config=neo4j_config, max_retries=3, delay=1.0)
        driver = manager.driver

        assert driver == mock_driver
        assert mock_graph_db.driver.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("rag.db.graph_db.GraphDatabase")
    def test_driver_max_retries_exceeded(self, mock_graph_db, neo4j_config):
        """Test that ConnectionError is raised after max retries."""
        mock_graph_db.driver.side_effect = Exception("Connection failed")

        manager = GraphDBManager(config=neo4j_config, max_retries=2)

        with pytest.raises(ConnectionError, match="Failed to connect"):
            _ = manager.driver

    @patch("rag.db.graph_db.GraphDatabase")
    def test_driver_caching(self, mock_graph_db, neo4j_config, mock_driver):
        """Test that driver is cached after first initialization."""
        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)
        driver1 = manager.driver
        driver2 = manager.driver

        assert driver1 == driver2
        mock_graph_db.driver.assert_called_once()

    def test_close_driver(self, neo4j_config):
        """Test closing the driver."""
        manager = GraphDBManager(config=neo4j_config)
        mock_driver = Mock()
        manager._driver = mock_driver

        manager.close()

        mock_driver.close.assert_called_once()
        assert manager._driver is None

    def test_close_no_driver(self):
        """Test closing when no driver exists."""
        manager = GraphDBManager()
        manager.close()  # Should not raise


class TestGraphDBManagerSchema:
    """Tests for schema creation."""

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_schema_success(
        self, mock_graph_db, neo4j_config, vector_config, mock_session, mock_driver
    ):
        """Test successful schema creation."""
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        # Intercept execute_write and capture the transaction
        tx = Mock()

        def fake_execute_write(fn, *args, **kwargs):
            return fn(tx)

        mock_session.execute_write.side_effect = fake_execute_write

        manager = GraphDBManager(config=neo4j_config)

        # Mock _create_vector_index
        manager._create_vector_index = Mock()

        manager.create_schema(vector_config)

        # Verify session was used
        mock_driver.session.assert_called_once()

        # Now assert the number of schema statements executed inside the TX:
        # 1 constraint + 2 indexes
        assert tx.run.call_count == 3

        # Verify vector index creation was called
        manager._create_vector_index.assert_called_once()

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_schema_failure(
        self, mock_graph_db, neo4j_config, vector_config, mock_session, mock_driver
    ):
        """Test schema creation failure."""
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_session.run.side_effect = Exception("Schema creation failed")

        manager = GraphDBManager(config=neo4j_config)

        with pytest.raises(Exception, match="Schema creation failed"):
            manager.create_schema(vector_config)


class TestGraphDBManagerIndexOperations:
    """Tests for index operations."""

    def test_index_exist_true(self, mock_session, vector_config):
        """Test checking if index exists (returns True)."""
        mock_record = Mock()
        mock_record.get.return_value = True
        mock_session.run.return_value.single.return_value = mock_record

        manager = GraphDBManager()
        exists = manager._index_exists(mock_session, vector_config)

        assert exists is True
        mock_session.run.assert_called_once()

    def test_index_exist_false(self, mock_session, vector_config):
        """Test checking if index exists (returns False)."""
        mock_record = Mock()
        mock_record.get.return_value = False
        mock_session.run.return_value.single.return_value = mock_record

        manager = GraphDBManager()
        exists = manager._index_exists(mock_session, vector_config)

        assert exists is False

    def test_drop_index(self, mock_session, vector_config):
        """Test dropping an index."""
        manager = GraphDBManager()
        manager._drop_index(mock_session, vector_config)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args[0][0]
        assert "DROP INDEX" in call_args
        assert vector_config.name in call_args

    def test_create_index(self, mock_session, vector_config):
        """Test creating an index."""
        manager = GraphDBManager()
        manager._create_index(mock_session, vector_config)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "db.index.vector.createNodeIndex" in call_args[0][0]

        # Verify parameters (call_args is a tuple of (args, kwargs))
        # The second argument is the params dict
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["index_name"] == vector_config.name
        assert params["node_label"] == vector_config.node_label
        assert params["dimension"] == vector_config.dimension

    def test_create_vector_index_new(self, mock_session, vector_config):
        """Test creating vector index when it doesn't exist."""
        manager = GraphDBManager()
        manager._index_exists = Mock(return_value=False)
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config)

        manager._index_exists.assert_called_once()
        manager._create_index.assert_called_once()

    def test_create_vector_index_exists_overwrite(self, mock_session, vector_config):
        """Test creating vector index when it exists and overwrite=True."""
        manager = GraphDBManager()
        manager._index_exists = Mock(return_value=True)
        manager._drop_index = Mock()
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config, overwrite=True)

        manager._drop_index.assert_called_once()
        manager._create_index.assert_called_once()

    def test_create_vector_index_exists_no_overwrite(self, mock_session, vector_config):
        """Test creating vector index when it exists and overwrite=False."""
        manager = GraphDBManager()
        manager._index_exists = Mock(return_value=True)
        manager._drop_index = Mock()
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config, overwrite=False)

        manager._drop_index.assert_not_called()
        manager._create_index.assert_not_called()


class TestGraphDBManagerNodeOperations:
    """Tests for node operations."""

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_node_success(
        self, mock_graph_db, neo4j_config, vector_config, code_element
    ):
        """Test successful node creation."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock context manager for session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        # Mock transaction for execute_write
        mock_tx = Mock()
        mock_session.execute_write.side_effect = lambda fn: fn(mock_tx)

        manager = GraphDBManager(config=neo4j_config)

        # Reset any prior session() calls
        mock_driver.session.reset_mock()

        # Act
        manager.create_node(code_element, vector_config)

        # Assert
        mock_driver.session.assert_called_once()
        mock_session.execute_write.assert_called_once()
        mock_tx.run.assert_called_once()
        cypher = mock_tx.run.call_args[0][0]
        assert "MERGE" in cypher
        assert vector_config.node_label in cypher

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_node_failure(
        self,
        mock_graph_db,
        neo4j_config,
        vector_config,
        code_element,
    ):
        """Test node creation failure."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_graph_db.driver.return_value = mock_driver

        # Mock context manager for session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_write to raise exception
        mock_session.execute_write.side_effect = Exception("Node creation failed")

        manager = GraphDBManager(config=neo4j_config)

        with pytest.raises(Exception, match="Node creation failed"):
            manager.create_node(code_element, vector_config)


class TestGraphDBManagerRelationshipOperations:
    """Tests for relationship operations."""

    def test_add_inheritance(self, mock_session, vector_config):
        """Test adding inheritance relationships."""
        code_element = CodeElement(
            type="class",
            name="DerivedClass",
            docstring="",
            code="",
            file_path="/test.py",
            base_classes=["BaseClass1", "BaseClass2"],
        )

        # Mock transaction for execute_write
        mock_tx = Mock()
        mock_session.execute_write.side_effect = lambda fn: fn(mock_tx)

        manager = GraphDBManager()
        manager._add_inheritance(mock_session, vector_config, code_element)

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()
        # Verify query contains INHERITS_FROM
        cypher = mock_tx.run.call_args[0][0]
        assert "INHERITS_FROM" in cypher
        # Verify parameters
        params = mock_tx.run.call_args[0][1]
        assert params["bases"] == ["BaseClass1", "BaseClass2"]

    def test_add_calls(self, mock_session, vector_config):
        """Test adding function call relationships."""
        code_element = CodeElement(
            type="function",
            name="caller",
            docstring="",
            code="",
            file_path="/test.py",
            calls=["module.function1", "function2"],
        )

        # Mock transaction for execute_write
        mock_tx = Mock()
        mock_session.execute_write.side_effect = lambda fn: fn(mock_tx)

        manager = GraphDBManager()
        manager._add_calls(mock_session, vector_config, code_element)

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()
        # Verify query contains CALLS
        cypher = mock_tx.run.call_args[0][0]
        assert "CALLS" in cypher
        # Verify parameters
        params = mock_tx.run.call_args[0][1]
        assert params["caller_id"] == code_element.id
        assert params["calls"] == ["module.function1", "function2"]

    def test_add_dependencies(self, mock_session, vector_config):
        """Test adding dependency relationships."""
        code_element = CodeElement(
            type="module",
            name="test_module",
            docstring="",
            code="",
            file_path="/test.py",
            dependencies=["dep1.module", "dep2"],
        )

        # Mock transaction for execute_write
        mock_tx = Mock()
        mock_session.execute_write.side_effect = lambda fn: fn(mock_tx)

        manager = GraphDBManager()
        manager._add_dependencies(mock_session, vector_config, code_element)

        # Verify execute_write was called
        mock_session.execute_write.assert_called_once()
        # Verify query contains DEPENDS_ON
        cypher = mock_tx.run.call_args[0][0]
        assert "DEPENDS_ON" in cypher
        # Verify parameters
        params = mock_tx.run.call_args[0][1]
        assert params["source_id"] == code_element.id
        assert params["dependencies"] == ["dep1.module", "dep2"]

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_relationships_success(
        self, mock_graph_db, neo4j_config, vector_config, mock_session, mock_driver
    ):
        """Test successful relationship creation."""
        code_element = CodeElement(
            type="class",
            name="TestClass",
            docstring="",
            code="",
            file_path="/test.py",
            base_classes=["BaseClass"],
            calls=["function"],
            dependencies=["module"],
        )
        # Mock context manager for session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        # Mock transaction for execute_write
        mock_tx = Mock()
        mock_session.execute_write.side_effect = lambda fn: fn(mock_tx)

        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)
        manager.create_relationships(code_element, vector_config)

        mock_driver.session.assert_called_once()
        # Should call execute_write for: base_classes, calls, dependencies = 3 times
        assert mock_session.execute_write.call_count == 3

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_relationships_failure(
        self,
        mock_graph_db,
        neo4j_config,
        vector_config,
        code_element,
        mock_session,
        mock_driver,
    ):
        """Test relationship creation failure."""
        code_element.base_classes = ["BaseClass"]

        # Mock context manager for session
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_write to raise exception
        mock_session.execute_write.side_effect = Exception(
            "Relationship creation failed"
        )

        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)

        with pytest.raises(Exception, match="Relationship creation failed"):
            manager.create_relationships(code_element, vector_config)


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
    async def test_get_vector_index_config_from_env(self):
        """Test getting config from environment variables."""
        config = get_vector_index_config()

        assert config.name == "test_index"
        assert config.dimension == 512
        assert config.node_label == "TestLabel"
        assert config.vector_property == "vec"
        assert config.similarity_metric == "euclidean"
        assert config.neo4j_url == "bolt://test:7687"
        assert config.neo4j_user == "test_user"
        assert config.neo4j_password == "test_pass"


class TestGraphDBManagerIntegration:
    """Integration tests for GraphDBManager."""

    @patch("rag.db.graph_db.GraphDatabase")
    def test_full_workflow(
        self,
        mock_graph_db,
        neo4j_config,
        vector_config,
        code_element,
        mock_session,
        mock_driver,
    ):
        """Test full workflow: connect, create schema, add nodes, add relationships, close."""
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)

        # Create schema
        manager._create_vector_index = Mock()
        manager.create_schema(vector_config)

        # Create node
        manager.create_node(code_element, vector_config)

        # Create relationships (code_element has calls)
        if code_element.calls:
            manager.create_relationships(code_element, vector_config)

        # Verify workflow before closing
        mock_driver.session.assert_called()

        # Close
        manager.close()

        # Verify close was called
        mock_driver.close.assert_called_once()
        assert manager._driver is None
