from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from neo4j import Driver

from rag.db.graph_db import GraphDBManager, get_vector_index_config
from rag.schemas import CodeElement
from rag.schemas.vector_config import Neo4jConfig, VectorIndexConfig


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
    return CodeElement(
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

        manager = GraphDBManager(config=neo4j_config)

        # Mock _create_vector_index
        manager._create_vector_index = Mock()

        manager.create_schema(vector_config)

        # Verify session was used
        mock_driver.session.assert_called_once()

        # Verify statements were executed (5 statements: 1 constraint + 4 indexes)
        assert mock_session.run.call_count == 5

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
        exists = manager._index_exist(mock_session, vector_config)

        assert exists is True
        mock_session.run.assert_called_once()

    def test_index_exist_false(self, mock_session, vector_config):
        """Test checking if index exists (returns False)."""
        mock_record = Mock()
        mock_record.get.return_value = False
        mock_session.run.return_value.single.return_value = mock_record

        manager = GraphDBManager()
        exists = manager._index_exist(mock_session, vector_config)

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
        manager._index_exist = Mock(return_value=False)
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config)

        manager._index_exist.assert_called_once()
        manager._create_index.assert_called_once()

    def test_create_vector_index_exists_overwrite(self, mock_session, vector_config):
        """Test creating vector index when it exists and overwrite=True."""
        manager = GraphDBManager()
        manager._index_exist = Mock(return_value=True)
        manager._drop_index = Mock()
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config, overwrite=True)

        manager._drop_index.assert_called_once()
        manager._create_index.assert_called_once()

    def test_create_vector_index_exists_no_overwrite(self, mock_session, vector_config):
        """Test creating vector index when it exists and overwrite=False."""
        manager = GraphDBManager()
        manager._index_exist = Mock(return_value=True)
        manager._drop_index = Mock()
        manager._create_index = Mock()

        manager._create_vector_index(mock_session, vector_config, overwrite=False)

        manager._drop_index.assert_not_called()
        manager._create_index.assert_not_called()


class TestGraphDBManagerNodeOperations:
    """Tests for node operations."""

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_nodes_success(
        self,
        mock_graph_db,
        neo4j_config,
        vector_config,
        code_element,
        mock_session,
        mock_driver,
    ):
        """Test successful node creation."""
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)
        manager.create_nodes(code_element, vector_config)

        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once()

        # Verify query contains MERGE
        call_args = mock_session.run.call_args[0][0]
        assert "MERGE" in call_args
        assert vector_config.node_label in call_args

    @patch("rag.db.graph_db.GraphDatabase")
    def test_create_nodes_failure(
        self,
        mock_graph_db,
        neo4j_config,
        vector_config,
        code_element,
        mock_session,
        mock_driver,
    ):
        """Test node creation failure."""
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_session.run.side_effect = Exception("Node creation failed")

        manager = GraphDBManager(config=neo4j_config)

        with pytest.raises(Exception, match="Node creation failed"):
            manager.create_nodes(code_element, vector_config)


class TestGraphDBManagerRelationshipOperations:
    """Tests for relationship operations."""

    def test_add_classes(self, mock_session, vector_config):
        """Test adding class-method relationships."""
        code_element = CodeElement(
            type="class",
            name="TestClass",
            docstring="",
            code="",
            file_path="/test.py",
            methods=["method1", "method2"],
        )

        manager = GraphDBManager()
        manager._add_classes(mock_session, vector_config, code_element, "test_id")

        assert mock_session.run.call_count == 2

        # Verify query contains HAS_METHOD
        call_args = mock_session.run.call_args[0][0]
        assert "HAS_METHOD" in call_args

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

        manager = GraphDBManager()
        manager._add_inheritance(mock_session, vector_config, code_element, "test_id")

        assert mock_session.run.call_count == 2

        # Verify query contains INHERITS_FROM
        call_args = mock_session.run.call_args[0][0]
        assert "INHERITS_FROM" in call_args

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

        manager = GraphDBManager()
        manager._add_calls(mock_session, vector_config, code_element, "test_id")

        assert mock_session.run.call_count == 2

        # Verify query contains CALLS
        call_args = mock_session.run.call_args[0][0]
        assert "CALLS" in call_args

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

        manager = GraphDBManager()
        manager._add_dependencies(mock_session, vector_config, code_element, "test_id")

        assert mock_session.run.call_count == 2

        # Verify query contains DEPENDS_ON
        call_args = mock_session.run.call_args[0][0]
        assert "DEPENDS_ON" in call_args

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
            methods=["method1"],
            base_classes=["BaseClass"],
            calls=["function"],
            dependencies=["module"],
        )

        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        manager = GraphDBManager(config=neo4j_config)
        manager.create_relationships(code_element, vector_config)

        mock_driver.session.assert_called_once()
        # Should call run for: methods, base_classes, calls, dependencies = 4 times
        assert mock_session.run.call_count == 4

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
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_session.run.side_effect = Exception("Relationship creation failed")

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
        manager.create_nodes(code_element, vector_config)

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
