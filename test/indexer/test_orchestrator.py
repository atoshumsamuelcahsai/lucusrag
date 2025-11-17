"""
Tests for the orchestrator module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from rag.schemas.vector_config import VectorIndexConfig
from rag.indexer.orchestrator import CodeGraphIndexer, Mode, BuildResult


@pytest.fixture
def temp_ast_dir(tmp_path):
    """Create a temporary AST directory with sample files."""
    ast_dir = tmp_path / "ast_cache"
    ast_dir.mkdir()

    # Create sample AST JSON files
    sample_ast = {
        "type": "function",
        "name": "sample_function",
        "docstring": "A sample function for testing",
        "code": "def sample_function():\n    return 'test'",
        "file_path": "test.py",
        "parameters": [],
        "return_type": "str",
        "dependencies": [],
        "calls": [],
    }

    (ast_dir / "test1.json").write_text(json.dumps(sample_ast))
    (ast_dir / "test2.json").write_text(json.dumps(sample_ast))

    return ast_dir


@pytest.fixture
def indexer(temp_ast_dir):
    """Create a CodeGraphIndexer instance."""
    manifest_path = temp_ast_dir / ".rag_manifest.json"
    return CodeGraphIndexer(
        ast_cache_dir=str(temp_ast_dir),
        top_k=3,
        manifest_path=str(manifest_path),
    )


class TestCodeGraphIndexer:
    """Test suite for CodeGraphIndexer."""

    @pytest.mark.asyncio
    async def test_snapshot_files(self, indexer, temp_ast_dir):
        """Test file snapshot creation."""
        snapshot = await indexer._snapshot_files()

        assert len(snapshot) == 2  # Two JSON files
        for file_path, meta in snapshot.items():
            assert "mtime" in meta
            assert "sha256" in meta
            assert isinstance(meta["mtime"], int)
            assert isinstance(meta["sha256"], str)
            assert len(meta["sha256"]) == 64  # SHA256 hex length

    @pytest.mark.asyncio
    async def test_sha256(self, indexer, tmp_path):
        """Test SHA256 hashing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = await indexer._sha256(test_file)

        # Known SHA256 of "Hello, World!"
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert hash_result == expected

    def test_embed_signature(self, indexer):
        """Test embedding signature generation."""
        config = VectorIndexConfig(
            name="test_index",
            dimension=1536,
            node_label="TestNode",
            vector_property="embedding",
            similarity_metric="cosine",
            neo4j_url="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        sig = indexer._embed_signature(config)
        assert sig == "1536:cosine"

    @pytest.mark.asyncio
    async def test_save_and_load_manifest(self, indexer, temp_ast_dir):
        """Test manifest save and load."""
        files = {"test.json": {"mtime": 123456, "sha256": "abc123"}}
        embed_sig = "1536:cosine"

        await indexer._save_manifest(files, embed_sig)

        # Verify file was created
        assert indexer.manifest_path.exists()

        # Load and verify
        loaded = indexer._load_manifest()
        assert loaded["version"] == indexer.schema_version
        assert loaded["files"] == files
        assert loaded["embed_sig"] == embed_sig

    def test_load_manifest_missing(self, indexer):
        """Test loading manifest when file doesn't exist."""
        # Ensure manifest doesn't exist
        if indexer.manifest_path.exists():
            indexer.manifest_path.unlink()

        manifest = indexer._load_manifest()

        assert manifest["version"] == indexer.schema_version
        assert manifest["files"] == {}
        assert manifest["embed_sig"] == ""

    def test_diff_no_changes(self, indexer):
        """Test diff when no files changed."""
        prev = {
            "version": 1,
            "embed_sig": "1536:cosine",
            "files": {
                "test1.json": {"mtime": 123, "sha256": "abc"},
                "test2.json": {"mtime": 456, "sha256": "def"},
            },
        }
        now = prev["files"]

        changed, removed = indexer._diff(prev, now, "1536:cosine")

        assert len(changed) == 0
        assert len(removed) == 0

    def test_diff_with_changes(self, indexer):
        """Test diff when files changed."""
        prev = {
            "version": 1,
            "embed_sig": "1536:cosine",
            "files": {
                "test1.json": {"mtime": 123, "sha256": "abc"},
                "test2.json": {"mtime": 456, "sha256": "def"},
            },
        }
        now = {
            "test1.json": {"mtime": 789, "sha256": "xyz"},  # Changed
            "test2.json": {"mtime": 456, "sha256": "def"},  # Same
            "test3.json": {"mtime": 999, "sha256": "new"},  # New
        }

        changed, removed = indexer._diff(prev, now, "1536:cosine")

        assert "test1.json" in changed  # Modified
        assert "test3.json" in changed  # New
        assert len(removed) == 0

    def test_diff_with_removed(self, indexer):
        """Test diff when files removed."""
        prev = {
            "version": 1,
            "embed_sig": "1536:cosine",
            "files": {
                "test1.json": {"mtime": 123, "sha256": "abc"},
                "test2.json": {"mtime": 456, "sha256": "def"},
            },
        }
        now = {"test1.json": {"mtime": 123, "sha256": "abc"}}  # test2.json removed

        changed, removed = indexer._diff(prev, now, "1536:cosine")

        assert len(changed) == 0
        assert "test2.json" in removed

    def test_diff_schema_version_changed(self, indexer):
        """Test diff when schema version changes."""
        prev = {
            "version": 1,
            "embed_sig": "1536:cosine",
            "files": {"test.json": {"mtime": 123, "sha256": "abc"}},
        }
        now = {"test.json": {"mtime": 123, "sha256": "abc"}}

        indexer.schema_version = 2  # Changed version

        changed, removed = indexer._diff(prev, now, "1536:cosine")

        # Schema change should trigger full rebuild
        assert len(changed) > 0

    def test_diff_embed_sig_changed(self, indexer):
        """Test diff when embedding signature changes."""
        prev = {
            "version": 1,
            "embed_sig": "1536:cosine",
            "files": {"test.json": {"mtime": 123, "sha256": "abc"}},
        }
        now = {"test.json": {"mtime": 123, "sha256": "abc"}}

        changed, removed = indexer._diff(prev, now, "768:euclidean")  # Different sig

        # Embed sig change should trigger full rebuild
        assert len(changed) > 0

    def test_query_without_build_raises_error(self, indexer):
        """Test that query fails before build."""
        with pytest.raises(RuntimeError, match="Indexer not built"):
            indexer.query("test query")


class TestBuildResult:
    """Test BuildResult dataclass."""

    def test_build_result_creation(self):
        """Test BuildResult instantiation."""
        result = BuildResult(
            documents=10, elapsed_s=5.5, mode=Mode.BUILD, schema_version=1
        )

        assert result.documents == 10
        assert result.elapsed_s == 5.5
        assert result.mode == Mode.BUILD
        assert result.schema_version == 1

    def test_build_result_str(self):
        """Test BuildResult string representation."""
        result = BuildResult(
            documents=10, elapsed_s=5.5, mode=Mode.BUILD, schema_version=1
        )

        result_str = str(result)
        assert "[BUILD]" in result_str
        assert "10 docs" in result_str
        assert "5.50s" in result_str
        assert "schema v1" in result_str

    def test_build_result_frozen(self):
        """Test that BuildResult is immutable."""
        result = BuildResult(
            documents=10, elapsed_s=5.5, mode=Mode.BUILD, schema_version=1
        )

        with pytest.raises(Exception):  # dataclass frozen
            result.documents = 20


class TestMode:
    """Test Mode enum."""

    def test_mode_values(self):
        """Test Mode enum values."""
        assert Mode.BUILD.value == "build"
        assert Mode.REFRESH.value == "refresh"


class TestCodeGraphIndexerBuild:
    """Test build() method."""

    @pytest.mark.asyncio
    async def test_build_success(self, indexer):
        """Test successful build."""
        with patch("rag.indexer.orchestrator.graph_configure_settings"), patch(
            "rag.indexer.orchestrator.VectorIndexConfig"
        ) as mock_config_class, patch(
            "rag.indexer.orchestrator.process_code_files"
        ) as mock_process, patch(
            "rag.indexer.orchestrator.create_vector_index_from_existing_nodes"
        ) as mock_create_index, patch(
            "rag.indexer.orchestrator.make_query_engine"
        ) as mock_make_engine:

            mock_config_instance = Mock()
            mock_config_instance.dimension = 1536
            mock_config_instance.similarity_metric = "cosine"
            mock_config_class.from_env.return_value = mock_config_instance

            mock_docs = [Mock() for _ in range(5)]
            mock_process.return_value = mock_docs

            mock_index = Mock()
            mock_create_index.return_value = mock_index

            mock_engine = Mock()
            mock_make_engine.return_value = mock_engine

            result = await indexer.build()

            assert result.documents == 5
            assert result.mode == Mode.BUILD
            assert result.elapsed_s > 0
            assert indexer._index == mock_index
            assert indexer._engine == mock_engine

    @pytest.mark.asyncio
    async def test_build_updates_manifest(self, indexer):
        """Test that build updates manifest."""
        with patch("rag.indexer.orchestrator.graph_configure_settings"), patch(
            "rag.indexer.orchestrator.VectorIndexConfig"
        ) as mock_config_class, patch(
            "rag.indexer.orchestrator.process_code_files"
        ) as mock_process, patch(
            "rag.indexer.orchestrator.create_vector_index_from_existing_nodes"
        ) as mock_create_index, patch(
            "rag.indexer.orchestrator.make_query_engine"
        ) as mock_make_engine, patch.object(
            indexer, "_update_manifest"
        ) as mock_update:

            mock_config_instance = Mock()
            mock_config_instance.dimension = 1536
            mock_config_instance.similarity_metric = "cosine"
            mock_config_class.from_env.return_value = mock_config_instance

            mock_process.return_value = []
            mock_create_index.return_value = Mock()
            mock_make_engine.return_value = Mock()

            await indexer.build()

            mock_update.assert_called_once()


class TestCodeGraphIndexerRefresh:
    """Test refresh() method."""

    @pytest.mark.asyncio
    async def test_refresh_no_changes(self, indexer):
        """Test refresh when no changes detected."""
        with patch.object(indexer, "_load_manifest") as mock_load, patch.object(
            indexer, "_snapshot_files"
        ) as mock_snapshot, patch.object(indexer, "_diff") as mock_diff, patch(
            "rag.indexer.orchestrator.VectorIndexConfig"
        ) as mock_config_class:

            mock_config_instance = Mock()
            mock_config_instance.dimension = 1536
            mock_config_instance.similarity_metric = "cosine"
            mock_config_class.from_env.return_value = mock_config_instance

            mock_load.return_value = {
                "version": 1,
                "files": {},
                "embed_sig": "1536:cosine",
            }
            mock_snapshot.return_value = {}
            mock_diff.return_value = ([], [])  # No changes

            result = await indexer.refresh()

            assert result.documents == 0
            assert result.mode == Mode.REFRESH
            assert result.elapsed_s == 0.0

    @pytest.mark.asyncio
    async def test_refresh_with_changes(self, indexer):
        """Test refresh when changes detected."""
        with patch.object(indexer, "_load_manifest") as mock_load, patch.object(
            indexer, "_snapshot_files"
        ) as mock_snapshot, patch.object(indexer, "_diff") as mock_diff, patch.object(
            indexer, "_save_manifest"
        ), patch(
            "rag.indexer.orchestrator.VectorIndexConfig"
        ) as mock_config_class, patch(
            "rag.indexer.orchestrator.process_code_files"
        ) as mock_process, patch(
            "rag.indexer.orchestrator.create_vector_index_from_existing_nodes"
        ) as mock_create_index, patch(
            "rag.indexer.orchestrator.make_query_engine"
        ) as mock_make_engine:

            mock_config_instance = Mock()
            mock_config_instance.dimension = 1536
            mock_config_instance.similarity_metric = "cosine"
            mock_config_class.from_env.return_value = mock_config_instance

            mock_load.return_value = {
                "version": 1,
                "files": {},
                "embed_sig": "1536:cosine",
            }
            mock_snapshot.return_value = {"file1.json": {"mtime": 123, "sha256": "abc"}}
            mock_diff.return_value = (["file1.json"], [])  # Changed files

            mock_docs = [Mock() for _ in range(3)]
            mock_process.return_value = mock_docs
            mock_create_index.return_value = Mock()
            mock_make_engine.return_value = Mock()

            result = await indexer.refresh()

            assert result.documents == 3
            assert result.mode == Mode.REFRESH
            assert result.elapsed_s > 0
            mock_process.assert_called_once_with(
                str(indexer.ast_dir), force_rebuild_graph=True
            )

    @pytest.mark.asyncio
    async def test_refresh_with_removed_files(self, indexer):
        """Test refresh when files are removed."""
        with patch.object(indexer, "_load_manifest") as mock_load, patch.object(
            indexer, "_snapshot_files"
        ) as mock_snapshot, patch.object(indexer, "_diff") as mock_diff, patch.object(
            indexer, "_save_manifest"
        ), patch(
            "rag.indexer.orchestrator.VectorIndexConfig"
        ) as mock_config_class, patch(
            "rag.indexer.orchestrator.process_code_files"
        ) as mock_process, patch(
            "rag.indexer.orchestrator.create_vector_index_from_existing_nodes"
        ) as mock_create_index, patch(
            "rag.indexer.orchestrator.make_query_engine"
        ) as mock_make_engine:

            mock_config_instance = Mock()
            mock_config_instance.dimension = 1536
            mock_config_instance.similarity_metric = "cosine"
            mock_config_class.from_env.return_value = mock_config_instance

            mock_load.return_value = {
                "version": 1,
                "files": {"old.json": {}},
                "embed_sig": "1536:cosine",
            }
            mock_snapshot.return_value = {}
            mock_diff.return_value = ([], ["old.json"])  # Removed files

            mock_docs = []
            mock_process.return_value = mock_docs
            mock_create_index.return_value = Mock()
            mock_make_engine.return_value = Mock()

            result = await indexer.refresh()

            assert result.mode == Mode.REFRESH
            mock_process.assert_called_once()


class TestCodeGraphIndexerQuery:
    """Test query methods."""

    @pytest.mark.asyncio
    async def test_aquery_success(self, indexer):
        """Test async query when engine is built."""
        mock_engine = Mock()
        mock_timed_engine = Mock()
        mock_timed_engine.aquery = AsyncMock(return_value="test response")
        indexer._engine = mock_engine
        indexer._timed_engine = mock_timed_engine

        result = await indexer.aquery("test query")

        assert result == "test response"
        mock_timed_engine.aquery.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_aquery_creates_timed_engine_if_missing(self, indexer):
        """Test that aquery creates timed engine if missing."""
        mock_engine = Mock()
        indexer._engine = mock_engine
        indexer._timed_engine = None

        with patch("rag.indexer.orchestrator.TimedQueryEngine") as mock_timed_class:
            mock_timed_engine = Mock()
            mock_timed_engine.aquery = AsyncMock(return_value="test response")
            mock_timed_class.return_value = mock_timed_engine

            result = await indexer.aquery("test query")

            assert result == "test response"
            assert indexer._timed_engine == mock_timed_engine

    @pytest.mark.asyncio
    async def test_aquery_raises_when_not_built(self, indexer):
        """Test that aquery raises error when not built."""
        indexer._engine = None

        with pytest.raises(RuntimeError, match="Indexer not built"):
            await indexer.aquery("test query")

    def test_query_success(self, indexer):
        """Test sync query when engine is built."""
        mock_engine = Mock()
        mock_engine.query.return_value = "test response"
        indexer._engine = mock_engine

        result = indexer.query("test query")

        assert result == "test response"
        mock_engine.query.assert_called_once_with("test query")

    def test_query_raises_when_not_built(self, indexer):
        """Test that query raises error when not built."""
        indexer._engine = None

        with pytest.raises(RuntimeError, match="Indexer not built"):
            indexer.query("test query")


class TestCodeGraphIndexerRetrieveDocuments:
    """Test retrieve_documents method."""

    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, indexer):
        """Test successful document retrieval."""
        mock_engine = Mock()
        indexer._engine = mock_engine

        with patch(
            "rag.indexer.orchestrator.retrieve_documents_from_engine"
        ) as mock_retrieve:
            mock_retrieve.return_value = [
                {"node_id": "id1", "text": "content1", "score": 0.9}
            ]

            result = await indexer.retrieve_documents("test query", k=10)

            assert len(result) == 1
            assert result[0]["node_id"] == "id1"
            mock_retrieve.assert_called_once_with(mock_engine, "test query", 10)

    @pytest.mark.asyncio
    async def test_retrieve_documents_raises_when_not_built(self, indexer):
        """Test that retrieve_documents raises error when not built."""
        indexer._engine = None

        with pytest.raises(RuntimeError, match="Indexer not built"):
            await indexer.retrieve_documents("test query")


class TestCodeGraphIndexerInitialization:
    """Test CodeGraphIndexer initialization."""

    def test_indexer_init_defaults(self, temp_ast_dir):
        """Test indexer initialization with defaults."""
        indexer = CodeGraphIndexer(ast_cache_dir=str(temp_ast_dir))

        assert indexer.ast_dir == Path(temp_ast_dir)
        assert indexer.schema_version == 1
        assert indexer.top_k == 5
        assert indexer._index is None
        assert indexer._engine is None

    def test_indexer_init_custom_params(self, temp_ast_dir):
        """Test indexer initialization with custom parameters."""
        manifest_path = temp_ast_dir / "custom_manifest.json"
        indexer = CodeGraphIndexer(
            ast_cache_dir=str(temp_ast_dir),
            schema_version=2,
            top_k=10,
            manifest_path=str(manifest_path),
        )

        assert indexer.schema_version == 2
        assert indexer.top_k == 10
        assert indexer.manifest_path == manifest_path


# Integration test placeholder
@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires Neo4j and API keys - run manually")
async def test_full_build_integration(temp_ast_dir):
    """
    Integration test for full build process.

    NOTE: This requires:
    - Running Neo4j instance
    - Valid API keys in .env
    - Run with: pytest -v -s test/indexer/test_orchestrator.py -k integration
    """
    indexer = CodeGraphIndexer(ast_cache_dir=str(temp_ast_dir), top_k=3)

    # Build
    result = await indexer.build()
    assert result.documents > 0
    assert result.mode == Mode.BUILD
    assert result.elapsed_s > 0

    # Query
    response = indexer.query("sample function")
    assert response is not None

    # Refresh with no changes
    result2 = await indexer.refresh()
    assert result2.documents == 0
    assert result2.mode == Mode.REFRESH
