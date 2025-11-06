"""
Tests for the orchestrator module.
"""

import pytest
import json
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
    return CodeGraphIndexer(ast_cache_dir=str(temp_ast_dir), top_k=3)


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
