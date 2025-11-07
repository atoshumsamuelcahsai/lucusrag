"""Unit tests for ProgressState dataclass."""

import json
import pytest
from rag.schemas.processed_files_tracker import ProgressState


@pytest.fixture
def temp_progress_file(tmp_path):
    """Create a temporary progress file path."""
    return tmp_path / "progress.json"


@pytest.fixture
def sample_progress_data():
    """Sample progress data for testing."""
    return {
        "timestamp": "20250101_120000",
        "processed_elements": ["elem1", "elem2"],
        "processed_files": ["file1.py", "file2.py"],
        "failed_elements": ["elem3"],
    }


class TestProgressStateInitialization:
    """Test ProgressState initialization."""

    def test_default_initialization(self):
        """Test creating ProgressState with defaults."""
        state = ProgressState()

        assert isinstance(state.timestamp, str)
        assert len(state.timestamp) == 15  # Format: YYYYMMDD_HHMMSS
        assert state.processed_elements == []
        assert state.processed_files == []
        assert state.failed_elements == []

    def test_initialization_with_values(self):
        """Test creating ProgressState with custom values."""
        state = ProgressState(
            timestamp="20250101_120000",
            processed_elements=["elem1"],
            processed_files=["file1.py"],
            failed_elements=["elem2"],
        )

        assert state.timestamp == "20250101_120000"
        assert state.processed_elements == ["elem1"]
        assert state.processed_files == ["file1.py"]
        assert state.failed_elements == ["elem2"]


class TestProgressStateFileOperations:
    """Test ProgressState file I/O operations."""

    def test_from_file_creates_new_when_not_exists(self, temp_progress_file):
        """Test from_file creates new state when file doesn't exist."""
        assert not temp_progress_file.exists()

        state = ProgressState.from_file(temp_progress_file)

        assert temp_progress_file.exists()
        assert isinstance(state, ProgressState)
        assert state.processed_elements == []
        assert state.processed_files == []
        assert state.failed_elements == []

    def test_from_file_loads_existing(self, temp_progress_file, sample_progress_data):
        """Test from_file loads existing progress state."""
        # Create existing file
        with open(temp_progress_file, "w") as f:
            json.dump(sample_progress_data, f)

        state = ProgressState.from_file(temp_progress_file)

        assert state.timestamp == "20250101_120000"
        assert state.processed_elements == ["elem1", "elem2"]
        assert state.processed_files == ["file1.py", "file2.py"]
        assert state.failed_elements == ["elem3"]

    def test_from_file_handles_missing_keys(self, temp_progress_file):
        """Test from_file handles missing keys gracefully."""
        # Create file with partial data
        partial_data = {"timestamp": "20250101_120000"}
        with open(temp_progress_file, "w") as f:
            json.dump(partial_data, f)

        state = ProgressState.from_file(temp_progress_file)

        assert state.timestamp == "20250101_120000"
        assert state.processed_elements == []
        assert state.processed_files == []
        assert state.failed_elements == []

    def test_to_file_saves_state(self, temp_progress_file):
        """Test to_file saves state correctly."""
        state = ProgressState(
            timestamp="20250101_120000",
            processed_elements=["elem1"],
            processed_files=["file1.py"],
            failed_elements=["elem2"],
        )

        state.to_file(temp_progress_file)

        assert temp_progress_file.exists()
        with open(temp_progress_file, "r") as f:
            data = json.load(f)

        assert data["timestamp"] == "20250101_120000"
        assert data["processed_elements"] == ["elem1"]
        assert data["processed_files"] == ["file1.py"]
        assert data["failed_elements"] == ["elem2"]

    def test_to_dict_returns_correct_structure(self):
        """Test to_dict returns correct dictionary structure."""
        state = ProgressState(
            timestamp="20250101_120000",
            processed_elements=["elem1"],
            processed_files=["file1.py"],
            failed_elements=["elem2"],
        )

        result = state.to_dict()

        assert result == {
            "timestamp": "20250101_120000",
            "processed_elements": ["elem1"],
            "processed_files": ["file1.py"],
            "failed_elements": ["elem2"],
        }


class TestProgressStateTracking:
    """Test ProgressState tracking methods."""

    def test_add_element(self, temp_progress_file):
        """Test adding an element."""
        state = ProgressState()

        state.add_element("elem1", temp_progress_file)

        assert "elem1" in state.processed_elements
        assert temp_progress_file.exists()

    def test_add_element_avoids_duplicates(self, temp_progress_file):
        """Test adding same element twice doesn't duplicate."""
        state = ProgressState()

        state.add_element("elem1", temp_progress_file)
        state.add_element("elem1", temp_progress_file)

        assert state.processed_elements.count("elem1") == 1

    def test_add_file(self, temp_progress_file):
        """Test adding a file."""
        state = ProgressState()

        state.add_file("file1.py", temp_progress_file)

        assert "file1.py" in state.processed_files
        assert temp_progress_file.exists()

    def test_add_file_avoids_duplicates(self, temp_progress_file):
        """Test adding same file twice doesn't duplicate."""
        state = ProgressState()

        state.add_file("file1.py", temp_progress_file)
        state.add_file("file1.py", temp_progress_file)

        assert state.processed_files.count("file1.py") == 1

    def test_add_failed(self, temp_progress_file):
        """Test adding a failed element."""
        state = ProgressState()

        state.add_failed("elem1", temp_progress_file)

        assert "elem1" in state.failed_elements
        assert temp_progress_file.exists()

    def test_add_failed_avoids_duplicates(self, temp_progress_file):
        """Test adding same failed element twice doesn't duplicate."""
        state = ProgressState()

        state.add_failed("elem1", temp_progress_file)
        state.add_failed("elem1", temp_progress_file)

        assert state.failed_elements.count("elem1") == 1


class TestProgressStateQueries:
    """Test ProgressState query methods."""

    def test_is_element_processed_returns_true_when_exists(self):
        """Test is_element_processed returns True for processed elements."""
        state = ProgressState(processed_elements=["elem1", "elem2"])

        assert state.is_element_processed("elem1") is True
        assert state.is_element_processed("elem2") is True

    def test_is_element_processed_returns_false_when_not_exists(self):
        """Test is_element_processed returns False for unprocessed elements."""
        state = ProgressState(processed_elements=["elem1"])

        assert state.is_element_processed("elem2") is False

    def test_is_file_processed_returns_true_when_exists(self):
        """Test is_file_processed returns True for processed files."""
        state = ProgressState(processed_files=["file1.py", "file2.py"])

        assert state.is_file_processed("file1.py") is True
        assert state.is_file_processed("file2.py") is True

    def test_is_file_processed_returns_false_when_not_exists(self):
        """Test is_file_processed returns False for unprocessed files."""
        state = ProgressState(processed_files=["file1.py"])

        assert state.is_file_processed("file2.py") is False

    def test_get_processed_elements_set(self):
        """Test get_processed_elements_set returns a set."""
        state = ProgressState(processed_elements=["elem1", "elem2", "elem3"])

        result = state.get_processed_elements_set()

        assert isinstance(result, set)
        assert result == {"elem1", "elem2", "elem3"}

    def test_get_processed_elements_set_empty(self):
        """Test get_processed_elements_set returns empty set when no elements."""
        state = ProgressState()

        result = state.get_processed_elements_set()

        assert isinstance(result, set)
        assert len(result) == 0


class TestProgressStateRoundTrip:
    """Test ProgressState save/load round-trip."""

    def test_save_and_load_preserves_state(self, temp_progress_file):
        """Test saving and loading preserves all state."""
        original = ProgressState(
            timestamp="20250101_120000",
            processed_elements=["elem1", "elem2", "elem3"],
            processed_files=["file1.py", "file2.py"],
            failed_elements=["elem4"],
        )

        original.to_file(temp_progress_file)
        loaded = ProgressState.from_file(temp_progress_file)

        assert loaded.timestamp == original.timestamp
        assert loaded.processed_elements == original.processed_elements
        assert loaded.processed_files == original.processed_files
        assert loaded.failed_elements == original.failed_elements

    def test_multiple_saves_updates_file(self, temp_progress_file):
        """Test multiple saves update the file correctly."""
        state = ProgressState()

        state.add_element("elem1", temp_progress_file)
        state.add_element("elem2", temp_progress_file)
        state.add_file("file1.py", temp_progress_file)

        loaded = ProgressState.from_file(temp_progress_file)

        assert loaded.processed_elements == ["elem1", "elem2"]
        assert loaded.processed_files == ["file1.py"]
