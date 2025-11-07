from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import typing as t
import json


@dataclass
class ProgressState:
    """Tracks progress of AST file processing."""

    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    processed_elements: t.List[str] = field(default_factory=list)
    processed_files: t.List[str] = field(default_factory=list)
    failed_elements: t.List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, progress_file: Path) -> "ProgressState":
        """Load progress state from JSON file, or create new if file doesn't exist.

        Args:
            progress_file: Path to progress JSON file

        Returns:
            ProgressState instance
        """
        if progress_file.exists():
            with open(progress_file, "r") as f:
                data = json.load(f)
                # Ensure all required keys exist
                return cls(
                    timestamp=data.get(
                        "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
                    ),
                    processed_elements=data.get("processed_elements", []),
                    processed_files=data.get("processed_files", []),
                    failed_elements=data.get("failed_elements", []),
                )
        else:
            # Create new progress state
            state = cls()
            state.to_file(progress_file)
            return state

    def to_file(self, progress_file: Path) -> None:
        """Save progress state to JSON file.

        Args:
            progress_file: Path to progress JSON file
        """
        with open(progress_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp,
            "processed_elements": self.processed_elements,
            "processed_files": self.processed_files,
            "failed_elements": self.failed_elements,
        }

    def add_element(self, element_id: str, progress_file: Path) -> None:
        """Mark element as processed and save to file.

        Args:
            element_id: Unique identifier for the element
            progress_file: Path to progress JSON file
        """
        if element_id not in self.processed_elements:
            self.processed_elements.append(element_id)
            self.to_file(progress_file)

    def add_file(self, file_path: str, progress_file: Path) -> None:
        """Mark file as processed and save to file.

        Args:
            file_path: Path to the processed file
            progress_file: Path to progress JSON file
        """
        if file_path not in self.processed_files:
            self.processed_files.append(file_path)
            self.to_file(progress_file)

    def add_failed(self, element_id: str, progress_file: Path) -> None:
        """Mark element as failed and save to file.

        Args:
            element_id: Unique identifier for the failed element
            progress_file: Path to progress JSON file
        """
        if element_id not in self.failed_elements:
            self.failed_elements.append(element_id)
            self.to_file(progress_file)

    def is_element_processed(self, element_id: str) -> bool:
        """Check if element was already processed.

        Args:
            element_id: Unique identifier for the element

        Returns:
            True if element was processed, False otherwise
        """
        return element_id in self.processed_elements

    def is_file_processed(self, file_path: str) -> bool:
        """Check if file was already processed.

        Args:
            file_path: Path to the file

        Returns:
            True if file was processed, False otherwise
        """
        return file_path in self.processed_files

    def get_processed_elements_set(self) -> t.Set[str]:
        """Get processed elements as a set for fast lookup.

        Returns:
            Set of processed element IDs
        """
        return set(self.processed_elements)
