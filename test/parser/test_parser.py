"""Unit tests for parser module."""

import pytest
from llama_index.core.schema import TextNode
from llama_index.core import Document

from rag.parser.parser import (
    process_code_element,
    parse_documents_to_nodes,
    create_text_representation,
    format_parameters,
)
from rag.schemas.code_element import CodeElement


@pytest.fixture
def sample_code_element():
    """Create a sample CodeElement for testing."""
    return CodeElement(
        type="function",
        name="calculate_sum",
        docstring="Calculate sum of two numbers.",
        code="def calculate_sum(a, b):\n    return a + b",
        file_path="src/math_utils.py",
        parameters=[
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"},
        ],
        return_type="int",
        decorators=["staticmethod"],
        dependencies=["math"],
        base_classes=[],
        calls=["add"],
    )


class TestProcessCodeElement:
    """Test process_code_element function."""

    def test_document_metadata(self, sample_code_element):
        """Test that document metadata is set correctly."""
        doc = process_code_element(sample_code_element)

        assert doc.metadata["name"] == sample_code_element.name
        assert doc.metadata["type"] == sample_code_element.type
        assert doc.metadata["file_path"] == sample_code_element.file_path
        assert doc.metadata["docstring"] == sample_code_element.docstring

    def test_handles_optional_fields(self):
        """Test handling of optional fields."""
        minimal_element = CodeElement(
            type="function",
            name="test_func",
            docstring="",
            code="def test_func(): pass",
            file_path="test.py",
        )

        doc = process_code_element(minimal_element)

        assert doc.metadata["parameters"] == []
        assert doc.metadata["return_type"] == ""
        assert doc.metadata["dependencies"] == []

    def test_preserves_list_fields(self, sample_code_element):
        """Test that list fields are preserved."""
        doc = process_code_element(sample_code_element)

        assert doc.metadata["parameters"] == sample_code_element.parameters
        assert doc.metadata["dependencies"] == sample_code_element.dependencies
        assert doc.metadata["calls"] == sample_code_element.calls
        assert doc.metadata["decorators"] == sample_code_element.decorators


class TestParseDocumentsToNodes:
    """Test parse_documents_to_nodes function."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return Document(
            text="def test(): pass",
            metadata={
                "id": "test.py:function:test",
                "name": "test",
                "type": "function",
                "file_path": "test.py",
                "code": "def test(): pass",
                "docstring": "Test function",
                "parameters": [],
                "return_type": "None",
            },
        )

    def test_call_with_single_document(self, sample_document):
        """Test parsing single document."""
        nodes = parse_documents_to_nodes([sample_document])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_call_with_multiple_documents(self, sample_document):
        """Test parsing multiple documents."""
        docs = [sample_document, sample_document]

        nodes = parse_documents_to_nodes(docs)

        assert len(nodes) == 2

    def test_call_with_empty_list(self):
        """Test parsing empty list."""
        nodes = parse_documents_to_nodes([])

        assert nodes == []

    def test_node_metadata_structure(self, sample_document):
        """Test that node metadata contains required fields."""
        nodes = parse_documents_to_nodes([sample_document])

        metadata = nodes[0].metadata
        assert "id" in metadata
        assert "name" in metadata
        assert "type" in metadata
        assert "file_path" in metadata

    def test_node_id_format(self, sample_document):
        """Test node ID format."""
        nodes = parse_documents_to_nodes([sample_document])

        node_id = nodes[0].metadata["id"]
        assert ":" in node_id
        assert "test.py" in node_id

    def test_handles_missing_optional_metadata(self):
        """Test handling of documents with minimal metadata."""
        doc = Document(
            text="test",
            metadata={
                "name": "test",
                "type": "function",
                "file_path": "test.py",
            },
        )

        nodes = parse_documents_to_nodes([doc])

        assert len(nodes) == 1

    def test_handles_document_errors_gracefully(self):
        """Test error handling for malformed documents."""
        bad_doc = Document(
            text="test",
            metadata={},  # Missing required fields
        )

        # Should not raise, but log and skip
        nodes = parse_documents_to_nodes([bad_doc])

        # Should handle error gracefully
        assert isinstance(nodes, list)

    def test_show_progress_parameter(self, sample_document):
        """Test that show_progress parameter is accepted."""
        nodes = parse_documents_to_nodes([sample_document], show_progress=True)

        assert len(nodes) == 1

    def test_accepts_kwargs(self, sample_document):
        """Test that additional kwargs are accepted."""
        nodes = parse_documents_to_nodes([sample_document], some_param=True)

        assert len(nodes) == 1


class TestCreateTextRepresentation:
    """Test create_text_representation function."""

    def test_creates_formatted_text(self):
        """Test creating formatted text representation."""
        metadata = {
            "name": "test_func",
            "type": "function",
            "code": "def test_func(): pass",
            "docstring": "Test function",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "None",
        }

        text = create_text_representation(metadata)

        assert "test_func" in text
        assert "function" in text
        assert "Test function" in text

    def test_handles_empty_lists(self):
        """Test handling of empty list fields."""
        metadata = {
            "name": "test",
            "type": "function",
            "code": "pass",
            "docstring": "",
            "parameters": [],
            "dependencies": [],
            "base_classes": [],
            "calls": [],
        }

        text = create_text_representation(metadata)

        assert isinstance(text, str)
        assert "test" in text

    def test_handles_none_values(self):
        """Test handling of None values."""
        metadata = {
            "name": "test",
            "type": "function",
            "code": "pass",
            "docstring": None,
            "parameters": None,
        }

        text = create_text_representation(metadata)

        assert isinstance(text, str)

    def test_includes_all_sections(self):
        """Test that all sections are included in output."""
        metadata = {
            "name": "func",
            "type": "function",
            "code": "def func(): pass",
            "docstring": "Docs",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "None",
            "dependencies": ["os"],
            "base_classes": [],
            "calls": ["print"],
        }

        text = create_text_representation(metadata)

        assert "Name:" in text
        assert "Type:" in text
        assert "Description:" in text
        assert "Code:" in text
        assert "Parameters:" in text
        assert "Return Type:" in text
        assert "Dependencies:" in text


class TestFormatParameters:
    """Test format_parameters function."""

    def test_formats_single_parameter(self):
        """Test formatting single parameter."""
        params = [{"name": "x", "type": "int"}]

        result = format_parameters(params)

        assert result == "x:int"

    def test_formats_multiple_parameters(self):
        """Test formatting multiple parameters."""
        params = [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "str"},
        ]

        result = format_parameters(params)

        assert "x:int" in result
        assert "y:str" in result
        assert "," in result

    def test_handles_empty_list(self):
        """Test handling empty parameter list."""
        result = format_parameters([])

        assert result is None

    def test_handles_none(self):
        """Test handling None."""
        result = format_parameters(None)

        assert result is None

    def test_handles_missing_type(self):
        """Test handling parameter with missing type."""
        params = [{"name": "x"}]

        result = format_parameters(params)

        assert result == "x"

    def test_handles_missing_name(self):
        """Test handling parameter with missing name."""
        params = [{"type": "int"}]

        result = format_parameters(params)

        assert ":int" in result

    def test_handles_complex_types(self):
        """Test handling complex type annotations."""
        params = [{"name": "data", "type": "List[Dict[str, Any]]"}]

        result = format_parameters(params)

        assert "data:List[Dict[str, Any]]" in result


class TestIntegration:
    """Integration tests for parser module."""

    def test_end_to_end_parsing(self, sample_code_element):
        """Test end-to-end parsing flow."""
        # 1. CodeElement to Document
        doc = process_code_element(sample_code_element)

        assert isinstance(doc, Document)
        assert doc.metadata["name"] == sample_code_element.name

        # 2. Document to TextNode
        nodes = parse_documents_to_nodes([doc])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)
        assert sample_code_element.name in nodes[0].text

    def test_batch_processing(self):
        """Test processing multiple elements."""
        elements = [
            CodeElement(
                type="function",
                name=f"func_{i}",
                docstring=f"Function {i}",
                code=f"def func_{i}(): pass",
                file_path=f"test_{i}.py",
            )
            for i in range(3)
        ]

        # Convert to documents
        docs = [process_code_element(elem) for elem in elements]

        assert len(docs) == 3

        # Parse to nodes
        nodes = parse_documents_to_nodes(docs)

        assert len(nodes) == 3
        assert all(isinstance(n, TextNode) for n in nodes)
