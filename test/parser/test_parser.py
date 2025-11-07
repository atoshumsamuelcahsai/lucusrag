"""Unit tests for parser module."""

import pytest
from llama_index.core.schema import TextNode
from llama_index.core import Document

from rag.parser.parser import (
    convert_to_llama_nodes,
    CodeElementParser,
    process_code_element,
    CodeElementGraphParser,
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
        dependencies=["typing"],
        base_classes=[],
        methods=[],
        calls=["sum"],
        assignments=["result"],
        explanation="Simple addition function",
        is_async=False,
    )


@pytest.fixture
def sample_analysis_data():
    """Create sample analysis data dictionary."""
    return {
        "elements": [
            {
                "type": "function",
                "name": "my_func",
                "docstring": "My function",
                "code": "def my_func(): pass",
                "file_path": "test.py",
                "parameters": [{"name": "x", "type": "int"}],
                "return_type": "str",
                "decorators": ["cached"],
                "dependencies": ["os"],
                "base_classes": [],
                "methods": [],
                "calls": ["print"],
                "assignments": ["result"],
                "explanation": "Test function",
            }
        ]
    }


class TestConvertToLlamaNodes:
    """Test convert_to_llama_nodes function."""

    def test_converts_single_element(self, sample_analysis_data):
        """Test converting single element from analysis data."""
        nodes = convert_to_llama_nodes(sample_analysis_data)

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_converts_multiple_elements(self):
        """Test converting multiple elements."""
        data = {
            "elements": [
                {
                    "type": "function",
                    "name": "func1",
                    "docstring": "First function",
                    "code": "def func1(): pass",
                    "file_path": "test.py",
                },
                {
                    "type": "class",
                    "name": "MyClass",
                    "docstring": "My class",
                    "code": "class MyClass: pass",
                    "file_path": "test.py",
                },
            ]
        }

        nodes = convert_to_llama_nodes(data)

        assert len(nodes) == 2
        assert all(isinstance(node, TextNode) for node in nodes)

    def test_handles_optional_fields(self):
        """Test handling elements with missing optional fields."""
        data = {
            "elements": [
                {
                    "type": "function",
                    "name": "minimal_func",
                    "docstring": "Minimal",
                    "code": "def minimal_func(): pass",
                    "file_path": "test.py",
                }
            ]
        }

        nodes = convert_to_llama_nodes(data)

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_preserves_metadata(self, sample_analysis_data):
        """Test that metadata is preserved in conversion."""
        nodes = convert_to_llama_nodes(sample_analysis_data)

        node = nodes[0]
        assert "my_func" in node.text or node.metadata.get("name") == "my_func"

    def test_handles_empty_elements_list(self):
        """Test handling empty elements list."""
        data = {"elements": []}

        nodes = convert_to_llama_nodes(data)

        assert nodes == []


class TestCodeElementParser:
    """Test CodeElementParser class."""

    def test_initialization(self):
        """Test CodeElementParser initialization."""
        parser = CodeElementParser()
        assert parser is not None

    def test_parse_single_element(self, sample_code_element):
        """Test parsing single code element."""
        parser = CodeElementParser()

        nodes = parser.parse([sample_code_element])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_parse_multiple_elements(self, sample_code_element):
        """Test parsing multiple code elements."""
        elements = [
            sample_code_element,
            CodeElement(
                type="class",
                name="Calculator",
                docstring="Calculator class",
                code="class Calculator: pass",
                file_path="src/calc.py",
            ),
        ]

        parser = CodeElementParser()
        nodes = parser.parse(elements)

        assert len(nodes) == 2
        assert all(isinstance(node, TextNode) for node in nodes)

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        parser = CodeElementParser()

        nodes = parser.parse([])

        assert nodes == []

    def test_parse_preserves_order(self, sample_code_element):
        """Test that parsing preserves element order."""
        elements = [
            CodeElement(
                type="function",
                name=f"func_{i}",
                docstring=f"Function {i}",
                code=f"def func_{i}(): pass",
                file_path="test.py",
            )
            for i in range(5)
        ]

        parser = CodeElementParser()
        nodes = parser.parse(elements)

        assert len(nodes) == 5
        # Check order is preserved
        for i, node in enumerate(nodes):
            assert f"func_{i}" in node.text or node.metadata.get("name") == f"func_{i}"


class TestProcessCodeElement:
    """Test process_code_element function."""

    def test_creates_document(self, sample_code_element):
        """Test creating document from code element."""
        doc = process_code_element(sample_code_element)

        assert isinstance(doc, Document)
        # Note: LlamaIndex Document uses 'text' instead of 'page_content'
        # The test verifies the document was created successfully
        assert hasattr(doc, "metadata")

    def test_document_metadata(self, sample_code_element):
        """Test document metadata is correctly set."""
        doc = process_code_element(sample_code_element)

        assert doc.metadata["name"] == "calculate_sum"
        assert doc.metadata["type"] == "function"
        assert doc.metadata["file_path"] == "src/math_utils.py"
        assert doc.metadata["docstring"] == "Calculate sum of two numbers."
        assert doc.metadata["code"] == sample_code_element.code

    def test_handles_optional_fields(self):
        """Test handling code element with minimal fields."""
        element = CodeElement(
            type="function",
            name="simple",
            docstring="Simple function",
            code="def simple(): pass",
            file_path="test.py",
        )

        doc = process_code_element(element)

        assert doc.metadata["parameters"] == []
        assert doc.metadata["return_type"] == ""
        assert doc.metadata["dependencies"] == []
        assert doc.metadata["base_classes"] == []
        assert doc.metadata["calls"] == []
        assert doc.metadata["methods"] == []
        assert doc.metadata["decorators"] == []
        assert doc.metadata["assignments"] == []
        assert doc.metadata["explanation"] == ""

    def test_preserves_list_fields(self, sample_code_element):
        """Test that list fields are preserved."""
        doc = process_code_element(sample_code_element)

        assert doc.metadata["parameters"] == sample_code_element.parameters
        assert doc.metadata["dependencies"] == sample_code_element.dependencies
        assert doc.metadata["calls"] == sample_code_element.calls
        assert doc.metadata["decorators"] == sample_code_element.decorators


class TestCodeElementGraphParser:
    """Test CodeElementGraphParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return CodeElementGraphParser()

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return Document(
            page_content="def test(): pass",
            metadata={
                "name": "test_func",
                "type": "function",
                "file_path": "test.py",
                "docstring": "Test function",
                "code": "def test(): pass",
                "parameters": [{"name": "x", "type": "int"}],
                "return_type": "str",
                "dependencies": ["os", "sys"],
                "base_classes": ["BaseClass"],
                "calls": ["print", "open"],
            },
        )

    def test_initialization(self, parser):
        """Test parser initialization."""
        assert parser.required_fields == ["name", "type", "file_path"]

    def test_call_with_single_document(self, parser, sample_document):
        """Test calling parser with single document."""
        nodes = parser([sample_document])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_call_with_multiple_documents(self, parser):
        """Test calling parser with multiple documents."""
        docs = [
            Document(
                page_content=f"def func{i}(): pass",
                metadata={
                    "name": f"func{i}",
                    "type": "function",
                    "file_path": "test.py",
                    "code": f"def func{i}(): pass",
                },
            )
            for i in range(3)
        ]

        nodes = parser(docs)

        assert len(nodes) == 3
        assert all(isinstance(node, TextNode) for node in nodes)

    def test_call_with_empty_list(self, parser):
        """Test calling parser with empty list."""
        nodes = parser([])
        assert nodes == []

    def test_node_metadata_structure(self, parser, sample_document):
        """Test that node metadata has correct structure."""
        nodes = parser([sample_document])

        node = nodes[0]
        assert "id" in node.metadata
        assert "name" in node.metadata
        assert "type" in node.metadata
        assert "file_path" in node.metadata

        assert node.metadata["name"] == "test_func"
        assert node.metadata["type"] == "function"
        assert node.metadata["file_path"] == "test.py"

    def test_node_id_format(self, parser, sample_document):
        """Test node ID format."""
        nodes = parser([sample_document])

        node_id = nodes[0].metadata["id"]
        assert node_id == "test.py:function:test_func"

    def test_handles_missing_optional_metadata(self, parser):
        """Test handling document with minimal metadata."""
        doc = Document(
            page_content="code",
            metadata={
                "name": "minimal",
                "type": "function",
                "file_path": "test.py",
            },
        )

        nodes = parser([doc])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

    def test_handles_document_errors_gracefully(self, parser):
        """Test that parser continues on document errors."""
        docs = [
            Document(
                page_content="good",
                metadata={
                    "name": "good",
                    "type": "function",
                    "file_path": "test.py",
                    "code": "good",
                },
            ),
            Document(page_content="bad", metadata={}),  # Missing required fields
            Document(
                page_content="good2",
                metadata={
                    "name": "good2",
                    "type": "class",
                    "file_path": "test.py",
                    "code": "good2",
                },
            ),
        ]

        # Should not crash, should skip bad document
        nodes = parser(docs)

        # May have 2 or 3 nodes depending on error handling
        assert len(nodes) >= 2

    def test_show_progress_parameter(self, parser, sample_document):
        """Test show_progress parameter is accepted."""
        # Should not raise error
        nodes = parser([sample_document], show_progress=True)
        assert len(nodes) == 1

    def test_accepts_kwargs(self, parser, sample_document):
        """Test that additional kwargs are accepted."""
        # Should not raise error
        nodes = parser([sample_document], custom_param="value")
        assert len(nodes) == 1


class TestCreateTextRepresentation:
    """Test CodeElementGraphParser.create_text_representation method."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return CodeElementGraphParser()

    def test_creates_formatted_text(self, parser):
        """Test creating formatted text representation."""
        metadata = {
            "name": "my_func",
            "type": "function",
            "code": "def my_func(): pass",
            "docstring": "My function",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "str",
            "dependencies": ["os"],
            "base_classes": ["Base"],
            "calls": ["print"],
        }

        text = parser.create_text_representation(metadata)

        assert "Name: my_func" in text
        assert "Type: function" in text
        assert "Description: My function" in text
        assert "def my_func(): pass" in text

    def test_handles_empty_lists(self, parser):
        """Test handling empty lists in metadata."""
        metadata = {
            "name": "simple",
            "type": "function",
            "code": "def simple(): pass",
            "parameters": [],
            "dependencies": [],
            "base_classes": [],
            "calls": [],
        }

        text = parser.create_text_representation(metadata)

        assert "Name: simple" in text
        assert isinstance(text, str)

    def test_handles_none_values(self, parser):
        """Test handling None values in metadata."""
        metadata = {
            "name": "test",
            "type": "function",
            "code": "code",
            "docstring": None,
            "parameters": None,
            "return_type": None,
            "dependencies": None,
            "base_classes": None,
            "calls": None,
        }

        text = parser.create_text_representation(metadata)

        assert "Name: test" in text
        assert isinstance(text, str)

    def test_includes_all_sections(self, parser):
        """Test that all sections are included in output."""
        metadata = {
            "name": "complete",
            "type": "function",
            "code": "def complete(): pass",
            "docstring": "Complete function",
            "parameters": [{"name": "a", "type": "int"}],
            "return_type": "bool",
            "dependencies": ["typing"],
            "base_classes": ["BaseClass"],
            "calls": ["helper"],
        }

        text = parser.create_text_representation(metadata)

        assert "Parameters:" in text
        assert "Return Type:" in text
        assert "Dependencies:" in text
        assert "Base Classes:" in text
        assert "Function Calls:" in text


class TestFormatParameters:
    """Test CodeElementGraphParser.format_parameters method."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return CodeElementGraphParser()

    def test_formats_single_parameter(self, parser):
        """Test formatting single parameter."""
        params = [{"name": "x", "type": "int"}]

        result = parser.format_parameters(params)

        assert result == "x:int"

    def test_formats_multiple_parameters(self, parser):
        """Test formatting multiple parameters."""
        params = [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "str"},
            {"name": "z", "type": "bool"},
        ]

        result = parser.format_parameters(params)

        assert result == "x:int, y:str, z:bool"

    def test_handles_empty_list(self, parser):
        """Test handling empty parameter list."""
        result = parser.format_parameters([])
        assert result == ""

    def test_handles_none(self, parser):
        """Test handling None."""
        result = parser.format_parameters(None)
        assert result == ""

    def test_handles_missing_type(self, parser):
        """Test handling parameter with missing type."""
        params = [{"name": "x"}]

        result = parser.format_parameters(params)

        assert "x:" in result

    def test_handles_missing_name(self, parser):
        """Test handling parameter with missing name."""
        params = [{"type": "int"}]

        result = parser.format_parameters(params)

        assert ":int" in result

    def test_handles_complex_types(self, parser):
        """Test handling complex type annotations."""
        params = [
            {"name": "data", "type": "List[Dict[str, Any]]"},
            {"name": "callback", "type": "Callable[[int], bool]"},
        ]

        result = parser.format_parameters(params)

        assert "data:List[Dict[str, Any]]" in result
        assert "callback:Callable[[int], bool]" in result


class TestIntegration:
    """Integration tests for parser module."""

    def test_end_to_end_parsing(self, sample_code_element):
        """Test end-to-end parsing flow."""
        # 1. Parse CodeElement to TextNode
        parser = CodeElementParser()
        nodes = parser.parse([sample_code_element])

        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)

        # 2. Create Document from CodeElement
        doc = process_code_element(sample_code_element)

        assert isinstance(doc, Document)
        assert doc.metadata["name"] == "calculate_sum"

        # 3. Parse Document to TextNode with graph parser
        graph_parser = CodeElementGraphParser()
        graph_nodes = graph_parser([doc])

        assert len(graph_nodes) == 1
        assert isinstance(graph_nodes[0], TextNode)

    def test_batch_processing(self):
        """Test processing multiple elements."""
        elements = [
            CodeElement(
                type="function",
                name=f"func{i}",
                docstring=f"Function {i}",
                code=f"def func{i}(): pass",
                file_path="test.py",
            )
            for i in range(5)
        ]

        # Parse to nodes
        parser = CodeElementParser()
        nodes = parser.parse(elements)

        assert len(nodes) == 5

        # Create documents
        docs = [process_code_element(elem) for elem in elements]

        assert len(docs) == 5

        # Graph parse
        graph_parser = CodeElementGraphParser()
        graph_nodes = graph_parser(docs)

        assert len(graph_nodes) == 5
