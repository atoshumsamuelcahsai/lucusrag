"""Tests for AST builder functionality."""

import json
import tempfile
from pathlib import Path
from rag.ast.ast_builder import ASTParser, analyze_and_store_python_files
from rag.schemas.code_element import CodeElement


class TestASTParser:
    """Test AST parsing functionality."""

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = '''
def hello_world(name: str) -> str:
    """Say hello."""
    return f"Hello {name}"
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        func = elements[0]
        assert func.name == "hello_world"
        assert func.type == "function"
        assert func.docstring == "Say hello."
        assert func.return_type == "str"
        assert len(func.parameters) == 1
        assert func.parameters[0]["name"] == "name"
        assert func.parameters[0]["type"] == "str"

    def test_parse_simple_class(self):
        """Test parsing a simple class."""
        code = '''
class MyClass:
    """A test class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def get_value(self) -> int:
        return self.value
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        cls = elements[0]
        assert cls.name == "MyClass"
        assert cls.type == "class"
        assert cls.docstring == "A test class."
        assert len(cls.methods) == 2
        assert "__init__" in cls.methods
        assert "get_value" in cls.methods

    def test_parse_function_with_calls(self):
        """Test extracting function calls."""
        code = '''
import os

def process_data(path: str) -> dict:
    """Process a file."""
    data = os.path.exists(path)
    result = json.loads(data)
    return result
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        func = elements[0]
        assert "os.path.exists" in func.calls
        assert "json.loads" in func.calls

    def test_parse_class_with_inheritance(self):
        """Test extracting base classes."""
        code = '''
class Child(Parent, Mixin):
    """A child class."""
    pass
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        cls = elements[0]
        assert "Parent" in cls.base_classes
        assert "Mixin" in cls.base_classes

    def test_parse_async_function(self):
        """Test detecting async functions."""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    return await client.get(url)
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        func = elements[0]
        assert func.is_async is True


class TestCodeElement:
    """Test CodeElement dataclass."""

    def test_code_element_creation(self):
        """Test creating a CodeElement instance."""
        info = CodeElement(
            name="test_func",
            type="function",
            docstring="Test function",
            code="def test_func(): pass",
            file_path="test.py",
        )
        assert info.name == "test_func"
        assert info.type == "function"
        assert info.explanation == ""

    def test_code_element_to_dict(self):
        """Test serializing CodeElement to dict."""
        info = CodeElement(
            name="test_func",
            type="function",
            docstring="Test function",
            code="def test_func(): pass",
            file_path="test.py",
            parameters=[{"name": "x", "type": "int"}],
            return_type="str",
        )
        data = info.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test_func"
        assert data["type"] == "function"
        assert data["parameters"] == [{"name": "x", "type": "int"}]
        assert data["return_type"] == "str"

    def test_code_element_from_dict(self):
        """Test deserializing CodeElement from dict."""
        data = {
            "name": "test_func",
            "type": "function",
            "docstring": "Test",
            "code": "def test_func(): pass",
            "file_path": "test.py",
            "explanation": "",
            "parameters": [],
            "return_type": "None",
            "decorators": [],
            "dependencies": [],
            "base_classes": [],
            "methods": [],
            "calls": [],
            "assignments": [],
            "is_async": False,
        }
        info = CodeElement.from_dict(data)
        assert info.name == "test_func"
        assert info.type == "function"


class TestAnalyzeAndStore:
    """Test end-to-end analysis and storage."""

    def test_analyze_simple_project(self):
        """Test analyzing a simple Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            (src_dir / "module1.py").write_text(
                '''
def func1():
    """Function 1."""
    pass
'''
            )

            (src_dir / "module2.py").write_text(
                '''
class MyClass:
    """Test class."""
    
    def method1(self):
        pass
'''
            )

            # Run analysis
            output_dir = Path(tmpdir) / "output"
            result = analyze_and_store_python_files(
                root_dir=str(src_dir), output_dir=str(output_dir), parser_type="ast"
            )

            # Verify output
            assert Path(result).exists()
            results_dir = Path(result) / "results_ast"
            assert results_dir.exists()

            # Check JSON files were created
            json_files = list(results_dir.glob("*.json"))
            assert len(json_files) >= 2  # At least func1 and MyClass

            # Verify JSON structure
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                    assert "name" in data
                    assert "type" in data
                    assert "file_path" in data
                    assert data["type"] in ["function", "class"]

    def test_analyze_preserves_progress(self):
        """Test that progress file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            (src_dir / "test.py").write_text("def test(): pass")

            output_dir = Path(tmpdir) / "output"
            analyze_and_store_python_files(
                root_dir=str(src_dir), output_dir=str(output_dir), parser_type="ast"
            )

            # Check progress file exists (in root of output_dir, not results_ast)
            progress_file = output_dir / "progress.json"
            assert progress_file.exists()

            with open(progress_file) as f:
                progress = json.load(f)
                assert "processed_files" in progress
                assert len(progress["processed_files"]) > 0
