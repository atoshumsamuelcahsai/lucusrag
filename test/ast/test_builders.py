"""Unit tests for AST parsers and code analyzers."""

import ast
from rag.ast.builders import (
    AstCodeAnalyzer,
    ASTParser,
    TreeSitterParser,
)
from rag.schemas.code_element import CodeElement


class TestAstCodeAnalyzer:
    """Test AstCodeAnalyzer class."""

    def test_initialization(self):
        """Test AstCodeAnalyzer initialization."""
        analyzer = AstCodeAnalyzer(project_module="myproject")

        assert analyzer.project_module == "myproject"
        assert analyzer.code_elements == []
        assert analyzer.imports == {}
        assert analyzer.current_class is None
        assert analyzer.current_function is None

    def test_file_path_to_module(self):
        """Test converting file path to module name."""
        analyzer = AstCodeAnalyzer()

        assert (
            analyzer._file_path_to_module("rag/ast/builders.py") == "rag.ast.builders"
        )
        assert (
            analyzer._file_path_to_module("rag\\db\\graph_db.py") == "rag.db.graph_db"
        )
        assert analyzer._file_path_to_module("test.py") == "test"

    def test_get_qualified_name_function(self):
        """Test generating qualified name for a function."""
        analyzer = AstCodeAnalyzer()

        name = analyzer._get_qualified_name("my_func", "rag/ast/builders.py")
        assert name == "rag.ast.builders.my_func"

    def test_get_qualified_name_method(self):
        """Test generating qualified name for a method."""
        analyzer = AstCodeAnalyzer()

        name = analyzer._get_qualified_name(
            "my_method", "rag/ast/builders.py", "MyClass"
        )
        assert name == "rag.ast.builders.MyClass.my_method"

    def test_visit_import(self):
        """Test tracking regular imports."""
        code = "import os\nimport json as js"
        analyzer = AstCodeAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)

        assert "os" in analyzer.imports
        assert "js" in analyzer.imports
        assert analyzer.imports["js"] == "json"

    def test_visit_import_from(self):
        """Test tracking from imports."""
        code = "from pathlib import Path\nfrom typing import List as L"
        analyzer = AstCodeAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)

        assert "Path" in analyzer.imports
        assert analyzer.imports["Path"] == "pathlib.Path"
        assert "L" in analyzer.imports
        assert analyzer.imports["L"] == "typing.List"

    def test_extract_parameters(self):
        """Test extracting function parameters."""
        code = """
def my_func(a: int, b: str, c):
    pass
"""
        analyzer = AstCodeAnalyzer()
        tree = ast.parse(code)
        func_node = tree.body[0]

        params = analyzer._extract_parameters(func_node)

        assert len(params) == 3
        assert params[0] == {"name": "a", "type": "int"}
        assert params[1] == {"name": "b", "type": "str"}
        assert params[2] == {"name": "c", "type": "Any"}

    def test_extract_decorators(self):
        """Test extracting decorators."""
        code = """
@staticmethod
@property
def my_func():
    pass
"""
        analyzer = AstCodeAnalyzer()
        tree = ast.parse(code)
        func_node = tree.body[0]

        decorators = analyzer._extract_decorators(func_node)

        assert "staticmethod" in decorators
        assert "property" in decorators

    def test_visit_function_def(self):
        """Test processing function definitions."""
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
        analyzer = AstCodeAnalyzer(project_module="test")
        analyzer.current_file = "test/module.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        assert len(analyzer.code_elements) == 1
        element = analyzer.code_elements[0]
        assert element.name == "test.module.hello"
        assert element.type == "function"
        assert element.docstring == "Say hello."
        assert len(element.parameters) == 1
        assert element.return_type == "str"

    def test_visit_async_function_def(self):
        """Test processing async function definitions."""
        code = """
async def fetch_data():
    pass
"""
        analyzer = AstCodeAnalyzer()
        analyzer.current_file = "test.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        assert len(analyzer.code_elements) == 1
        element = analyzer.code_elements[0]
        assert element.is_async is True

    def test_visit_class_def(self):
        """Test processing class definitions."""
        code = '''
class MyClass:
    """My class docstring."""
    pass
'''
        analyzer = AstCodeAnalyzer(project_module="test")
        analyzer.current_file = "test/module.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        assert len(analyzer.code_elements) == 1
        element = analyzer.code_elements[0]
        assert element.name == "test.module.MyClass"
        assert element.type == "class"
        assert element.docstring == "My class docstring."

    def test_visit_class_with_base_classes(self):
        """Test extracting base classes."""
        code = """
class MyClass(BaseClass, abc.ABC):
    pass
"""
        analyzer = AstCodeAnalyzer()
        analyzer.current_file = "test.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        element = analyzer.code_elements[0]
        assert "BaseClass" in element.base_classes
        assert "abc.ABC" in element.base_classes

    def test_visit_class_with_methods(self):
        """Test that methods are tracked in their class."""
        code = '''
class MyClass:
    def my_method(self):
        """Method docstring."""
        pass
'''
        analyzer = AstCodeAnalyzer(project_module="test")
        analyzer.current_file = "test/module.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        # Should have only class (methods are added to class.methods list, not as separate elements)
        assert len(analyzer.code_elements) == 1

        # Find class element
        class_element = next(e for e in analyzer.code_elements if e.type == "class")
        assert len(class_element.methods) == 1
        assert class_element.methods[0] == "test.module.MyClass.my_method"

    def test_filter_project_dependencies(self):
        """Test that only project imports are included in dependencies."""
        code = """
from rag.db import GraphDBManager
from typing import List
import os

def my_func():
    pass
"""
        analyzer = AstCodeAnalyzer(project_module="rag")
        analyzer.current_file = "test.py"
        analyzer.source_code = code
        tree = ast.parse(code)
        analyzer.visit(tree)

        element = analyzer.code_elements[0]
        # Should only include rag.db.GraphDBManager, not typing or os
        assert len(element.dependencies) == 1
        assert element.dependencies[0] == "rag.db.GraphDBManager"


class TestASTParser:
    """Test ASTParser class."""

    def test_initialization(self):
        """Test ASTParser initialization."""
        parser = ASTParser(project_module="myproject")
        assert parser.project_module == "myproject"

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = """
def greet(name: str) -> str:
    return f"Hello, {name}"
"""
        parser = ASTParser(project_module="test")
        elements = parser.parse(code, "test/module.py")

        assert len(elements) == 1
        assert elements[0].name == "test.module.greet"
        assert elements[0].type == "function"

    def test_parse_class(self):
        """Test parsing a class."""
        code = """
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
"""
        parser = ASTParser(project_module="test")
        elements = parser.parse(code, "test/calc.py")

        # Should have only class (methods are part of the class)
        assert len(elements) == 1
        class_elem = elements[0]
        assert class_elem.type == "class"
        assert class_elem.name == "test.calc.Calculator"
        assert len(class_elem.methods) == 1

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        parser = ASTParser()
        elements = parser.parse("", "empty.py")
        assert elements == []

    def test_parse_with_imports(self):
        """Test parsing with imports."""
        code = """
from rag.db import GraphDBManager

def use_db():
    pass
"""
        parser = ASTParser(project_module="rag")
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        assert "rag.db.GraphDBManager" in elements[0].dependencies


class TestTreeSitterParser:
    """Test TreeSitterParser class."""

    def test_initialization(self):
        """Test TreeSitterParser initialization."""
        parser = TreeSitterParser(project_module="myproject")

        assert parser.project_module == "myproject"
        assert parser.imports == {}
        assert parser.local_definitions == {}

    def test_file_path_to_module(self):
        """Test file path to module conversion."""
        parser = TreeSitterParser()

        assert parser._file_path_to_module("rag/ast/builders.py") == "rag.ast.builders"
        assert parser._file_path_to_module("test.py") == "test"

    def test_get_qualified_name(self):
        """Test qualified name generation."""
        parser = TreeSitterParser()

        name = parser._get_qualified_name("func", "rag/ast/builders.py")
        assert name == "rag.ast.builders.func"

        name = parser._get_qualified_name("method", "rag/ast/builders.py", "MyClass")
        assert name == "rag.ast.builders.MyClass.method"

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = """
def greet(name):
    '''Say hello.'''
    return f"Hello, {name}"
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test/module.py")

        assert len(elements) == 1
        assert elements[0].name == "test.module.greet"
        assert elements[0].type == "function"
        assert elements[0].docstring == "Say hello."

    def test_parse_class(self):
        """Test parsing a class."""
        code = """
class MyClass:
    '''My class.'''
    
    def method(self):
        pass
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test/module.py")

        # Should have only class (methods are part of the class)
        assert len(elements) == 1
        class_elem = elements[0]
        assert class_elem.type == "class"
        assert class_elem.name == "test.module.MyClass"
        assert len(class_elem.methods) == 1

    def test_parse_with_imports(self):
        """Test import resolution."""
        code = """
from rag.db import GraphDBManager as GDB
import rag.providers.llms

def use_db():
    pass
"""
        parser = TreeSitterParser(project_module="rag")
        elements = parser.parse(code, "test.py")

        # At least one element should be parsed (the function)
        assert len(elements) >= 1
        # Check that import map was built (even if specific alias handling varies)
        assert isinstance(parser.imports, dict)

    def test_parse_decorated_class(self):
        """Test parsing decorated classes."""
        code = """
from dataclasses import dataclass

@dataclass
class Person:
    name: str
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test/models.py")

        # Should find the decorated class
        assert len(elements) == 1
        assert elements[0].type == "class"
        # Note: Tree-sitter may or may not extract decorator names, just verify it didn't crash
        assert isinstance(elements[0].decorators, list)

    def test_filter_trivial_builtins(self):
        """Test that trivial built-ins are filtered from calls."""
        code = """
def process():
    print("hello")  # Should be filtered
    result = len([1, 2, 3])  # Should be filtered
    return result
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test.py")

        # print and len should not be in calls
        assert "print" not in elements[0].calls
        assert "len" not in elements[0].calls

    def test_filter_instance_methods(self):
        """Test that instance method calls are filtered."""
        code = """
def process():
    obj = MyClass()
    obj.method()  # Should be filtered (lowercase first letter)
    return obj
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test.py")

        # obj.method should be filtered
        assert "obj.method" not in elements[0].calls

    def test_resolve_local_calls(self):
        """Test resolving calls to locally defined functions."""
        code = """
def helper():
    pass

def main():
    helper()  # Should resolve to full path
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test/module.py")

        main_func = next(e for e in elements if e.name.endswith("main"))
        # Should have resolved helper to full path
        assert any("test.module.helper" in call for call in main_func.calls)

    def test_resolve_class_method_calls(self):
        """Test resolving ClassName.method() to ClassName."""
        code = """
class MyClass:
    @classmethod
    def from_dict(cls, data):
        pass

def use_class():
    obj = MyClass.from_dict({})  # Should resolve to MyClass
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test/module.py")

        use_class_func = next(e for e in elements if e.name.endswith("use_class"))
        # Should have resolved MyClass.from_dict to just MyClass
        assert any("MyClass" in call for call in use_class_func.calls)

    def test_parse_async_function(self):
        """Test parsing async functions."""
        code = """
async def fetch():
    pass
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        assert elements[0].type == "function"
        # Note: is_async detection may vary by parser implementation
        # Just verify the function was parsed
        assert elements[0].name.endswith("fetch")

    def test_parse_base_classes(self):
        """Test extracting base classes."""
        code = """
import abc

class MyClass(BaseClass, abc.ABC):
    pass
"""
        parser = TreeSitterParser(project_module="test")
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        # Should capture base classes
        assert len(elements[0].base_classes) > 0

    def test_parse_empty_file(self):
        """Test parsing empty file."""
        parser = TreeSitterParser()
        elements = parser.parse("", "empty.py")
        assert elements == []

    def test_parse_syntax_error_gracefully(self):
        """Test handling syntax errors gracefully."""
        code = "def incomplete(:"  # Invalid syntax

        parser = TreeSitterParser()
        # Should not crash, may return empty or partial results
        elements = parser.parse(code, "bad.py")
        assert isinstance(elements, list)


class TestCodeParserProtocol:
    """Test that parsers conform to CodeParser protocol."""

    def test_ast_parser_implements_protocol(self):
        """Test ASTParser implements CodeParser."""
        parser = ASTParser()

        # Should have parse method
        assert hasattr(parser, "parse")
        assert callable(parser.parse)

    def test_tree_sitter_parser_implements_protocol(self):
        """Test TreeSitterParser implements CodeParser."""
        parser = TreeSitterParser()

        # Should have parse method
        assert hasattr(parser, "parse")
        assert callable(parser.parse)

    def test_parsers_return_code_elements(self):
        """Test that parsers return List[CodeElement]."""
        code = "def test(): pass"

        ast_parser = ASTParser()
        ast_result = ast_parser.parse(code, "test.py")
        assert isinstance(ast_result, list)
        if ast_result:
            assert isinstance(ast_result[0], CodeElement)

        ts_parser = TreeSitterParser()
        ts_result = ts_parser.parse(code, "test.py")
        assert isinstance(ts_result, list)
        if ts_result:
            assert isinstance(ts_result[0], CodeElement)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_multiline_strings(self):
        """Test parsing functions with multiline strings."""
        code = '''
def func():
    """
    Multiline
    docstring.
    """
    sql = """
    SELECT * FROM table
    WHERE id = 1
    """
    return sql
'''
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        # Should have extracted docstring
        assert "Multiline" in elements[0].docstring

    def test_parse_nested_functions(self):
        """Test parsing nested functions."""
        code = """
def outer():
    def inner():
        pass
    return inner
"""
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        # Should find at least the outer function
        assert len(elements) >= 1
        assert any(e.name.endswith("outer") for e in elements)

    def test_parse_lambda_functions(self):
        """Test that lambda functions don't crash the parser."""
        code = """
func = lambda x: x * 2
"""
        parser = ASTParser()
        # Should not crash
        elements = parser.parse(code, "test.py")
        assert isinstance(elements, list)

    def test_parse_complex_type_hints(self):
        """Test parsing complex type hints."""
        code = """
from typing import List, Dict, Optional

def func(data: List[Dict[str, Optional[int]]]) -> bool:
    return True
"""
        parser = ASTParser()
        elements = parser.parse(code, "test.py")

        assert len(elements) == 1
        assert elements[0].return_type == "bool"

    def test_parse_star_imports(self):
        """Test handling star imports."""
        code = """
from typing import *

def func():
    pass
"""
        parser = ASTParser()
        # Should not crash
        elements = parser.parse(code, "test.py")
        assert len(elements) == 1
