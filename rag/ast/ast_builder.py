import ast
from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from rag.schemas.code_element import CodeElement
import logging

logger = logging.getLogger(__name__)

# Optional dependencies - only import if needed
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


class RateLimit:
    def __init__(self):
        self.requests = 0
        self.tokens = 0
        self.last_reset = datetime.now()
        self.daily_requests = 0
        self.daily_tokens = 0
        self.day_start = datetime.now().date()


class CodeParser(ABC):
    """Abstract base for code parsers."""

    @abstractmethod
    def parse(self, source_code: str, file_path: str) -> List[CodeElement]:
        """Parse source code and extract code elements."""
        pass


class AstCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.code_elements: List[CodeElement] = []
        self.current_file = ""
        self.imports: Dict[str, str] = {}  # Track imports for dependency analysis
        self.current_class: Optional[str] = None
        self.current_function: Optional[CodeElement] = None
        self.source_code = ""  # Add this to store the source code

    def visit_Import(self, node):
        """Track regular imports."""
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from imports."""
        module = node.module or ""
        for name in node.names:
            full_name = f"{module}.{name.name}" if module else name.name
            self.imports[name.asname or name.name] = full_name
        self.generic_visit(node)

    def _get_type_annotation(self, node: ast.AST) -> str:
        """Convert AST annotation to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            value = self._get_type_annotation(node.value)
            slice_value = self._get_type_annotation(node.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_type_annotation(node.value)}.{node.attr}"
        return "Any"

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Extract function parameters with their type hints."""
        parameters = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type": (
                    self._get_type_annotation(arg.annotation)
                    if arg.annotation
                    else "Any"
                ),
            }
            parameters.append(param_info)
        return parameters

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """Extract decorator names."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
        return decorators

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        is_method = self.current_class is not None

        code_info = CodeElement(
            name=node.name,
            type="method" if is_method else "function",
            docstring=ast.get_docstring(node) or "",
            code=self._get_source_segment(node),
            file_path=self.current_file,
            parameters=self._extract_parameters(node),
            return_type=(
                self._get_type_annotation(node.returns) if node.returns else None
            ),
            decorators=self._extract_decorators(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

        # Store the current function for analyzing its contents
        self.current_function = code_info
        self.generic_visit(node)
        self.current_function = None

        if is_method:
            # Add to current class's methods
            for element in self.code_elements:
                if element.type == "class" and element.name == self.current_class:
                    element.methods.append(node.name)
        else:
            self.code_elements.append(code_info)

    def visit_AsyncFunctionDef(self, node):
        """Handle async functions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Process class definitions."""
        previous_class = self.current_class
        self.current_class = node.name

        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle nested attributes more carefully
                parts = []
                current = base
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    base_classes.append(".".join(reversed(parts)))
                else:
                    # Handle other cases gracefully
                    base_classes.append(str(base))

        code_info = CodeElement(
            name=node.name,
            type="class",
            docstring=ast.get_docstring(node) or "",
            code=self._get_source_segment(node),
            file_path=self.current_file,
            decorators=self._extract_decorators(node),
            base_classes=base_classes,
        )

        self.code_elements.append(code_info)
        self.generic_visit(node)
        self.current_class = previous_class

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle nested attributes (e.g., a.b.c())
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None

    def visit_Call(self, node):
        """Track function calls."""
        if self.current_function:
            call_name = self._get_call_name(node)
            if call_name:
                self.current_function.calls.append(call_name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Track assignments."""
        if self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.current_function.assignments.append(target.id)
        self.generic_visit(node)

    def _get_source_segment(self, node):
        """Get the source code for a node."""
        try:
            return ast.get_source_segment(self.source_code, node) or ""
        except Exception:
            return ""


class ProgressTracker:
    def __init__(
        self, total_elements: int, total_expected_tokens: int, total_cost: float
    ):
        self.start_time = datetime.now()
        self.total_elements = total_elements
        self.processed_elements = 0
        self.total_expected_tokens = total_expected_tokens
        self.tokens_used = 0
        self.total_expected_cost = total_cost
        self.cost_so_far = 0
        self.last_update = self.start_time

    def update(self, tokens_used: int, is_element_complete: bool = True):
        self.tokens_used += tokens_used
        if is_element_complete:
            self.processed_elements += 1

        # Update every 30 seconds or every 50 elements
        now = datetime.now()
        if (now - self.last_update).seconds >= 30 or self.processed_elements % 50 == 0:
            self._print_progress()
            self.last_update = now

    def _print_progress(self):
        elapsed_time = datetime.now() - self.start_time
        if self.processed_elements > 0:
            avg_time_per_element = elapsed_time / self.processed_elements
            estimated_remaining_time = avg_time_per_element * (
                self.total_elements - self.processed_elements
            )
        else:
            estimated_remaining_time = timedelta(0)

        # Calculate costs
        input_cost = (self.tokens_used / 1000) * 0.00025  # Claude input cost
        output_cost = (self.tokens_used / 1000) * 0.00075  # Claude output cost
        current_cost = input_cost + output_cost

        logger.info("\n" + "=" * 50)
        logger.info(f"Progress Update ({datetime.now().strftime('%H:%M:%S')})")
        logger.info(
            f"Elements: {self.processed_elements:,}/{self.total_elements:,} ({(self.processed_elements/self.total_elements*100):.1f}%)"
        )
        logger.info(
            f"Tokens Used: {self.tokens_used:,}/{self.total_expected_tokens:,} ({(self.tokens_used/self.total_expected_tokens*100):.1f}%)"
        )
        logger.info(f"Cost: ${current_cost:.2f}/${self.total_expected_cost:.2f}")
        logger.info(f"Time Elapsed: {str(elapsed_time).split('.')[0]}")
        logger.info(
            f"Estimated Remaining: {str(estimated_remaining_time).split('.')[0]}"
        )
        logger.info("=" * 50)


class ASTParser(CodeParser):
    """Python's built-in AST parser."""

    def parse(self, source_code: str, file_path: str) -> List[CodeElement]:
        analyzer = AstCodeAnalyzer()
        tree = ast.parse(source_code)
        analyzer.current_file = file_path
        analyzer.source_code = source_code
        analyzer.visit(tree)
        return analyzer.code_elements


class TreeSitterParser(CodeParser):
    """Tree-sitter based parser with relationship extraction.

    Note: Requires tree-sitter and tree-sitter-python packages.
    Install with: pip install tree-sitter tree-sitter-python
    """

    def __init__(self):
        if not HAS_TREE_SITTER:
            raise ImportError(
                "tree-sitter not installed. "
                "Install with: pip install tree-sitter tree-sitter-python\n"
                "Or use --parser-type ast instead."
            )
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser()
        self.parser.language = self.PY_LANGUAGE
        self.source_code = ""

    def _extract_function_calls(self, node) -> List[str]:
        """Extract function calls with their full qualified names when possible."""
        calls = []
        for child in node.children:
            if child.type == "call":
                func_node = child.child_by_field_name("function")
                if func_node:
                    if func_node.type == "identifier":
                        # Simple function call
                        calls.append(func_node.text.decode("utf8"))
                    elif func_node.type == "attribute":
                        # Method or qualified function call (e.g., model.pie.UserHeatDay.async_init)
                        parts = []
                        current = func_node
                        while current:
                            if current.type == "attribute":
                                attr = current.child_by_field_name("attribute")
                                if attr:
                                    parts.insert(0, attr.text.decode("utf8"))
                                current = current.child_by_field_name("object")
                            elif current.type == "identifier":
                                parts.insert(0, current.text.decode("utf8"))
                                current = None
                            else:
                                break
                        qualified_name = ".".join(parts)
                        calls.append(qualified_name)

            # Recursively check children
            calls.extend(self._extract_function_calls(child))
        return calls

    def _extract_dependencies(self, node) -> List[str]:
        """Extract dependencies from imports."""
        deps = []
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    deps.append(child.text.decode("utf8"))
        elif node.type == "import_from_statement":
            module = None
            for child in node.children:
                if child.type == "dotted_name":
                    module = child.text.decode("utf8")
                elif child.type == "import_statement":
                    for import_child in child.children:
                        if import_child.type == "dotted_name":
                            name = import_child.text.decode("utf8")
                            if module:
                                deps.append(f"{module}.{name}")
                            else:
                                deps.append(name)
        return deps

    def _process_function_node(self, node, file_path: str) -> Optional[CodeElement]:
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            body_node = node.child_by_field_name("body")
            if not body_node:
                return None

            # Extract code text
            start_byte = node.start_byte
            end_byte = node.end_byte
            code = self.source_code[start_byte:end_byte]

            # Extract function calls and dependencies
            calls = self._extract_function_calls(body_node)
            dependencies = []
            for child in node.children:
                dependencies.extend(self._extract_dependencies(child))

            return CodeElement(
                name=name_node.text.decode("utf8"),
                type="function",
                docstring=self._extract_docstring(body_node),
                code=code,
                file_path=file_path,
                parameters=self._extract_parameters(node),
                calls=calls,
                dependencies=dependencies,
                is_async=node.type == "async_function_definition",
            )
        except Exception as e:
            logger.exception(f"Error processing function node: {e}")
            return None

    def _process_class_node(self, node, file_path: str) -> Optional[CodeElement]:
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            body_node = node.child_by_field_name("body")
            if not body_node:
                return None

            # Extract code text
            start_byte = node.start_byte
            end_byte = node.end_byte
            code = self.source_code[start_byte:end_byte]

            # Extract base classes
            base_classes = []
            bases_node = node.child_by_field_name("bases")
            if bases_node:
                for base in bases_node.children:
                    if base.type == "identifier":
                        base_classes.append(base.text.decode("utf8"))

            # Extract method calls and dependencies
            calls = self._extract_function_calls(body_node)
            dependencies = []
            for child in node.children:
                dependencies.extend(self._extract_dependencies(child))

            return CodeElement(
                name=name_node.text.decode("utf8"),
                type="class",
                docstring=self._extract_docstring(body_node),
                code=code,
                file_path=file_path,
                base_classes=base_classes,
                calls=calls,
                dependencies=dependencies,
                methods=self._extract_methods(body_node),
            )
        except Exception as e:
            logger.exception(f"Error processing class node: {e}")
            return None

    def _extract_docstring(self, body_node) -> str:
        """Extract docstring from a function or class body."""
        try:
            first_node = body_node.children[0]
            if first_node.type == "expression_statement":
                string_node = first_node.children[0]
                if string_node.type in ("string", "string_literal"):
                    return string_node.text.decode("utf8").strip("'\"")
        except (IndexError, AttributeError):
            pass
        return ""

    def _extract_parameters(self, function_node) -> List[Dict[str, str]]:
        """Extract function parameters with type hints."""
        params = []
        try:
            params_node = function_node.child_by_field_name("parameters")
            if params_node:
                for param in params_node.children:
                    if param.type == "identifier":
                        params.append(
                            {"name": param.text.decode("utf8"), "type": "Any"}
                        )
        except Exception:
            pass
        return params

    def _extract_methods(self, class_body_node) -> List[str]:
        """Extract method names from class body."""
        methods = []
        try:
            for node in class_body_node.children:
                if node.type == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        methods.append(name_node.text.decode("utf8"))
        except Exception:
            pass
        return methods

    def parse(self, source_code: str, file_path: str) -> List[CodeElement]:
        """Parse source code with relationship extraction."""
        self.source_code = source_code
        tree = self.parser.parse(bytes(source_code, "utf8"))

        # First pass: collect all imports at file level
        file_level_deps = []
        for child in tree.root_node.children:
            deps = self._extract_dependencies(child)
            file_level_deps.extend(deps)

        code_elements = []
        for child in tree.root_node.children:
            if child.type == "function_definition":
                code_info = self._process_function_node(child, file_path)
                if code_info:
                    code_info.dependencies.extend(file_level_deps)
                    code_elements.append(code_info)
            elif child.type == "class_definition":
                code_info = self._process_class_node(child, file_path)
                if code_info:
                    code_info.dependencies.extend(file_level_deps)
                    code_elements.append(code_info)

        return code_elements


def analyze_and_store_python_files(
    root_dir: str,
    output_dir: str = "./ast_cache",
    parser_type: str = "ast",
    generate_explanations: bool = False,
    llm_provider: str = "anthropic",
) -> str:
    """Analyze Python files and extract code elements.

    Args:
        root_dir: Root directory containing Python files to analyze
        output_dir: Directory to store output JSON files
        parser_type: Parser to use - "ast" (built-in) or "tree-sitter" (more accurate)
        generate_explanations: If True, generate AI explanations using configured LLM
        llm_provider: LLM provider to use for explanations (default: "anthropic")

    Returns:
        Path to output directory

    Raises:
        ValueError: If parser_type is invalid
        ImportError: If tree-sitter is requested but not installed
    """
    # Create appropriate parser
    if parser_type == "ast":
        parser = ASTParser()
    elif parser_type == "tree-sitter":
        if not HAS_TREE_SITTER:
            raise ImportError(
                "tree-sitter not installed. Use --parser-type ast or install tree-sitter"
            )
        parser = TreeSitterParser()
    else:
        raise ValueError(
            f"Unknown parser type: {parser_type}. Use 'ast' or 'tree-sitter'"
        )

    # Setup directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / f"results_{parser_type}"
    results_dir.mkdir(exist_ok=True)
    progress_file = output_dir / "progress.json"

    # Load or initialize progress
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            # Ensure all required keys exist
            if "processed_elements" not in progress:
                progress["processed_elements"] = []
            if "processed_files" not in progress:
                progress["processed_files"] = []
            if "failed_elements" not in progress:
                progress["failed_elements"] = []
            processed_elements = set(progress.get("processed_elements", []))
    else:
        progress = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "processed_elements": [],
            "processed_files": [],
            "failed_elements": [],
        }
        processed_elements = set()
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # Process each Python file
    for file_path in Path(root_dir).rglob("*.py"):
        file_path_str = str(file_path)

        if file_path_str in progress["processed_files"]:
            continue

        logger.info(f"Processing file: {file_path_str} .....")

        code_elements = parser.parse(file_path.read_text(), str(file_path))

        for element in code_elements:
            element_id = f"{file_path_str}:{element.type}:{element.name}"

            if element_id in processed_elements:
                continue

            try:
                # Only generate explanation if flag is True
                if generate_explanations:
                    element.generate_explanation(llm_provider)

                # Save CodeElement
                element_file = results_dir / f"{element_id.replace('/', '_')}.json"
                with open(element_file, "w", encoding="utf-8") as f:
                    json.dump(element.to_dict(), f, indent=2)

                # Update progress
                processed_elements.add(element_id)
                if element_id not in progress["processed_elements"]:
                    progress["processed_elements"].append(element_id)
                    with open(progress_file, "w") as f:
                        json.dump(progress, f, indent=2)

            except Exception:
                progress["failed_elements"].append(element_id)
                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)

        # Mark file as complete
        if file_path_str not in progress["processed_files"]:
            progress["processed_files"].append(file_path_str)
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)

    return str(output_dir)


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze Python codebase and extract code elements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with built-in AST parser
  python ast_builder.py --root_dir rag/ --output-dir examples/ast_cache
  
  # With tree-sitter for better accuracy
  python ast_builder.py --root_dir rag/ --parser-type tree-sitter
  
  # Generate AI explanations (requires Anthropic API key)
  python ast_builder.py --root_dir rag/ --generate-explanations
        """,
    )

    parser.add_argument(
        "--root_dir", required=True, help="Root directory of Python project to analyze"
    )
    parser.add_argument(
        "--output-dir",
        default="./ast_cache",
        help="Directory to store AST analysis (default: ./ast_cache)",
    )
    parser.add_argument(
        "--parser-type",
        default="ast",
        choices=["ast", "tree-sitter"],
        help="Parser type: ast (fast, built-in) or tree-sitter (accurate, requires install)",
    )
    parser.add_argument(
        "--generate-explanations",
        action="store_true",
        help="Generate AI explanations using configured LLM provider",
    )
    parser.add_argument(
        "--llm-provider",
        default="anthropic",
        help="LLM provider for explanations (default: anthropic)",
    )

    args = parser.parse_args()

    try:
        output_path = analyze_and_store_python_files(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            parser_type=args.parser_type,
            generate_explanations=args.generate_explanations,
            llm_provider=args.llm_provider,
        )
        logger.info("Analysis complete!")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.exception(f"\n Error: {e}")
        exit(1)
