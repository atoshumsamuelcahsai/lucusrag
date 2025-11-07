from __future__ import annotations
import ast
import typing as t
import logging

from rag.schemas.code_element import CodeElement

from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

logger = logging.getLogger(__name__)


@t.runtime_checkable
class CodeParser(t.Protocol):
    def parse(self, source_code: str, file_path: str) -> t.List[CodeElement]: ...


class AstCodeAnalyzer(ast.NodeVisitor):
    def __init__(self, project_module: str = "rag") -> None:
        self.code_elements: t.List[CodeElement] = []
        self.current_file = ""
        self.imports: t.Dict[str, str] = {}  # Track imports for dependency analysis
        self.current_class: t.Optional[str] = None
        self.current_function: t.Optional[CodeElement] = None
        self.source_code = ""  # Add this to store the source code
        self.project_module = project_module  # Filter to only project imports

    def _file_path_to_module(self, file_path: str) -> str:
        """Convert file path to Python module name."""
        module = file_path.replace("/", ".").replace("\\", ".")
        if module.endswith(".py"):
            module = module[:-3]
        return module

    def _get_qualified_name(
        self, name: str, file_path: str, class_name: t.Optional[str] = None
    ) -> str:
        """Generate fully-qualified name for a code element."""
        module = self._file_path_to_module(file_path)

        if class_name:
            return f"{module}.{class_name}.{name}"
        else:
            return f"{module}.{name}"

    def visit_Import(self, node) -> None:  # type: ignore
        """Track regular imports."""
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):  # type: ignore
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

    def _extract_parameters(self, node: ast.FunctionDef) -> t.List[t.Dict[str, str]]:
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

    def _extract_decorators(self, node: ast.FunctionDef) -> t.List[str]:
        """Extract decorator names."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
        return decorators

    def visit_FunctionDef(self, node):  # type: ignore
        """Process function definitions."""
        is_method = self.current_class is not None

        # Filter dependencies to only include project imports
        project_deps = [
            dep for dep in self.imports.values() if dep.startswith(self.project_module)
        ]

        # Generate qualified name
        qualified_name = self._get_qualified_name(
            node.name,
            self.current_file,
            class_name=self.current_class if is_method else None,
        )

        code_info = CodeElement(
            name=qualified_name,
            type="method" if is_method else "function",
            docstring=ast.get_docstring(node) or "",
            code=self._get_source_segment(node),
            file_path=self.current_file,
            parameters=self._extract_parameters(node),
            return_type=(
                self._get_type_annotation(node.returns) if node.returns else None
            ),
            decorators=self._extract_decorators(node),
            dependencies=project_deps,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

        # Store the current function for analyzing its contents
        self.current_function = code_info
        self.generic_visit(node)
        self.current_function = None

        if is_method:
            # Add to current class's methods (use qualified name for linking)
            for element in self.code_elements:
                # Check if element name ends with current class (since it's now qualified)
                if element.type == "class" and element.name.endswith(
                    f".{self.current_class}"
                ):
                    element.methods.append(qualified_name)
        else:
            self.code_elements.append(code_info)

    def visit_AsyncFunctionDef(self, node):  # type: ignore
        """Handle async functions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):  # type: ignore
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
                current: ast.expr = base
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    base_classes.append(".".join(reversed(parts)))
                else:
                    # Handle other cases gracefully
                    base_classes.append(str(base))

        # Filter dependencies to only include project imports
        project_deps = [
            dep for dep in self.imports.values() if dep.startswith(self.project_module)
        ]

        # Generate qualified name
        qualified_name = self._get_qualified_name(node.name, self.current_file)

        code_info = CodeElement(
            name=qualified_name,
            type="class",
            docstring=ast.get_docstring(node) or "",
            code=self._get_source_segment(node),
            file_path=self.current_file,
            decorators=self._extract_decorators(node),
            base_classes=base_classes,
            dependencies=project_deps,
        )

        self.code_elements.append(code_info)
        self.generic_visit(node)
        self.current_class = previous_class

    def _get_call_name(self, node: ast.Call) -> t.Optional[str]:
        """Extract the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle nested attributes (e.g., a.b.c())
            parts = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None

    def visit_Call(self, node):  # type: ignore
        """Track function calls."""
        if self.current_function:
            call_name = self._get_call_name(node)
            if call_name:
                self.current_function.calls.append(call_name)
        self.generic_visit(node)

    def visit_Assign(self, node):  # type: ignore
        """Track assignments."""
        if self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.current_function.assignments.append(target.id)
        self.generic_visit(node)

    def _get_source_segment(self, node):  # type: ignore
        """Get the source code for a node."""
        try:
            return ast.get_source_segment(self.source_code, node) or ""
        except Exception:
            return ""


class ASTParser(CodeParser):
    """Python's built-in AST parser."""

    def __init__(self, project_module: str = "rag") -> None:
        self.project_module = project_module

    def parse(self, source_code: str, file_path: str) -> t.List[CodeElement]:
        analyzer = AstCodeAnalyzer(project_module=self.project_module)
        tree = ast.parse(source_code)
        analyzer.current_file = file_path
        analyzer.source_code = source_code
        analyzer.visit(tree)
        return analyzer.code_elements


class TreeSitterParser(CodeParser):
    """
    Tree-sitter based parser with relationship extraction.
    """

    _TRIVIAL_BUILTIN_BLOCKLIST = {
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "len",
        "print",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "map",
        "filter",
        "sorted",
        "reversed",
        "abs",
        "round",
        "hex",
        "bin",
        "ord",
        "chr",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "isinstance",
        "issubclass",
        "logger.info",
        "logger.debug",
        "logger.exception",
        "logger.warning",
    }

    def __init__(self, project_module: str = "rag") -> None:
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser()
        self.parser.language = self.PY_LANGUAGE
        self.source_code = ""
        self.project_module = project_module
        self.imports: t.Dict[str, str] = {}  # Map alias/name → full qualified path
        self.current_file = ""  # Track current file for same-file resolution
        self.local_definitions: t.Dict[str, str] = (
            {}
        )  # Map local name → full qualified path

    def _file_path_to_module(self, file_path: str) -> str:
        """Convert file path to Python module name.

        Examples:
            rag/indexer/vector_indexer.py -> rag.indexer.vector_indexer
            rag/ast/builders.py -> rag.ast.builders
        """
        # Remove .py extension and convert slashes to dots
        module = file_path.replace("/", ".").replace("\\", ".")
        if module.endswith(".py"):
            module = module[:-3]
        return module

    def _get_qualified_name(
        self, name: str, file_path: str, class_name: t.Optional[str] = None
    ) -> str:
        """Generate fully-qualified name for a code element.

        Args:
            name: Function or class name
            file_path: Path to the file
            class_name: Optional class name if this is a method

        Returns:
            Fully-qualified name (e.g., rag.indexer.vector_indexer.get_vector_store_index)
        """
        module = self._file_path_to_module(file_path)

        if class_name:
            # Method: rag.module.ClassName.method_name
            return f"{module}.{class_name}.{name}"
        else:
            # Function or class: rag.module.name
            return f"{module}.{name}"

    def _build_import_map(self, tree) -> None:  # type: ignore
        """Build a map of imported names to their fully-qualified paths.

        Examples:
            import ast                          → {'ast': 'ast'}
            from rag.db import GraphDBManager   → {'GraphDBManager': 'rag.db.GraphDBManager'}
            from rag.db import GraphDBManager as GDB → {'GDB': 'rag.db.GraphDBManager'}
        """
        self.imports = {}
        for child in tree.root_node.children:
            if child.type == "import_statement":
                # import ast, json
                for grandchild in child.children:
                    if grandchild.type == "dotted_name":
                        text = grandchild.text
                        if text:
                            name = text.decode("utf8")
                            self.imports[name] = name
            elif child.type == "import_from_statement":
                # from rag.db import GraphDBManager, get_config as gc
                dotted_names = []
                module = ""

                for grandchild in child.children:
                    if grandchild.type == "dotted_name":
                        name = grandchild.text.decode("utf8") if grandchild.text else ""
                        dotted_names.append(name)
                    elif grandchild.type == "aliased_import":
                        # Handle "GraphDBManager as GDB"
                        alias_name = None
                        real_name = None
                        for alias_child in grandchild.children:
                            if (
                                alias_child.type == "identifier"
                                or alias_child.type == "dotted_name"
                            ):
                                if alias_child.text:
                                    if real_name is None:
                                        real_name = alias_child.text.decode("utf8")
                                    else:
                                        alias_name = alias_child.text.decode("utf8")

                        if module and real_name:
                            full_path = f"{module}.{real_name}"
                            self.imports[alias_name if alias_name else real_name] = (
                                full_path
                            )

                # First dotted_name is usually the module
                if dotted_names:
                    module = dotted_names[0]
                    # Remaining dotted_names are imported items
                    for imported_name in dotted_names[1:]:
                        self.imports[imported_name] = f"{module}.{imported_name}"

    def _build_local_definitions_map(self, tree) -> None:  # type: ignore
        """Build a map of locally-defined functions and classes to their fully-qualified paths.

        Examples:
            def get_query_engine(...) → {'get_query_engine': 'rag.engine.get_query_engine'}
            class GraphDBManager(...) → {'GraphDBManager': 'rag.db.GraphDBManager'}
        """
        self.local_definitions = {}
        module = self._file_path_to_module(self.current_file)

        for child in tree.root_node.children:
            if child.type in ("function_definition", "async_function_definition"):
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text:
                    simple_name = name_node.text.decode("utf8")
                    qualified_name = f"{module}.{simple_name}"
                    self.local_definitions[simple_name] = qualified_name
            elif child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text:
                    simple_name = name_node.text.decode("utf8")
                    qualified_name = f"{module}.{simple_name}"
                    self.local_definitions[simple_name] = qualified_name
            elif child.type == "decorated_definition":
                # Handle decorated definitions (@dataclass, @decorator, etc.)
                for subchild in child.children:
                    if subchild.type in (
                        "function_definition",
                        "async_function_definition",
                        "class_definition",
                    ):
                        name_node = subchild.child_by_field_name("name")
                        if name_node and name_node.text:
                            simple_name = name_node.text.decode("utf8")
                            qualified_name = f"{module}.{simple_name}"
                            self.local_definitions[simple_name] = qualified_name

    def _resolve_call_name(self, call_name: str) -> str:
        """Resolve a call name to its fully-qualified path using import and local definition maps.

        Args:
            call_name: Simple or dotted call name (e.g., 'get_config', 'db.GraphDBManager')

        Returns:
            Fully-qualified path if resolvable, otherwise original name
        """
        # First check if it's a local definition (same file)
        if call_name in self.local_definitions:
            return self.local_definitions[call_name]

        # Then check if it's in our import map
        if call_name in self.imports:
            return self.imports[call_name]

        # Check if it's a dotted name like 'db.GraphDBManager' or 'ClassName.method_name'
        if "." in call_name:
            parts = call_name.split(".")
            first_part = parts[0]

            # Helper to check if a name looks like a class (PascalCase)
            def is_likely_class_name(name: str) -> bool:
                """Check if name follows PascalCase convention (likely a class)."""
                return bool(name and name[0].isupper() and not name.isupper())

            # Check if first part is a local class
            if first_part in self.local_definitions:
                local_full = self.local_definitions[first_part]
                # Only shorten to class if it looks like a class method call
                if len(parts) == 2 and is_likely_class_name(first_part):
                    # ClassName.method_name → resolve to just ClassName
                    # because class methods are not extracted
                    # we consider a class as node
                    return local_full
                else:
                    # module.function or complex access
                    return f"{local_full}.{'.'.join(parts[1:])}"

            # Check if first part is an import
            if first_part in self.imports:
                full_first = self.imports[first_part]
                # Only shorten if it looks like a class method call
                if len(parts) == 2 and is_likely_class_name(first_part):
                    # ClassName.method_name → resolve to just ClassName
                    return full_first
                else:
                    # module.function or complex access
                    return f"{full_first}.{'.'.join(parts[1:])}"

        # If unresolvable, return as-is (might be external library or dynamic)
        return call_name

    def _extract_function_calls(self, node: Node) -> t.List[str]:
        """Extract function calls with their full qualified names when possible."""
        calls = []

        # If this node itself is a call, extract it
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                call_name = None

                if func_node.type == "identifier":
                    # Simple function call
                    if func_node.text:
                        call_name = func_node.text.decode("utf8")
                elif func_node.type == "attribute":
                    # Method or qualified function call
                    # (e.g., db.get_config or self.method)
                    parts: t.List[str] = []
                    current: t.Optional[Node] = func_node
                    while current:
                        if current.type == "attribute":
                            attr = current.child_by_field_name("attribute")
                            if attr and attr.text:
                                parts.insert(0, attr.text.decode("utf8"))
                            current = current.child_by_field_name("object")
                        elif current.type == "identifier":
                            if current.text:
                                parts.insert(0, current.text.decode("utf8"))
                            current = None
                        else:
                            break
                    if parts:
                        call_name = ".".join(parts)

                # Filter and resolve the call name
                if call_name and call_name not in self._TRIVIAL_BUILTIN_BLOCKLIST:
                    # Skip self.* calls as they're method calls within the class
                    if not call_name.startswith("self."):
                        # Check if it's an instance method call to skip
                        skip_instance_method = False
                        if "." in call_name:
                            first_part = call_name.split(".")[0]
                            # If first part starts with lowercase, it's likely an instance variable
                            if first_part and first_part[0].islower():
                                # But allow module.function calls (modules are imported)
                                if (
                                    first_part not in self.imports
                                    and first_part not in self.local_definitions
                                ):
                                    skip_instance_method = True

                        if not skip_instance_method:
                            resolved_name = self._resolve_call_name(call_name)
                            calls.append(resolved_name)

        # Always recurse into all children to find nested calls
        for child in node.children:
            calls.extend(self._extract_function_calls(child))

        return calls

    def _extract_dependencies(self, node) -> t.List[str]:  # type: ignore
        """Extract dependencies from imports."""
        deps = []
        if node.type == "import_statement":
            # import ast, json
            for child in node.children:
                if child.type == "dotted_name":
                    text = child.text
                    if text:
                        deps.append(text.decode("utf8"))
        elif node.type == "import_from_statement":
            # from rag.schemas import CodeElement
            dotted_names = [
                child.text.decode("utf8")
                for child in node.children
                if child.type == "dotted_name" and child.text
            ]
            if len(dotted_names) >= 2:
                # dotted_names[0] is module, rest are imported names
                module = dotted_names[0]
                for name in dotted_names[1:]:
                    deps.append(f"{module}.{name}")
            elif len(dotted_names) == 1:
                # Just the module name
                deps.append(dotted_names[0])
        return deps

    def _process_function_node(  # type: ignore
        self, node, file_path: str
    ) -> t.Optional[CodeElement]:
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

            # Get simple and qualified names
            simple_name = name_node.text.decode("utf8")
            qualified_name = self._get_qualified_name(simple_name, file_path)

            return CodeElement(
                name=qualified_name,
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

    def _process_class_node(  # type: ignore
        self, node, file_path: str
    ) -> t.Optional[CodeElement]:
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
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node:
                # superclasses is an argument_list like (CodeParser, OtherClass)
                for base in superclasses_node.children:
                    if base.type == "identifier":
                        if base.text:
                            base_classes.append(base.text.decode("utf8"))
                    elif base.type == "attribute":
                        # Handle dotted base classes (e.g., ast.NodeVisitor)
                        parts: t.List[str] = []
                        current: t.Optional[Node] = base
                        while current:
                            if current.type == "attribute":
                                attr = current.child_by_field_name("attribute")
                                if attr and attr.text:
                                    parts.insert(0, attr.text.decode("utf8"))
                                current = current.child_by_field_name("object")
                            elif current.type == "identifier":
                                if current.text:
                                    parts.insert(0, current.text.decode("utf8"))
                                current = None
                            else:
                                break
                        if parts:
                            base_classes.append(".".join(parts))

            # Extract method calls and dependencies
            calls = self._extract_function_calls(body_node)
            dependencies = []
            for child in node.children:
                dependencies.extend(self._extract_dependencies(child))

            # Get simple and qualified names
            simple_name = name_node.text.decode("utf8")
            qualified_name = self._get_qualified_name(simple_name, file_path)

            return CodeElement(
                name=qualified_name,
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

    def _extract_docstring(self, body_node: Node) -> str:
        """Extract docstring from a function or class body."""
        try:
            first_node = body_node.children[0]
            if first_node.type == "expression_statement":
                string_node = first_node.children[0]
                if (
                    string_node.type in ("string", "string_literal")
                    and string_node.text
                ):
                    return string_node.text.decode("utf8").strip("'\"")
        except (IndexError, AttributeError):
            pass
        return ""

    def _extract_parameters(self, function_node) -> t.List[t.Dict[str, str]]:  # type: ignore
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

    def _extract_methods(self, class_body_node) -> t.List[str]:  # type: ignore
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

    def _file_level_project_only_imports(self, tree) -> t.List[str]:  # type: ignore
        file_level_deps = []
        for child in tree.root_node.children:
            deps = self._extract_dependencies(child)
            for dep in deps:
                if dep.startswith(self.project_module):
                    file_level_deps.extend(deps)
        return file_level_deps

    def _parse_definitions(self, tree, project_deps: t.List[str], file_path: str) -> t.List[CodeElement]:  # type: ignore
        code_elements = []
        for child in tree.root_node.children:
            if child.type in ("function_definition", "async_function_definition"):
                code_info = self._process_function_node(child, file_path)
                if code_info:
                    code_info.dependencies.extend(project_deps)
                    code_elements.append(code_info)
            elif child.type == "class_definition":
                code_info = self._process_class_node(child, file_path)
                if code_info:
                    code_info.dependencies.extend(project_deps)
                    code_elements.append(code_info)
            elif child.type == "decorated_definition":
                # Handle decorated functions/classes (e.g., @dataclass, @staticmethod)
                # The actual definition is nested inside
                for subchild in child.children:
                    if subchild.type in (
                        "function_definition",
                        "async_function_definition",
                    ):
                        code_info = self._process_function_node(subchild, file_path)
                        if code_info:
                            code_info.dependencies.extend(project_deps)
                            code_elements.append(code_info)
                    elif subchild.type == "class_definition":
                        code_info = self._process_class_node(subchild, file_path)
                        if code_info:
                            code_info.dependencies.extend(project_deps)
                            code_elements.append(code_info)
        return code_elements

    def parse(self, source_code: str, file_path: str) -> t.List[CodeElement]:
        """Parse source code with relationship extraction."""
        self.source_code = source_code
        self.current_file = file_path
        tree = self.parser.parse(bytes(source_code, "utf8"))

        # First pass: build import map for resolving call names
        self._build_import_map(tree)

        # Second pass: build local definitions map (functions/classes in this file)
        self._build_local_definitions_map(tree)

        # Third pass: collect all imports at file level
        project_deps = self._file_level_project_only_imports(tree=tree)

        # Fourth pass: collect class and function information
        code_elements = self._parse_definitions(tree, project_deps, file_path)
        return code_elements
