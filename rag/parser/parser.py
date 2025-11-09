from __future__ import annotations
import typing as t
from llama_index.core.schema import TextNode
from llama_index.core import Document

from rag.schemas import CodeElement
import textwrap
import logging

logger = logging.getLogger(__name__)


def format_parameters(params: t.List[dict]) -> t.Optional[str]:
    """Format parameters into a searchable string."""
    if not params:
        return None

    param_strs = []
    for param in params:
        name = param.get("name", "")
        param_type = param.get("type", "")
        default = param.get("default", "")

        param_str = f"{name}"
        if param_type:
            param_str += f":{param_type}"
        if default:
            param_str += f"={default}"
        param_strs.append(param_str)

    return ", ".join(param_strs)


def create_text_representation(metadata: t.Dict) -> str:
    """
    Create a formatted text representation of code element metadata.

    Used for:
    - Creating embeddings (embedding_loader.py)
    - Creating TextNodes for indexing

    Args:
        metadata: Dictionary containing code element metadata

    Returns:
        Formatted text representation
    """
    parameters = metadata.get("parameters") or []
    dependencies = metadata.get("dependencies") or []
    base_classes = metadata.get("base_classes") or []
    calls = metadata.get("calls") or []
    decorators = metadata.get("decorators") or []
    methods = metadata.get("methods") or []
    assignments = metadata.get("assignments") or []
    docstring = metadata.get("docstring") or ""
    return_type = metadata.get("return_type") or ""
    explanation = metadata.get("explanation") or ""
    file_path = metadata.get("file_path") or ""
    is_async = metadata.get("is_async", False)

    parts = []

    # Basic information
    parts.append(f"Name: {metadata.get('name', '')}")
    parts.append(f"Type: {metadata.get('type', '')}")
    parts.append(f"File: {file_path}")
    if is_async:
        parts.append("Async: true")

    # Description
    if docstring:
        parts.append(f"Description: {docstring}")

    # AI-generated explanation (if available)
    if explanation:
        parts.append(f"Explanation: {explanation}")

    # Code
    code = metadata.get("code", "")
    if code:
        parts.append(f"\nCode:\n{code}")

    # Decorators
    if decorators:
        parts.append(f"Decorators: {', '.join(str(d) for d in decorators)}")

    # Parameters
    param_str = format_parameters(parameters)
    if param_str:
        parts.append(f"Parameters: {param_str}")

    # Return type
    if return_type:
        parts.append(f"Return Type: {return_type}")

    # Relationships
    if dependencies:
        parts.append(f"Dependencies: {', '.join(str(d) for d in dependencies)}")

    if base_classes:
        parts.append(f"Base Classes: {', '.join(str(b) for b in base_classes)}")

    if calls:
        parts.append(f"Function Calls: {', '.join(str(c) for c in calls)}")

    if methods:
        parts.append(f"Methods: {', '.join(str(m) for m in methods)}")

    if assignments:
        parts.append(f"Assignments: {', '.join(str(a) for a in assignments)}")

    return "\n".join(parts)


def process_code_element(code_info: CodeElement) -> Document:
    """
    Convert a CodeElement to a LlamaIndex Document.

    Used by data_loader.py to prepare documents for indexing.

    Args:
        code_info: CodeElement instance

    Returns:
        Document with code element metadata
    """
    return Document(
        page_content=code_info.code,
        metadata={
            "id": code_info.id,
            "name": code_info.name,
            "type": code_info.type,
            "file_path": code_info.file_path,
            "docstring": code_info.docstring,
            "code": code_info.code,
            "parameters": code_info.parameters or [],
            "return_type": code_info.return_type or "",
            "dependencies": code_info.dependencies or [],
            "base_classes": code_info.base_classes or [],
            "calls": code_info.calls or [],
            "methods": code_info.methods or [],
            "decorators": code_info.decorators or [],
            "assignments": code_info.assignments or [],
            "explanation": code_info.explanation or "",
            "is_async": code_info.is_async,
        },
    )


def to_node(code_info: CodeElement) -> TextNode:
    """Convert to LlamaIndex node."""
    # Format code with minimal essential information
    text_content = f"""
    {code_info.type.upper()}: {code_info.name}
    File: {code_info.file_path}
    
    {textwrap.dedent(code_info.code).strip() if code_info.code else 'No code available'}
    """

    # Clean up the text content
    text_content = textwrap.dedent(text_content).strip()

    metadata = {
        "id": code_info.id,
        "type": code_info.type,
        "name": code_info.name,
        "file_path": code_info.file_path,
        "parameters": (
            format_parameters(code_info.parameters) if code_info.parameters else None
        ),
        "return_type": code_info.return_type,
        "decorators": ",".join(code_info.decorators) if code_info.decorators else None,
        "dependencies": (
            ",".join(code_info.dependencies) if code_info.dependencies else None
        ),
        "base_classes": (
            ",".join(code_info.base_classes) if code_info.base_classes else None
        ),
        "methods": ",".join(code_info.methods) if code_info.methods else None,
        "assignments": (
            ",".join(code_info.assignments) if code_info.assignments else None
        ),
        "calls": (
            ",".join(str(call) for call in code_info.calls) if code_info.calls else None
        ),
        "explanation": code_info.explanation if code_info.explanation else None,
    }
    metadata = {k: v for k, v in metadata.items() if v not in (None, "", [])}

    return TextNode(text=text_content, metadata=metadata)


def parse_documents_to_nodes(
    documents: t.List[Document],
    show_progress: t.Optional[bool] = None,
    **kwargs: t.Any,
) -> t.List[TextNode]:
    """
    Transform Documents into TextNodes with formatted text.

    Used by LlamaIndex as Settings.node_parser during indexing.

    Args:
        documents: List of Document objects
        show_progress: Whether to show progress (unused, for LlamaIndex compatibility)
        **kwargs: Additional arguments (unused, for LlamaIndex compatibility)

    Returns:
        List of TextNode objects with formatted text
    """
    result_nodes = []
    for doc in documents:
        try:
            # Create text representation
            text = create_text_representation(doc.metadata)

            node = TextNode(
                text=text,
                metadata={
                    "id": doc.metadata.get("id"),
                    "name": doc.metadata.get("name"),
                    "type": doc.metadata.get("type"),
                    "file_path": doc.metadata.get("file_path"),
                },
            )
            result_nodes.append(node)

        except Exception as e:
            logger.exception(f"Parser error for document: {str(e)}")
            continue

    return result_nodes
