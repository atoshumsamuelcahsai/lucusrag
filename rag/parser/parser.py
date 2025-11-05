from __future__ import annotations
import typing as t
from llama_index.core.schema import TextNode
from llama_index.core import Document

from rag.schemas import CodeElement
import logging

logger = logging.getLogger(__name__)


def convert_to_llama_nodes(analysis_data: t.Dict[str, t.Any]) -> t.List[TextNode]:
    """Convert analysis data to LlamaIndex CodeElement nodes."""
    nodes: t.List[TextNode] = []

    for element in analysis_data["elements"]:
        node = CodeElement(
            type=element["type"],
            name=element["name"],
            docstring=element["docstring"],
            code=element["code"],
            file_path=element["file_path"],
            parameters=element.get("parameters"),
            return_type=element.get("return_type"),
            decorators=element.get("decorators"),
            dependencies=element.get("dependencies"),
            base_classes=element.get("base_classes"),
            methods=element.get("methods"),
            calls=element.get("calls"),
            assignments=element.get("assignments"),
            explanation=element.get("explanation"),
        )
        nodes.append(node.to_node())
    return nodes


class CodeElementParser:
    def parse(self, code_elements: t.List[CodeElement]) -> t.List["TextNode"]:
        """Parse code elements into LlamaIndex TextNodes."""
        return [element.to_node() for element in code_elements]


def process_code_element(code_info: CodeElement):
    """Process a single code element into a document."""
    return Document(
        page_content=code_info.code,
        metadata={
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
        },
    )


class CodeElementGraphParser:
    """Parser for code elements to create structured documents."""

    def __init__(self):
        # Reduce to essential fields that match the document creation
        self.required_fields = ["name", "type", "file_path"]

    def __call__(
        self,
        nodes: t.List[Document],
        show_progress: t.Optional[bool] = None,
        **kwargs: t.Any,
    ) -> t.List[TextNode]:
        """Transform documents into nodes."""
        result_nodes = []
        for doc in nodes:
            try:
                # Extract the basic metadata we know exists
                metadata = {
                    "name": doc.metadata.get("name", ""),
                    "type": doc.metadata.get("type", ""),
                    "file_path": doc.metadata.get("file_path", ""),
                    "docstring": doc.metadata.get("docstring", ""),
                    "code": doc.metadata.get("code", ""),
                    "parameters": doc.metadata.get("parameters", []),
                    "return_type": doc.metadata.get("return_type", ""),
                    "dependencies": doc.metadata.get("dependencies", []),
                    "base_classes": doc.metadata.get("base_classes", []),
                    "calls": doc.metadata.get("calls", []),
                }

                # Create text representation
                text = self.create_text_representation(metadata)

                # Create node
                node = TextNode(
                    text=text,
                    metadata={
                        "id": f"{metadata['file_path']}:{metadata['type']}:{metadata['name']}",
                        "name": metadata["name"],
                        "type": metadata["type"],
                        "file_path": metadata["file_path"],
                    },
                )
                result_nodes.append(node)

            except Exception as e:
                logger.exception(f"Parser error for document: {str(e)}")
                continue

        return result_nodes

    def create_text_representation(self, metadata: t.Dict) -> str:
        """Create a formatted text representation of the code element."""
        # Convert None values to empty lists/strings
        parameters = metadata.get("parameters") or []
        dependencies = metadata.get("dependencies") or []
        base_classes = metadata.get("base_classes") or []
        calls = metadata.get("calls") or []
        docstring = metadata.get("docstring") or ""
        return_type = metadata.get("return_type") or ""

        return f"""
        Name: {metadata['name']}
        Type: {metadata['type']}
        Description: {docstring}

        Code:
        {metadata['code']}

        Parameters: {self.format_parameters(parameters)}
        Return Type: {return_type}
        Dependencies: {', '.join(str(d) for d in dependencies)}
        Base Classes: {', '.join(str(b) for b in base_classes)}
        Function Calls: {', '.join(str(c) for c in calls)}
        """

    def format_parameters(self, parameters: t.List[t.Dict]) -> str:
        """Format parameters into a readable string."""
        if not parameters:
            return ""
        return ", ".join(f"{p.get('name', '')}:{p.get('type', '')}" for p in parameters)
