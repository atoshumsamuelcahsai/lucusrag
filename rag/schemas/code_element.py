from __future__ import annotations
from llama_index.core.schema import TextNode
from dataclasses import dataclass
from typing import List, Optional
import textwrap


@dataclass
class CodeElement:
    """Schema for code elements."""

    type: str
    name: str
    docstring: str
    code: str
    file_path: str
    parameters: Optional[List[dict]] = None
    return_type: Optional[str] = None
    decorators: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    base_classes: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    calls: Optional[List[str]] = None
    assignments: Optional[List[str]] = None
    explanation: Optional[str] = None
    is_async: bool = False

    def format_parameters(self, params: List[dict]) -> Optional[str]:
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

    def to_node(self) -> TextNode:
        """Convert to LlamaIndex node."""
        # Format code with minimal essential information
        text_content = f"""
        {self.type.upper()}: {self.name}
        File: {self.file_path}
        
        {textwrap.dedent(self.code).strip() if self.code else 'No code available'}
        """

        # Clean up the text content
        text_content = textwrap.dedent(text_content).strip()

        metadata = {
            "type": self.type,
            "name": self.name,
            "file_path": self.file_path,
            "parameters": (
                self.format_parameters(self.parameters) if self.parameters else None
            ),
            "return_type": self.return_type,
            "decorators": ",".join(self.decorators) if self.decorators else None,
            "dependencies": ",".join(self.dependencies) if self.dependencies else None,
            "base_classes": ",".join(self.base_classes) if self.base_classes else None,
            "methods": ",".join(self.methods) if self.methods else None,
            "assignments": ",".join(self.assignments) if self.assignments else None,
            "calls": ",".join(str(call) for call in self.calls) if self.calls else None,
        }
        metadata = {k: v for k, v in metadata.items() if v not in (None, "", [])}

        return TextNode(text=text_content, metadata=metadata)

    def to_dict(self) -> dict:
        """Convert CodeElement to a dictionary representation."""
        return {
            "type": self.type,
            "name": self.name,
            "docstring": self.docstring,
            "code": self.code,
            "file_path": self.file_path,
            "parameters": self.parameters if self.parameters else None,
            "return_type": self.return_type if self.return_type else None,
            "decorators": list(self.decorators) if self.decorators else None,
            "dependencies": list(self.dependencies) if self.dependencies else None,
            "base_classes": list(self.base_classes) if self.base_classes else None,
            "methods": list(self.methods) if self.methods else None,
            "calls": list(self.calls) if self.calls else None,
            "assignments": list(self.assignments) if self.assignments else None,
            "explanation": self.explanation if self.explanation else None,
            "formatted_parameters": (
                self.format_parameters(self.parameters) if self.parameters else None
            ),
        }
