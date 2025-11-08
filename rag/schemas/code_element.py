from __future__ import annotations
from llama_index.core.schema import TextNode
from dataclasses import dataclass, field
from typing import List, Optional
import textwrap
from rag.prompts import PROMPTS
import uuid


@dataclass
class CodeElement:
    """Schema for code elements."""

    type: str
    name: str
    docstring: str
    code: str
    file_path: str
    parameters: List[dict] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    assignments: List[str] = field(default_factory=list)
    explanation: str = ""
    is_async: bool = False

    @property
    def id(self) -> str:
        """Generate a unique ID for the code element."""
        _id = f"{self.type}:{self.name}:{self.file_path}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, _id))

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
            "id": self.id,
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
            "explanation": self.explanation if self.explanation else None,
        }
        metadata = {k: v for k, v in metadata.items() if v not in (None, "", [])}

        return TextNode(text=text_content, metadata=metadata)

    def to_dict(self) -> dict:
        """Convert CodeElement to a dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "docstring": self.docstring,
            "code": self.code,
            "file_path": self.file_path,
            "parameters": self.parameters if self.parameters else [],
            "return_type": self.return_type if self.return_type else None,
            "decorators": list(self.decorators) if self.decorators else [],
            "dependencies": list(set(self.dependencies)) if self.dependencies else [],
            "base_classes": list(self.base_classes) if self.base_classes else [],
            "methods": list(self.methods) if self.methods else [],
            "calls": list(set(self.calls)) if self.calls else [],
            "assignments": list(set(self.assignments)) if self.assignments else [],
            "explanation": self.explanation if self.explanation else None,
            "is_async": self.is_async,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeElement":
        """Create CodeElement from dictionary."""
        return cls(**data)

    async def generate_explanation(self, llm_provider: str = "anthropic") -> None:
        """Generate AI explanation using configured LLM provider.

        Args:
            llm_provider: LLM provider to use (default: "anthropic")
        """
        try:
            # Use the provider abstraction instead of hardcoding Anthropic
            from rag.providers.llms import get_llm

            llm = get_llm(llm_provider)

            prompt = await PROMPTS.render(
                "code_explain_factual",
                type=self.type,
                name=self.name,
                file_path=self.file_path,
                code=self.code,
                parameters=self.format_parameters(self.parameters),
                return_type=self.return_type,
                docstring=self.docstring,
            )

            # Use the LLM provider's complete method
            response = llm.complete(prompt)
            self.explanation = str(response)

        except Exception as e:
            self.explanation = f"Error generating explanation: {str(e)}"
