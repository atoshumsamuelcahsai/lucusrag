from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
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

            from rag.parser import format_parameters

            prompt = await PROMPTS.render(
                "code_explain_factual",
                type=self.type,
                name=self.name,
                file_path=self.file_path,
                code=self.code,
                parameters=format_parameters(self.parameters),
                return_type=self.return_type,
                docstring=self.docstring,
            )

            # Use the LLM provider's complete method
            response = llm.complete(prompt)
            self.explanation = str(response)

        except Exception as e:
            self.explanation = f"Error generating explanation: {str(e)}"
