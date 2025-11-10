from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import typing as t
from rag.prompts.factory import PromptFactory
import uuid
import random
import asyncio


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

        from rag.parser.parser import format_parameters
        from rag.providers.llms import get_llm

        llm = get_llm(llm_provider)
        prompts = await PromptFactory.get_instance()
        prompt = await prompts.render(
            "code_explain_factual",
            type=self.type,
            name=self.name,
            file_path=self.file_path,
            code=self.code,
            parameters=format_parameters(self.parameters),
            return_type=self.return_type,
            docstring=self.docstring,
        )
        # Use the LLM provider's async complete method
        response = await self._with_retries(lambda: llm.acomplete(prompt))
        self.explanation = response.text

    async def _with_retries(
        self,
        coro_factory: t.Callable[[], t.Coroutine[t.Any, t.Any, t.Any]],
        retries: int = 5,
        timeout_s: float = 30,
    ) -> t.Any:
        """Retry a coroutine factory with exponential backoff; propagate cancellations."""
        backoff = 0.5
        err: t.Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                return await asyncio.wait_for(coro_factory(), timeout=timeout_s)
            except asyncio.CancelledError:
                # Allow gather()/TaskGroup to cancel cleanly
                raise
            except (asyncio.TimeoutError, TimeoutError) as e:
                err = e
            except Exception as e:
                err = e

            if attempt == retries:
                if err is None:
                    raise RuntimeError("Retry exhausted but no exception was captured")
                raise err
            await asyncio.sleep(backoff + random.random() * 0.25)
            backoff *= 2
