from __future__ import annotations
import aiofiles
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import typing as t

import yaml
from jinja2 import Environment

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str
    template: str
    meta: dict[str, t.Any] = field(default_factory=dict)
    # precompiled Jinja template
    compiled: t.Any = field(default=None, repr=False)


class PromptFactory:
    _instance: t.Optional["PromptFactory"] = None
    _instance_lock = asyncio.Lock()

    def __init__(self, prompt_dir: Path = PROMPT_DIR):
        self._dir = prompt_dir
        self._prompts: dict[str, Prompt] = {}
        self._loaded = False
        self._lock = asyncio.Lock()
        # nosec B701: autoescape=False is safe here because:
        # Templates are loaded from trusted YAML files (not user input)
        self.env = Environment(
            autoescape=False,  # nosec B701
            trim_blocks=True,
            lstrip_blocks=True,
            enable_async=True,
        )

    @classmethod
    async def get_instance(cls) -> "PromptFactory":
        if cls._instance is not None:
            return cls._instance
        async with cls._instance_lock:
            if cls._instance is None:
                inst = cls()
                await inst.load_once()
                cls._instance = inst
            return cls._instance

    async def _load_file(self, p: Path) -> Prompt:
        async with aiofiles.open(p, mode="r", encoding="utf-8") as f:
            text = await f.read()
        data = yaml.safe_load(text) or {}
        pid = data.get("id")
        tpl = data.get("template")
        if not pid or not tpl:
            raise ValueError(f"Prompt file {p} missing 'id' or 'template'")
        compiled = self.env.from_string(tpl)
        return Prompt(
            id=pid, template=tpl, meta=data.get("meta", {}), compiled=compiled
        )

    async def load_once(self) -> None:
        if self._loaded:
            return
        files = list(self._dir.glob("*.yaml"))
        prompts = await asyncio.gather(*(self._load_file(p) for p in files))
        ids = [pmpt.id for pmpt in prompts]
        assert len(ids) == len(set(ids)), "Duplicate prompt IDs found"
        self._prompts = {pmpt.id: pmpt for pmpt in prompts}
        self._loaded = True

    async def render(self, prompt_id: str, **kwargs: t.Any) -> str:
        if not self._loaded:
            await self.load_once()
        pmpt = self._prompts[prompt_id]
        return await pmpt.compiled.render_async(**kwargs)

    def get(self, prompt_id: str) -> Prompt:
        assert self._loaded, "PromptFactory not loaded; call load_once() first"
        return self._prompts[prompt_id]
