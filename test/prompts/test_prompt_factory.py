import asyncio
from pathlib import Path
import pytest

from rag.prompts.factory import PromptFactory

pytestmark = pytest.mark.asyncio

PROMPT_OK = """\
id: code_explain_generalized
meta:
  task: indexing
  version: "1"
template: |
  Name: {{ name }}
  Kind: {{ kind }}
  {% if file_path %}Path: {{ file_path }}{% endif %}
  {{ source }}
"""

PROMPT_BAD_MISSING_FIELDS = """\
meta: {task: indexing}
template: "Hello"
"""

PROMPT_DUP_A = """\
id: dup
template: "A"
"""
PROMPT_DUP_B = """\
id: dup
template: "B"
"""


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "ok.yaml").write_text(PROMPT_OK, encoding="utf-8")
    return d


async def test_load_once_and_render(prompt_dir: Path):
    pf = PromptFactory(prompt_dir)
    await pf.load_once()  # should succeed
    out = await pf.render(
        "code_explain_generalized",
        name="parse_ast",
        kind="function",
        file_path="src/parser.py",
        source="def parse_ast(...): pass",
    )
    assert "parse_ast" in out
    assert "function" in out
    assert "src/parser.py" in out
    assert "def parse_ast" in out


async def test_load_once_idempotent(prompt_dir: Path):
    pf = PromptFactory(prompt_dir)
    await asyncio.gather(pf.load_once(), pf.load_once(), pf.load_once())
    # should not raise; ensure we can render
    assert "parse_ast" in await pf.render(
        "code_explain_generalized", name="parse_ast", kind="function", source="s"
    )


async def test_missing_id_or_template_raises(tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "bad.yaml").write_text(PROMPT_BAD_MISSING_FIELDS, encoding="utf-8")
    pf = PromptFactory(d)
    with pytest.raises(ValueError):
        await pf.load_once()


async def test_duplicate_ids_rejected(tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "a.yaml").write_text(PROMPT_DUP_A, encoding="utf-8")
    (d / "b.yaml").write_text(PROMPT_DUP_B, encoding="utf-8")
    pf = PromptFactory(d)
    with pytest.raises(AssertionError):
        await pf.load_once()


async def test_concurrent_renders(prompt_dir: Path):
    pf = PromptFactory(prompt_dir)
    await pf.load_once()
    tasks = [
        pf.render("code_explain_generalized", name=f"n{i}", kind="k", source="s")
        for i in range(100)
    ]
    outs = await asyncio.gather(*tasks)
    assert len(outs) == 100
    assert all("k" in o for o in outs)


def test_get_requires_loaded(prompt_dir: Path):
    pf = PromptFactory(prompt_dir)
    with pytest.raises(AssertionError):
        pf.get("code_explain_generalized")
