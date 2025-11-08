from pathlib import Path
from .factory import PromptFactory

PROMPTS = PromptFactory(prompt_dir=Path(__file__).parent / "prompts")

# app startup
# await PROMPTS.load_once()

# usage anywhere (after startup)
# txt = await PROMPTS.render("code_explain_generalized", name=..., kind=..., file_path=..., source=...)
