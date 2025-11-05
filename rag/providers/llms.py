from __future__ import annotations
import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from enum import Enum
import typing as t

load_dotenv()


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"


def get_anthropic_llm(
    number_of_tokens: int = 4096, temperature: float = 0
) -> Anthropic:
    """Return a configured Anthropic LLM"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return Anthropic(
        model="claude-3-5-sonnet-20240620",
        api_key=api_key,
        max_tokens=number_of_tokens,
        temperature=temperature,
    )


LLMFACTORY = t.Callable[..., Anthropic]

GET_LLM_PROVIDER: t.Mapping[LLMProvider, LLMFACTORY] = {
    LLMProvider.ANTHROPIC: get_anthropic_llm,
}


def get_llm(provider: t.Union[str, LLMProvider], **llm_kwargs: t.Any) -> Anthropic:
    """
    Resolve an LLM instance by provider name or enum.

    Args:
        provider: Provider name (e.g., "anthropic") or LLMProvider enum.
        **llm_kwargs: Model-specific keyword arguments.

    Example:
        llm = get_llm("anthropic", max_tokens=2048, temperature=0.3)
    """
    # Convert string → Enum safely
    if isinstance(provider, str):
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError as e:
            valid = ", ".join(p.value for p in LLMProvider)
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Valid options: {valid}"
            ) from e
    else:
        provider_enum = provider

    factory = GET_LLM_PROVIDER.get(provider_enum)
    if factory is None:
        raise ValueError(f"LLM provider not implemented: {provider_enum.value}")

    return factory(**llm_kwargs)
