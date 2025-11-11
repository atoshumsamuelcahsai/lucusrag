from __future__ import annotations
import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
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
    max_output_tokens: int = 768,
    temperature: float = 0.0,
) -> Anthropic:
    """
    Return a configured Anthropic LLM instance.

    Reads configuration from environment variables:
    - ANTHROPIC_API_KEY (required)
    - LLM_MODEL (default: "claude-3-5-sonnet")
    - LLM_MAX_OUTPUT_TOKENS (optional)
    - LLM_TEMPERATURE (optional)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    model = os.getenv("LLM_MODEL", "claude-3-5-sonnet")
    max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", str(max_output_tokens)))
    temp = float(os.getenv("LLM_TEMPERATURE", str(temperature)))

    return Anthropic(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temp,
    )


def get_openai_llm(
    max_output_tokens: int = 756,
    temperature: float = 0.0,
) -> OpenAI:
    """
    Return a configured OpenAI LLM instance.

    Reads configuration from environment variables:
    - OPENAI_API_KEY (required)
    - LLM_MODEL (default: "gpt-4-turbo" or "gpt-4o-mini" recommended)
    - LLM_MAX_OUTPUT_TOKENS (optional)
    - LLM_TEMPERATURE (optional)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", str(max_output_tokens)))
    temp = float(os.getenv("LLM_TEMPERATURE", str(temperature)))

    return OpenAI(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temp,
    )


# Union type for LLM factories that can return different LLM types
LLMFACTORY = t.Callable[..., t.Union[Anthropic, OpenAI]]

GET_LLM_PROVIDER: t.Mapping[LLMProvider, LLMFACTORY] = {
    LLMProvider.ANTHROPIC: get_anthropic_llm,
    LLMProvider.OPENAI: get_openai_llm,
}


def get_llm(
    provider: t.Union[str, LLMProvider], **llm_kwargs: t.Any
) -> t.Union[Anthropic, OpenAI]:
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
