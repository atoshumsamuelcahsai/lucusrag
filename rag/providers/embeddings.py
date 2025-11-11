from __future__ import annotations
from enum import Enum
import os
import typing as t
from dotenv import load_dotenv
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()


class EmbeddingProvider(Enum):
    VOYAGE = "voyage"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"


@t.runtime_checkable
class EmbeddingLike(t.Protocol):
    # Define only what the rest of code actually calls
    # e.g., LlamaIndex embeddings commonly expose .get_text_embedding
    def get_text_embedding(self, text: str) -> t.List[float]: ...


# --- registry mapping provider name → constructor function ---
EmbeddingFactory = t.Callable[[], EmbeddingLike]
_EMBEDDING_REGISTRY: dict[str, EmbeddingFactory] = {}


def register_embedding(
    provider: str,
) -> t.Callable[[EmbeddingFactory], EmbeddingFactory]:
    """Decorator to register a new embedding provider factory."""

    def _decorator(factory: EmbeddingFactory) -> EmbeddingFactory:
        _EMBEDDING_REGISTRY[provider] = factory
        return factory

    return _decorator


# --- register providers declaratively ---
@register_embedding("voyage")
def _make_voyage() -> EmbeddingLike:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return VoyageEmbedding(voyage_api_key=api_key, model_name="voyage-3")


@register_embedding("voyage-large")
def _make_voyage_large() -> EmbeddingLike:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return VoyageEmbedding(voyage_api_key=api_key, model_name="voyage-large-2")


@register_embedding("voyage-lite")
def _make_voyage_lite() -> EmbeddingLike:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    return VoyageEmbedding(
        voyage_api_key=api_key,
        model_name="voyage-lite-02",  # 1024 dimensions
        embed_batch_size=100,
    )


@register_embedding("openai")
def _make_openai() -> EmbeddingLike:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    # Support dimension reduction - OpenAI models can output smaller dimensions
    dimensions_str = os.getenv("VECTOR_DIMENSION")
    if dimensions_str:
        dimensions_int = int(dimensions_str)
        return OpenAIEmbedding(
            api_key=api_key,
            model="text-embedding-3-small",
            embed_batch_size=100,
            dimensions=dimensions_int,  # Custom dimension (e.g., 756)
        )
    else:
        # Default: 1536 dimensions
        return OpenAIEmbedding(
            api_key=api_key,
            model="text-embedding-3-small",
            embed_batch_size=100,
        )


def get_embeddings(provider: str = "voyage") -> EmbeddingLike:
    """Get embeddings model for the given provider name."""
    factory = _EMBEDDING_REGISTRY.get(provider)
    if not factory:
        valid = ", ".join(sorted(_EMBEDDING_REGISTRY.keys()))
        raise ValueError(
            f"Unknown embedding provider '{provider}'. Valid options: {valid}"
        )

    return factory()
