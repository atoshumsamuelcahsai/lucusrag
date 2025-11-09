import pytest

# Adjust this import if your file lives elsewhere
import rag.providers.embeddings as factory_mod


def test_register_embedding_decorator_registers_and_returns_factory(monkeypatch):
    # Start with a clean registry
    monkeypatch.setattr(factory_mod, "_EMBEDDING_REGISTRY", {}, raising=True)

    # Define a dummy factory to register
    class DummyEmb:
        def __init__(self, api_key):
            self.api_key = api_key

        def get_text_embedding(self, text: str):
            return [0.0]

    @factory_mod.register_embedding("dummy")
    def make_dummy(api_key: str):
        return DummyEmb(api_key)

    # Check registration and return value
    assert "dummy" in factory_mod._EMBEDDING_REGISTRY
    assert factory_mod._EMBEDDING_REGISTRY["dummy"] is make_dummy


def test_get_embeddings_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANYTHING_API_KEY", raising=False)
    with pytest.raises(EnvironmentError) as e:
        factory_mod.get_embeddings("anything")
    assert "ANYTHING_API_KEY" in str(e.value)


def test_get_embeddings_raises_for_unknown_provider(monkeypatch):
    # Ensure an API key exists so we test the provider branch
    # Provider name "unknownprovider" uppercased becomes "UNKNOWNPROVIDER"
    monkeypatch.setenv("UNKNOWNPROVIDER_API_KEY", "TESTKEY")
    # Empty registry to guarantee "unknown"
    monkeypatch.setattr(factory_mod, "_EMBEDDING_REGISTRY", {}, raising=True)

    with pytest.raises(ValueError) as e:
        factory_mod.get_embeddings("unknownprovider")
    # error message should mention provider name
    assert "unknownprovider" in str(e.value)


def test_get_embeddings_calls_factory_with_api_key(monkeypatch):
    monkeypatch.setenv("DUMMY_API_KEY", "SECRET123")
    monkeypatch.setattr(factory_mod, "_EMBEDDING_REGISTRY", {}, raising=True)

    # Capture the api_key passed into the factory and return a dummy embedding
    captured = {"api_key": None}

    class DummyEmb:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def get_text_embedding(self, text: str):
            return [1.0, 2.0]

    def make_dummy(api_key: str):
        return DummyEmb(api_key)

    # Register a controlled provider
    factory_mod._EMBEDDING_REGISTRY["dummy"] = make_dummy

    emb = factory_mod.get_embeddings("dummy")
    assert isinstance(emb, DummyEmb)
    assert captured["api_key"] == "SECRET123"


def test_default_provider_resolves_voyage_when_registered(monkeypatch):
    # If module registers voyage at import time, great.
    # To avoid depending on VoyageEmbedding, we override the registry.
    monkeypatch.setenv("VOYAGE_API_KEY", "KEYX")
    monkeypatch.setattr(factory_mod, "_EMBEDDING_REGISTRY", {}, raising=True)

    class DummyEmb:
        def __init__(self, api_key):
            self.api_key = api_key

        def get_text_embedding(self, text: str):
            return [3.0]

    factory_mod._EMBEDDING_REGISTRY["voyage"] = lambda k: DummyEmb(k)

    # Call with default (provider not passed)
    emb = factory_mod.get_embeddings("voyage")
    assert isinstance(emb, DummyEmb)
    assert emb.api_key == "KEYX"
