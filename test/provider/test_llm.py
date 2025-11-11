import pytest

# Adjust import path to where your code lives
import rag.providers.llms as llm_mod


def test_get_anthropic_llm_raises_if_env_missing(monkeypatch):
    # Test: Raises RuntimeError when ANTHROPIC_API_KEY missing
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as e:
        llm_mod.get_anthropic_llm()

    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_get_anthropic_llm_returns_instance(monkeypatch):
    # Test: Returns Anthropic instance when key exists
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE_KEY")

    class DummyAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Patch Anthropic class used inside the module
    monkeypatch.setattr(llm_mod, "Anthropic", DummyAnthropic)

    llm = llm_mod.get_anthropic_llm(max_output_tokens=123, temperature=0.7)

    assert isinstance(llm, DummyAnthropic)
    assert llm.kwargs["model"] == "gpt-4o-mini"  # Default from env or function
    assert llm.kwargs["api_key"] == "FAKE_KEY"
    assert llm.kwargs["max_tokens"] == 756
    assert llm.kwargs["temperature"] == 0.7


def test_get_llm_with_string_provider(monkeypatch):
    # Test: get_llm resolves correctly for string provider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "TESTKEY")

    class DummyAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_mod, "Anthropic", DummyAnthropic)

    llm = llm_mod.get_llm("anthropic", max_output_tokens=222, temperature=0.1)
    assert isinstance(llm, DummyAnthropic)
    assert llm.kwargs["api_key"] == "TESTKEY"
    assert llm.kwargs["max_tokens"] == 756
    assert llm.kwargs["temperature"] == 0.1


def test_get_llm_with_enum_provider(monkeypatch):
    # Test: get_llm with Enum provider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ENUMKEY")

    class DummyAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_mod, "Anthropic", DummyAnthropic)

    llm = llm_mod.get_llm(llm_mod.LLMProvider.ANTHROPIC, max_output_tokens=64)
    assert isinstance(llm, DummyAnthropic)
    assert llm.kwargs["api_key"] == "ENUMKEY"


def test_get_llm_raises_for_unsupported_provider(monkeypatch):
    # Test: raises ValueError for unsupported provider string
    with pytest.raises(ValueError) as e:
        llm_mod.get_llm("invalid-provider")

    msg = str(e.value)
    assert "Unsupported LLM provider" in msg
    assert "anthropic" in msg  # message lists valid options


def test_get_llm_raises_when_factory_missing(monkeypatch):
    # Test: raises ValueError if provider known but factory missing
    monkeypatch.setenv("ANTHROPIC_API_KEY", "KEYX")
    # temporarily clear factory mapping
    monkeypatch.setattr(llm_mod, "GET_LLM_PROVIDER", {}, raising=True)

    with pytest.raises(ValueError) as e:
        llm_mod.get_llm("anthropic")

    assert "LLM provider not implemented" in str(e.value)
