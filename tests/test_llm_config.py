"""Provider-layer configuration tests.

These check wiring, not model behaviour: no network call is made.
"""

import pytest

from cish import llm


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("CISH_PROVIDER", raising=False)
    assert llm.provider() == "groq"


def test_provider_is_read_at_call_time(monkeypatch):
    """evaluate.py --compare switches providers inside one process, so this
    must not be cached at import."""
    monkeypatch.setenv("CISH_PROVIDER", "openai")
    assert llm.provider() == "openai"
    monkeypatch.setenv("CISH_PROVIDER", "ollama")
    assert llm.provider() == "ollama"


def test_active_model_identifies_the_provider(monkeypatch):
    """The audit log records this, so a decision can be traced to the exact
    model that produced it."""
    monkeypatch.delenv("CISH_MODEL", raising=False)
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    assert llm.active_model().startswith("groq/")
    monkeypatch.setenv("CISH_PROVIDER", "ollama")
    assert llm.active_model().startswith("ollama/")
    monkeypatch.setenv("CISH_PROVIDER", "anthropic")
    assert "claude" in llm.active_model()


def test_model_override_is_respected(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.setenv("CISH_MODEL", "qwen/qwen3-32b")
    assert llm.active_model() == "groq/qwen/qwen3-32b"


def test_missing_key_raises_a_useful_error(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CISH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.complete("hello")


def test_generic_key_var_works_for_any_provider(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("CISH_API_KEY", "test-key")
    base, model, key = llm._openai_config("openrouter")
    assert key == "test-key"
    assert base.startswith("https://")


def test_custom_provider_requires_a_base_url(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "custom")
    monkeypatch.setenv("CISH_API_KEY", "test-key")
    monkeypatch.delenv("CISH_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="CISH_BASE_URL"):
        llm._openai_config("custom")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "not-a-provider")
    with pytest.raises(RuntimeError, match="Unknown CISH_PROVIDER"):
        llm.complete("hello")


def test_available_is_false_without_a_key(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CISH_API_KEY", raising=False)
    assert llm.available() is False


def test_available_is_true_with_a_key(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert llm.available() is True


@pytest.mark.parametrize("name", ["groq", "openai", "openrouter", "together"])
def test_every_openai_compatible_provider_has_a_default_model(name,
                                                              monkeypatch):
    monkeypatch.delenv("CISH_MODEL", raising=False)
    monkeypatch.setenv("CISH_API_KEY", "test-key")
    base, model, _ = llm._openai_config(name)
    assert base.startswith("https://")
    assert model


def test_requests_send_a_custom_user_agent(monkeypatch):
    """urllib's default User-Agent gets a 403 from Cloudflare-fronted APIs
    such as Groq. This pins the fix so it cannot regress."""
    import urllib.request
    seen = {}

    class FakeResp:
        def read(self):
            return b'{"choices": [{"message": {"content": "{}"}}]}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm.complete("hi")
    assert seen["ua"] and "Python-urllib" not in seen["ua"]
