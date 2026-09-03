"""LLM presets and the guided /llm -> /paste connection flow."""

from __future__ import annotations

import httpx
import pytest

from app.llm import LLMError
from app.llm.manager import LLMManager
from app.llm.presets import PRESETS, catalogue, resolve, validate_key
from app.security.vault import get_vault, reset_vault


@pytest.fixture
def vault(environment):
    reset_vault()
    yield get_vault()
    reset_vault()


# --------------------------------------------------------------------------- #
# Catalogue sanity
# --------------------------------------------------------------------------- #
def test_catalogue_has_the_big_providers():
    keys = {p.key for p in catalogue()}
    for expected in ["anthropic", "openai", "openrouter", "groq", "gemini",
                     "deepseek", "omniroute", "ollama"]:
        assert expected in keys


def test_every_preset_is_well_formed():
    for key, preset in PRESETS.items():
        assert preset.key == key
        assert preset.base_url.startswith("http"), key
        assert preset.models, key
        assert preset.auth in {"key", "device", "none"}, key
        if preset.auth == "key":
            assert preset.signup_url.startswith("https://"), key
            assert preset.steps, f"{key} must tell the owner what to do"


def test_aliases_resolve():
    assert resolve("claude").key == "anthropic"
    assert resolve("chatgpt").key == "openai"
    assert resolve("Grok").key == "xai"
    assert resolve("local").key == "ollama"
    assert resolve("nonsense") is None


def test_key_validation_catches_mistakes():
    anthropic = PRESETS["anthropic"]
    assert validate_key(anthropic, "") is not None
    assert validate_key(anthropic, "short") is not None
    assert validate_key(anthropic, "sk-ant-with space here") is not None
    assert "sk-ant-" in validate_key(anthropic, "sk-wrongprefix-123456789012")
    assert validate_key(anthropic, "sk-ant-api03-abcdefghijklmnop") is None


def test_free_tier_flags():
    assert PRESETS["groq"].free is True
    assert PRESETS["omniroute"].free is True
    assert PRESETS["anthropic"].free is False


# --------------------------------------------------------------------------- #
# Connecting a preset
# --------------------------------------------------------------------------- #
async def test_connect_preset_stores_and_verifies(vault, monkeypatch):
    manager = LLMManager()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer gsk_abcdefghijklmnop"
        return httpx.Response(200, json={"data": [{"id": "llama-3.3-70b-versatile"}]})

    original = manager.client

    def patched(key=None):
        client = original(key)
        if getattr(client, "base_url", "").startswith("https://api.groq.com"):
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url=client.base_url,
                headers={"Authorization": f"Bearer {client.api_key}"},
            )
        return client

    monkeypatch.setattr(manager, "client", patched)

    result = await manager.connect_preset("groq", "gsk_abcdefghijklmnop")
    assert result["verified"] is True
    assert result["provider"] == "groq"
    assert vault.get("groq_api_key") == "gsk_abcdefghijklmnop"


async def test_connect_preset_rejects_bad_key(vault):
    manager = LLMManager()
    with pytest.raises(LLMError):
        await manager.connect_preset("anthropic", "nope")
    assert not vault.has("anthropic_api_key")


async def test_connect_preset_rejects_unknown_provider(vault):
    with pytest.raises(LLMError, match="unknown provider"):
        await LLMManager().connect_preset("skynet", "sk-1234567890123")


async def test_keyless_preset_cannot_be_connected_with_a_key(vault):
    with pytest.raises(LLMError, match="needs no key"):
        await LLMManager().connect_preset("omniroute", "sk-123456789012")


async def test_preset_becomes_selectable_after_paste(vault):
    """The whole point: paste a key, then the provider is usable."""
    manager = LLMManager()

    assert "groq" not in {p.key for p in manager.configured_providers()}
    with pytest.raises(LLMError, match="/llm groq"):
        await manager.set_active("groq")

    await vault.set("groq_api_key", "gsk_abcdefghijklmnop")

    assert "groq" in {p.key for p in manager.configured_providers()}
    info = await manager.set_active("groq")
    assert info.key == "groq"
    assert manager.active_model() == PRESETS["groq"].default_model


async def test_preset_client_uses_preset_base_url(vault):
    await vault.set("deepseek_api_key", "sk-deepseekkey123456")
    client = LLMManager().client("deepseek")
    assert client.base_url == "https://api.deepseek.com/v1"
    assert client.model == "deepseek-chat"
    assert client.api_key == "sk-deepseekkey123456"


async def test_anthropic_preset_uses_native_client(vault):
    from app.llm.anthropic_client import AnthropicClient

    await vault.set("anthropic_api_key", "sk-ant-api03-abcdefghijkl")
    client = LLMManager().client("anthropic")
    assert isinstance(client, AnthropicClient)


async def test_preset_alias_switch(vault):
    await vault.set("openai_api_key", "sk-openaikey1234567")
    manager = LLMManager()
    info = await manager.set_active("chatgpt")   # alias
    assert info.key == "openai"


async def test_preset_model_override_persists(vault):
    await vault.set("openrouter_api_key", "sk-or-abcdefghijklmno")
    manager = LLMManager()
    await manager.set_active("openrouter", "deepseek/deepseek-chat")
    assert manager.active_model() == "deepseek/deepseek-chat"

    restored = LLMManager()
    await restored.load()
    assert restored.active_key() == "openrouter"
    assert restored.active_model() == "deepseek/deepseek-chat"


async def test_unverified_key_is_still_stored(vault, monkeypatch):
    """A network blip must not lose the key the owner just pasted."""
    manager = LLMManager()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "upstream down"}})

    original = manager.client

    def patched(key=None):
        client = original(key)
        if hasattr(client, "_client"):
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url=getattr(client, "base_url", "http://x/v1"),
            )
        return client

    monkeypatch.setattr(manager, "client", patched)

    result = await manager.connect_preset("deepseek", "sk-deepseekkey123456")
    assert result["verified"] is False
    assert vault.get("deepseek_api_key") == "sk-deepseekkey123456"
