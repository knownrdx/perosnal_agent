"""Credential vault + Telegram-driven setup (keys, custom LLMs, accounts)."""

from __future__ import annotations

import pytest

from app.db import repo
from app.db.base import session_scope
from app.llm import LLMError
from app.llm.manager import LLMManager
from app.security.vault import CredentialVault, decrypt, encrypt, mask, reset_vault


@pytest.fixture
def vault(environment):
    reset_vault()
    from app.security.vault import get_vault

    yield get_vault()
    reset_vault()


# --------------------------------------------------------------------------- #
# Encryption
# --------------------------------------------------------------------------- #
def test_roundtrip_encryption(environment):
    secret = "sk-ant-super-secret-value-123"
    token = encrypt(secret)
    assert token != secret
    assert secret not in token
    assert decrypt(token) == secret


def test_mask_hides_the_secret():
    assert mask("sk-ant-abcdefghijklmnop") == "sk-ant...mnop"
    assert "abcdefghij" not in mask("sk-ant-abcdefghijklmnop")
    assert mask("") == "(not set)"


async def test_secret_is_encrypted_at_rest(vault):
    await vault.set("openai_api_key", "sk-plaintext-must-not-appear")

    async with session_scope() as session:
        rows = await repo.list_credentials(session)

    assert len(rows) == 1
    assert "sk-plaintext-must-not-appear" not in rows[0].value
    assert vault.get("openai_api_key") == "sk-plaintext-must-not-appear"


async def test_vault_survives_restart(vault):
    await vault.set("anthropic_api_key", "sk-ant-persisted")

    fresh = CredentialVault()
    await fresh.load()
    assert fresh.get("anthropic_api_key") == "sk-ant-persisted"


async def test_vault_delete(vault):
    await vault.set("openai_api_key", "sk-temp")
    assert await vault.delete("openai_api_key") is True
    assert vault.get("openai_api_key") == ""
    assert await vault.delete("openai_api_key") is False


async def test_vault_rejects_empty(vault):
    from app.security.vault import VaultError

    with pytest.raises(VaultError):
        await vault.set("", "x")
    with pytest.raises(VaultError):
        await vault.set("openai_api_key", "   ")


# --------------------------------------------------------------------------- #
# Vault drives the LLM manager
# --------------------------------------------------------------------------- #
async def test_key_from_vault_activates_provider(vault, monkeypatch):
    """Setting a key from Telegram must work without a restart or .env edit."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from app.config import reload_settings

    reload_settings()

    manager = LLMManager()
    before = {p.key: p.configured for p in manager.configured_providers()}
    assert before["anthropic"] is False

    await vault.set("anthropic_api_key", "sk-ant-from-telegram")

    after = {p.key: p.configured for p in manager.configured_providers()}
    assert after["anthropic"] is True
    assert await manager.set_active("anthropic")


async def test_vault_key_overrides_env(vault, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    from app.config import reload_settings
    from app.llm.manager import resolved_key

    reload_settings()
    assert resolved_key("openai") == "sk-from-env"

    await vault.set("openai_api_key", "sk-from-telegram")
    assert resolved_key("openai") == "sk-from-telegram"


# --------------------------------------------------------------------------- #
# Custom provider registration (/addllm)
# --------------------------------------------------------------------------- #
async def test_add_custom_provider(vault, environment):
    manager = LLMManager()
    info = await manager.add_custom_provider(
        "openrouter", "https://openrouter.ai/api", "anthropic/claude-sonnet-4.5", "sk-or-x"
    )
    assert info.key == "openrouter"

    keys = {p.key for p in manager.configured_providers()}
    assert "openrouter" in keys

    client = manager.client("openrouter")
    assert client.base_url == "https://openrouter.ai/api/v1"   # /v1 appended
    assert client.api_key == "sk-or-x"
    assert vault.has("openrouter_api_key")


async def test_custom_provider_survives_restart(vault, environment):
    manager = LLMManager()
    await manager.add_custom_provider("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b", "gsk-x")
    await manager.set_active("groq")

    restored = LLMManager()
    await restored.load()
    await get_fresh_vault()

    assert "groq" in {p.key for p in restored.configured_providers()}
    assert restored.active_key() == "groq"


async def get_fresh_vault() -> None:
    from app.security.vault import get_vault

    await get_vault().load()


async def test_reserved_names_rejected(environment):
    manager = LLMManager()
    for name in ["openai", "anthropic", "ollama", "omniroute", ""]:
        with pytest.raises(LLMError):
            await manager.add_custom_provider(name, "https://x.com/v1", "m")


async def test_bad_base_url_rejected(environment):
    manager = LLMManager()
    with pytest.raises(LLMError, match="http"):
        await manager.add_custom_provider("weird", "ftp://x.com", "m")


async def test_remove_custom_provider(vault, environment):
    manager = LLMManager()
    await manager.add_custom_provider("temp", "https://x.com/v1", "m", "key")
    assert await manager.remove_custom_provider("temp") is True
    assert "temp" not in {p.key for p in manager.configured_providers()}
    assert not vault.has("temp_api_key")
    assert await manager.remove_custom_provider("temp") is False


# --------------------------------------------------------------------------- #
# Bridge runtime configuration (Teams connect from Telegram)
# --------------------------------------------------------------------------- #
async def test_teams_bridge_configure_call(environment):
    import httpx

    from app.integrations import TeamsBridge

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"configured": True, "authenticated": True,
                                         "tenant": "tenant-1"})

    bridge = TeamsBridge("http://teams:8082", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://teams:8082",
        headers={"X-Bridge-Token": "tok"},
    )
    result = await bridge.configure("tenant-1", "client-1", "secret-1", "19:chat@thread.v2")

    assert seen["path"] == "/config"
    assert seen["body"]["tenant_id"] == "tenant-1"
    assert seen["body"]["default_chat"] == "19:chat@thread.v2"
    assert result["authenticated"] is True


async def test_teams_bridge_rejects_bad_credentials(environment):
    import httpx

    from app.integrations import BridgeError, TeamsBridge

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "credentials rejected: bad secret"})

    bridge = TeamsBridge("http://teams:8082", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://teams:8082"
    )
    with pytest.raises(BridgeError, match="rejected"):
        await bridge.configure("t", "c", "wrong")


async def test_whatsapp_qr_png_fetch(environment):
    import httpx

    from app.integrations import WhatsAppBridge

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/login/qr.png"
        return httpx.Response(200, content=png_bytes,
                              headers={"Content-Type": "image/png"})

    bridge = WhatsAppBridge("http://wa:8081", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://wa:8081"
    )
    data = await bridge.login_qr_png()
    assert data.startswith(b"\x89PNG")


async def test_whatsapp_qr_png_error_is_readable(environment):
    import httpx

    from app.integrations import BridgeError, WhatsAppBridge

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "already paired"})

    bridge = WhatsAppBridge("http://wa:8081", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://wa:8081"
    )
    with pytest.raises(BridgeError, match="already paired"):
        await bridge.login_qr_png()


# --------------------------------------------------------------------------- #
# Gateway health probe falls back to the chat endpoint
# --------------------------------------------------------------------------- #
async def test_gateway_health_uses_chat_when_models_is_gated(environment):
    """OmniRoute gates /models but allows /chat/completions - must report ok."""
    import httpx

    from app.llm.openai_client import OpenAIClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(401, json={"error": {"message": "Authentication required"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAIClient(api_key="omniroute", base_url="http://omniroute:20128/v1",
                          model="auto", provider_label="omniroute")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://omniroute:20128/v1"
    )
    health = await client.health()
    assert health["ok"] is True
    assert "chat endpoint reachable" in health.get("note", "")


async def test_gateway_health_fails_when_chat_also_dead(environment):
    import httpx

    from app.llm.openai_client import OpenAIClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Authentication required"}})

    client = OpenAIClient(api_key="x", base_url="http://gw/v1", model="auto")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://gw/v1"
    )
    health = await client.health()
    assert health["ok"] is False
