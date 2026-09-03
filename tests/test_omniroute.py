"""OmniRoute gateway as the zero-key default provider."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm.manager import KNOWN_MODELS, LLMManager


@pytest.fixture
def no_keys(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OMNIROUTE_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")  # configured but keyless
    from app.config import reload_settings

    return reload_settings()


async def test_defaults_to_omniroute_when_no_personal_key(no_keys):
    """With no Claude/ChatGPT key, the agent must still have a working model."""
    manager = LLMManager()
    assert manager.active_key() == "omniroute"
    assert manager.active_model() == "auto"


async def test_personal_key_wins_over_gateway(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from app.config import reload_settings

    reload_settings()
    assert LLMManager().active_key() == "anthropic"


async def test_omniroute_listed_first_and_always_configured(no_keys):
    providers = LLMManager().configured_providers()
    assert providers[0].key == "omniroute"
    assert providers[0].configured is True


async def test_can_disable_omniroute(environment, monkeypatch):
    monkeypatch.setenv("OMNIROUTE_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from app.config import reload_settings

    reload_settings()
    manager = LLMManager()
    assert "omniroute" not in {p.key for p in manager.configured_providers()}
    assert manager.active_key() == "anthropic"


async def test_omniroute_client_is_openai_compatible(no_keys):
    """The gateway speaks the OpenAI schema, so the OpenAI client drives it."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":"final","final_answer":"ok"}'}}]
        })

    manager = LLMManager()
    client = manager.client("omniroute")
    assert client.base_url.endswith("/v1")

    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=client.base_url
    )
    from app.llm import Message

    response = await client.chat([Message("user", "hi")])
    assert seen["path"].endswith("/chat/completions")
    assert seen["body"]["model"] == "auto"
    assert "final_answer" in response.content


async def test_switch_to_omniroute_from_telegram_alias(no_keys):
    manager = LLMManager()
    info = await manager.set_active("omniroute", "auto/coding")
    assert info.key == "omniroute"
    assert manager.active_model() == "auto/coding"


async def test_fallback_prefers_personal_keys_then_gateway(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-x")
    monkeypatch.setenv("OMNIROUTE_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    from app.config import reload_settings

    reload_settings()

    manager = LLMManager()
    await manager.set_active("anthropic")
    order = manager.fallback_order()

    assert order[0] == "anthropic"
    # paid keys before the shared gateway, local model last
    assert order.index("openai") < order.index("omniroute")
    assert order.index("omniroute") < order.index("ollama")


async def test_gateway_rescues_a_task_when_paid_provider_dies(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OMNIROUTE_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    from app.config import reload_settings

    reload_settings()

    from app.llm import LLMError, Message
    from app.llm.base import LLMClient, LLMResponse

    class Dead(LLMClient):
        name = "dead"

        async def chat(self, messages, *, temperature=None):
            raise LLMError("quota exhausted")

        async def health(self):
            return {"ok": False}

    class Gateway(LLMClient):
        name = "omniroute"

        async def chat(self, messages, *, temperature=None):
            return LLMResponse(content='{"action":"final","final_answer":"done via gateway"}')

        async def health(self):
            return {"ok": True}

    manager = LLMManager()
    await manager.set_active("anthropic")
    manager._clients["anthropic"] = Dead()
    manager._clients["omniroute"] = Gateway()

    decision = await manager.chat_json([Message("user", "do it")])
    assert decision["final_answer"] == "done via gateway"


def test_known_models_include_gateway_routes():
    assert "omniroute" in KNOWN_MODELS
    assert "auto" in KNOWN_MODELS["omniroute"]
