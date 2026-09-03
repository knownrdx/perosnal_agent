"""Multi-provider LLM layer: Claude, ChatGPT, local, switching and fallback."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm import LLMError, Message
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import extract_json
from app.llm.manager import LLMManager
from app.llm.openai_client import OpenAIClient


def _transport(handler):
    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------- #
# JSON extraction (local models wrap JSON in prose / fences)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        '{"action": "final", "final_answer": "hi"}',
        '```json\n{"action": "final", "final_answer": "hi"}\n```',
        'Sure! Here is the plan:\n{"action": "final", "final_answer": "hi"}\nHope that helps.',
        'text {"nested": {"a": "}"}, "action": "final", "final_answer": "hi"} tail',
    ],
)
def test_extract_json_handles_messy_model_output(raw):
    parsed = extract_json(raw)
    assert parsed is not None and parsed["action"] == "final"


def test_extract_json_returns_none_for_garbage():
    assert extract_json("no json at all") is None
    assert extract_json("") is None


# --------------------------------------------------------------------------- #
# OpenAI / ChatGPT
# --------------------------------------------------------------------------- #
async def test_openai_client_parses_response(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": '{"action":"final"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    client._client = httpx.AsyncClient(
        transport=_transport(handler), base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer sk-test"},
    )
    response = await client.chat([Message("user", "hello")])
    assert response.content == '{"action":"final"}'
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert "temperature" in seen["body"]


async def test_openai_reasoning_model_omits_temperature(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAIClient(api_key="sk-test", model="gpt-5")
    client._client = httpx.AsyncClient(transport=_transport(handler),
                                       base_url="https://api.openai.com/v1")
    await client.chat([Message("user", "hi")])
    assert "temperature" not in seen["body"]
    assert "max_completion_tokens" in seen["body"]


async def test_openai_auth_error_is_not_retried(environment):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = OpenAIClient(api_key="sk-bad")
    client._client = httpx.AsyncClient(transport=_transport(handler),
                                       base_url="https://api.openai.com/v1")
    with pytest.raises(LLMError, match="authentication failed"):
        await client.chat([Message("user", "hi")])
    assert calls["n"] == 1


async def test_openai_missing_key_fails_clearly(environment):
    with pytest.raises(LLMError, match="API key"):
        await OpenAIClient(api_key="").chat([Message("user", "hi")])


# --------------------------------------------------------------------------- #
# Anthropic / Claude
# --------------------------------------------------------------------------- #
async def test_anthropic_splits_system_prompt(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": '{"action":"final"}'}],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        })

    client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5")
    client._client = httpx.AsyncClient(
        transport=_transport(handler), base_url="https://api.anthropic.com",
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
    )
    response = await client.chat(
        [Message("system", "you are an agent"), Message("user", "do it")]
    )
    assert response.content == '{"action":"final"}'
    # Claude requires system at top level, not as a message.
    assert seen["body"]["system"] == "you are an agent"
    assert [m["role"] for m in seen["body"]["messages"]] == ["user"]
    assert seen["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_merges_consecutive_same_role_turns():
    system, turns = AnthropicClient._split_system([
        Message("system", "sys"),
        Message("user", "one"),
        Message("user", "two"),
        Message("assistant", "ok"),
    ])
    assert system == "sys"
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "one\n\ntwo"


def test_anthropic_first_turn_is_always_user():
    _, turns = AnthropicClient._split_system([Message("assistant", "hi")])
    assert turns[0]["role"] == "user"


# --------------------------------------------------------------------------- #
# Manager: switching, persistence, fallback
# --------------------------------------------------------------------------- #
async def test_manager_lists_providers_and_marks_configured(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app.config import reload_settings

    reload_settings()

    providers = {p.key: p for p in LLMManager().configured_providers()}
    assert providers["anthropic"].configured is True
    assert providers["openai"].configured is False
    assert providers["ollama"].configured is True


async def test_switch_provider_requires_api_key(environment, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app.config import reload_settings

    reload_settings()
    with pytest.raises(LLMError, match="no API key"):
        await LLMManager().set_active("openai")


async def test_switch_and_persist_across_restart(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    from app.config import reload_settings

    reload_settings()

    manager = LLMManager()
    info = await manager.set_active("anthropic", "claude-opus-4-1")
    assert info.key == "anthropic"
    assert manager.active_model() == "claude-opus-4-1"

    # A fresh manager (simulating a restart) restores the choice from the DB.
    restored = LLMManager()
    await restored.load()
    assert restored.active_key() == "anthropic"
    assert restored.active_model() == "claude-opus-4-1"


async def test_set_model_keeps_provider(environment, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.config import reload_settings

    reload_settings()
    manager = LLMManager()
    await manager.set_active("openai")
    await manager.set_model("gpt-4o")
    assert manager.active_key() == "openai"
    assert manager.active_model() == "gpt-4o"


async def test_unknown_provider_rejected(environment):
    with pytest.raises(LLMError, match="unknown provider"):
        await LLMManager().set_active("skynet")


async def test_fallback_used_when_active_provider_fails(environment, monkeypatch):
    """A cloud outage must not kill an overnight task."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-x")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    from app.config import reload_settings

    reload_settings()

    from app.llm.base import LLMClient, LLMResponse

    class Failing(LLMClient):
        name = "failing"

        async def chat(self, messages, *, temperature=None):
            raise LLMError("provider is down")

        async def health(self):
            return {"ok": False}

    class Working(LLMClient):
        name = "working"

        async def chat(self, messages, *, temperature=None):
            return LLMResponse(content='{"action":"final"}', model="backup")

        async def health(self):
            return {"ok": True}

    manager = LLMManager()
    await manager.set_active("anthropic")
    manager._clients["anthropic"] = Failing()
    manager._clients["openai"] = Working()

    response = await manager.chat([Message("user", "hi")])
    assert response.content == '{"action":"final"}'


async def test_all_providers_failing_raises(environment, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    from app.config import reload_settings

    reload_settings()

    from app.llm.base import LLMClient

    class Failing(LLMClient):
        name = "failing"

        async def chat(self, messages, *, temperature=None):
            raise LLMError("down")

        async def health(self):
            return {"ok": False}

    manager = LLMManager()
    await manager.set_active("anthropic")
    manager._clients["anthropic"] = Failing()

    with pytest.raises(LLMError, match="all LLM providers failed"):
        await manager.chat([Message("user", "hi")])


async def test_manager_chat_json_parses_fenced_output(environment, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.config import reload_settings

    reload_settings()

    from app.llm.base import LLMClient, LLMResponse

    class Fenced(LLMClient):
        name = "fenced"

        async def chat(self, messages, *, temperature=None):
            return LLMResponse(content='```json\n{"action": "tool", "tool": "file_read"}\n```')

        async def health(self):
            return {"ok": True}

    manager = LLMManager()
    await manager.set_active("openai")
    manager._clients["openai"] = Fenced()

    decision = await manager.chat_json([Message("user", "hi")])
    assert decision["tool"] == "file_read"
