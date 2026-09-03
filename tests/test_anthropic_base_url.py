"""Anthropic base-URL handling.

A gateway is usually advertised as ``https://host/v1`` while the official
Anthropic base is bare. Both must work, because the owner pastes whatever their
provider gave them.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm import LLMError, Message
from app.llm.anthropic_client import AnthropicClient, _normalise_base


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://api.anthropic.com", "https://api.anthropic.com"),
        ("https://api.anthropic.com/", "https://api.anthropic.com"),
        ("https://api.mwapi.dev/v1", "https://api.mwapi.dev"),
        ("https://api.mwapi.dev/v1/", "https://api.mwapi.dev"),
        ("  https://gw.example.com/v1  ", "https://gw.example.com"),
        ("https://gw.example.com/proxy", "https://gw.example.com/proxy"),
    ],
)
def test_base_url_is_normalised(given, expected):
    assert _normalise_base(given) == expected


def test_v1_is_stripped_only_once():
    """A path that merely contains 'v1' must not be mangled."""
    assert _normalise_base("https://h/v1/v1") == "https://h/v1"
    assert _normalise_base("https://h/api/v1x") == "https://h/api/v1x"


# --------------------------------------------------------------------------- #
# The request that actually goes out
# --------------------------------------------------------------------------- #
def _client(base_url: str, capture: dict) -> AnthropicClient:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["auth"] = request.headers.get("x-api-key", "")
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "pong"}],
                "model": "claude-opus-4-6",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    client = AnthropicClient(api_key="sk-test-key", base_url=base_url, model="claude-opus-4-6")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=client.base_url,
        headers={"x-api-key": "sk-test-key", "anthropic-version": "2023-06-01"},
    )
    return client


async def test_gateway_url_does_not_double_the_version(environment):
    """The bug this guards: /v1 + /v1/messages = 404."""
    seen: dict = {}
    client = _client("https://api.mwapi.dev/v1", seen)

    response = await client.chat([Message("user", "ping")])

    assert seen["url"] == "https://api.mwapi.dev/v1/messages"
    assert "/v1/v1/" not in seen["url"]
    assert response.content == "pong"


async def test_official_url_still_works(environment):
    seen: dict = {}
    client = _client("https://api.anthropic.com", seen)

    await client.chat([Message("user", "ping")])

    assert seen["url"] == "https://api.anthropic.com/v1/messages"


async def test_api_key_is_sent_as_x_api_key(environment):
    seen: dict = {}
    client = _client("https://api.mwapi.dev/v1", seen)
    await client.chat([Message("user", "ping")])
    assert seen["auth"] == "sk-test-key"


async def test_missing_key_fails_fast(environment):
    client = AnthropicClient(api_key="", base_url="https://api.mwapi.dev/v1")
    with pytest.raises(LLMError, match="API key"):
        await client.chat([Message("user", "ping")])


async def test_json_decision_parses_through_the_gateway(environment):
    """The agent loop needs chat_json, not just chat."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": '{"action": "final", "final_answer": "done"}'}
                ],
                "model": "claude-opus-4-6",
                "usage": {},
            },
        )

    client = AnthropicClient(api_key="k", base_url="https://api.mwapi.dev/v1")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=client.base_url
    )

    decision = await client.chat_json([Message("user", "do it")])
    assert decision["action"] == "final"
    assert decision["final_answer"] == "done"
