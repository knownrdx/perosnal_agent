"""Teams device-code sign-in (no client secret, no Azure app registration)."""

from __future__ import annotations

import httpx
import pytest

from app.integrations import BridgeError, TeamsBridge


def _bridge(handler) -> TeamsBridge:
    bridge = TeamsBridge("http://teams:8082", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://teams:8082",
        headers={"X-Bridge-Token": "tok"},
    )
    return bridge


async def test_start_device_login_returns_code_and_url(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["token"] = request.headers.get("X-Bridge-Token")
        return httpx.Response(200, json={
            "user_code": "H7QW2XK9",
            "verification_url": "https://microsoft.com/devicelogin",
            "interval": 5,
            "expires_in": 900,
        })

    result = await _bridge(handler).start_device_login()

    assert seen["path"] == "/login/start"
    assert seen["token"] == "tok"
    assert result["user_code"] == "H7QW2XK9"
    assert result["verification_url"].startswith("https://")
    assert result["interval"] == 5


async def test_poll_reports_pending_then_done(environment):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/login/poll"
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json={"done": False, "pending": True})
        return httpx.Response(200, json={
            "done": True, "pending": False, "account": "arif@contoso.com",
        })

    bridge = _bridge(handler)
    assert (await bridge.poll_device_login())["pending"] is True
    assert (await bridge.poll_device_login())["pending"] is True

    final = await bridge.poll_device_login()
    assert final["done"] is True
    assert final["account"] == "arif@contoso.com"


async def test_poll_reports_declined(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "done": False, "pending": False, "error": "sign-in was declined",
        })

    state = await _bridge(handler).poll_device_login()
    assert state["done"] is False
    assert state["pending"] is False
    assert "declined" in state["error"]


async def test_poll_reports_expired_code(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "done": False, "pending": False, "error": "the code expired; start again",
        })

    state = await _bridge(handler).poll_device_login()
    assert "expired" in state["error"]


async def test_bridge_down_during_login_is_a_bridge_error(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": "device code request failed"})

    with pytest.raises(BridgeError) as exc:
        await _bridge(handler).start_device_login()
    assert exc.value.temporary is True


async def test_status_reports_delegated_mode(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "configured": True, "authenticated": True, "mode": "delegated",
            "account": "arif@contoso.com", "tenant": "organizations",
        })

    status = await _bridge(handler).status()
    assert status["mode"] == "delegated"
    assert status["account"] == "arif@contoso.com"


async def test_status_when_not_signed_in(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "configured": False, "mode": "delegated",
            "error": "not configured - run /teams login to sign in",
        })

    status = await _bridge(handler).status()
    assert status["configured"] is False
    assert "login" in status["error"]


async def test_expired_refresh_token_surfaces_clearly(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "configured": True, "authenticated": False, "mode": "delegated",
            "error": "sign-in expired; run /teams login again",
        })

    status = await _bridge(handler).status()
    assert status["authenticated"] is False
    assert "run /teams login again" in status["error"]


async def test_chats_without_user_works_in_delegated_mode(environment):
    """Delegated sign-in reads /me/chats, so no user id is needed."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chats"
        return httpx.Response(200, json={
            "count": 1,
            "chats": [{"id": "19:abc@thread.v2", "topic": "Standup",
                       "chatType": "group"}],
        })

    bridge = _bridge(handler)
    result = await bridge.request("GET", "/chats", params={"limit": 5})
    assert result["count"] == 1
