"""Teams auth fallbacks: client-id override, token paste, blocked-tenant errors."""

from __future__ import annotations

import json

import httpx
import pytest

from app.integrations import BridgeError, TeamsBridge

AADSTS65002 = (
    "device code refused: AADSTS65002: Consent between first party application "
    "'04b07795-8ddb-461a-bbee-02f9e1bf7b46' and first party resource "
    "'00000003-0000-0000-c000-000000000000' must be configured via preauthorization"
)


def _bridge(handler) -> TeamsBridge:
    bridge = TeamsBridge("http://teams:8082", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://teams:8082",
        headers={"X-Bridge-Token": "tok"},
    )
    return bridge


# --------------------------------------------------------------------------- #
# Client id / tenant override
# --------------------------------------------------------------------------- #
async def test_login_without_override_sends_empty_body(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json={
            "user_code": "ABC123", "verification_url": "https://microsoft.com/devicelogin",
            "interval": 5, "expires_in": 900,
            "client_id": "14d82eec-204b-4c2f-b7e8-296a70dab67e",
        })

    result = await _bridge(handler).start_device_login()
    assert seen["body"] == {}
    # The Graph CLI client is the one preauthorized for Graph.
    assert result["client_id"] == "14d82eec-204b-4c2f-b7e8-296a70dab67e"


async def test_login_with_own_client_id(environment):
    """Blocked tenants can point the flow at their own app registration."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json={
            "user_code": "XYZ789", "verification_url": "https://microsoft.com/devicelogin",
            "interval": 5, "expires_in": 900, "client_id": seen["body"]["client_id"],
        })

    result = await _bridge(handler).start_device_login(
        "11111111-2222-3333-4444-555555555555", "contoso.onmicrosoft.com"
    )
    assert seen["body"]["client_id"] == "11111111-2222-3333-4444-555555555555"
    assert seen["body"]["tenant"] == "contoso.onmicrosoft.com"
    assert result["user_code"] == "XYZ789"


async def test_blocked_tenant_error_is_surfaced(environment):
    """AADSTS65002 must reach the caller so the owner sees a real fix."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": AADSTS65002})

    with pytest.raises(BridgeError) as exc:
        await _bridge(handler).start_device_login()
    assert "65002" in str(exc.value)


# --------------------------------------------------------------------------- #
# Token paste
# --------------------------------------------------------------------------- #
async def test_login_with_pasted_token(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "authenticated": True, "account": "arif@contoso.com",
            "expires_in_minutes": 55,
        })

    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9." + "x" * 120
    result = await _bridge(handler).login_with_token(token)

    assert seen["path"] == "/login/token"
    assert seen["body"]["access_token"] == token
    assert result["authenticated"] is True
    assert result["account"] == "arif@contoso.com"


async def test_bad_token_is_rejected(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "token rejected by Graph: 401"})

    with pytest.raises(BridgeError) as exc:
        await _bridge(handler).login_with_token("x" * 60)
    assert "rejected" in str(exc.value)
    assert exc.value.temporary is False


async def test_short_token_rejected_by_bridge(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "that does not look like an access token"})

    with pytest.raises(BridgeError, match="access token"):
        await _bridge(handler).login_with_token("abc")


async def test_expired_pasted_token_shows_in_status(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "configured": True, "authenticated": False, "mode": "delegated",
            "error": "not signed in; start device sign-in with /login",
        })

    status = await _bridge(handler).status()
    assert status["authenticated"] is False
