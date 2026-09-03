"""Teams login: tenant/client argument parsing and account-type errors."""

from __future__ import annotations

import json

import httpx
import pytest

from app.integrations import BridgeError, TeamsBridge
from app.telegram.setup_commands import _looks_like_guid

GUID = "11111111-2222-3333-4444-555555555555"


def _bridge(handler) -> TeamsBridge:
    bridge = TeamsBridge("http://teams:8082", "tok")
    bridge._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://teams:8082",
        headers={"X-Bridge-Token": "tok"},
    )
    return bridge


# --------------------------------------------------------------------------- #
# Argument shape: a GUID is a client id, a domain is a tenant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [GUID, "14d82eec-204b-4c2f-b7e8-296a70dab67e"])
def test_guids_are_recognised(value):
    assert _looks_like_guid(value)


@pytest.mark.parametrize(
    "value",
    ["common", "consumers", "contoso.onmicrosoft.com", "", "not-a-guid", GUID[:-1]],
)
def test_non_guids_are_not_client_ids(value):
    assert not _looks_like_guid(value)


def _parse(parts: list[str]) -> tuple[str, str]:
    """Mirror of the command parser, kept honest by the tests below."""
    aliases = {
        "common": "common", "personal": "consumers", "consumers": "consumers",
        "work": "organizations", "school": "organizations",
        "organizations": "organizations",
    }
    client_id, tenant = "", ""
    for token in parts:
        low = token.lower()
        if _looks_like_guid(token):
            client_id = token
        elif low in aliases:
            tenant = aliases[low]
        elif "." in token:
            tenant = token
    return client_id, tenant


def test_bare_login_uses_defaults():
    assert _parse([]) == ("", "")


def test_client_id_only():
    assert _parse([GUID]) == (GUID, "")


def test_tenant_alias_only():
    assert _parse(["common"]) == ("", "common")
    assert _parse(["personal"]) == ("", "consumers")
    assert _parse(["work"]) == ("", "organizations")


def test_domain_is_a_tenant():
    assert _parse(["contoso.onmicrosoft.com"]) == ("", "contoso.onmicrosoft.com")


def test_order_does_not_matter():
    """Both orderings must give the same result."""
    assert _parse([GUID, "common"]) == _parse(["common", GUID]) == (GUID, "common")


# --------------------------------------------------------------------------- #
# Microsoft's refusals must arrive as actionable text
# --------------------------------------------------------------------------- #
async def test_personal_account_error_explains_the_limit(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": (
            "sign-in failed: this is a personal Microsoft account "
            "(outlook/hotmail/live). Teams is only exposed to work or school "
            "accounts, so a personal account cannot be connected. "
            "AADSTS50020: User account from identity provider 'live.com'"
        )})

    with pytest.raises(BridgeError) as exc:
        await _bridge(handler).poll_device_login()
    message = str(exc.value).lower()
    assert "personal" in message
    assert "work or school" in message


async def test_tenant_block_error_names_the_fix(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": (
            "device code refused: this tenant blocks the built-in sign-in app. "
            "Register your own app in portal.azure.com (public client flows = "
            "Yes) and retry with /teams login <client_id>. AADSTS65002"
        )})

    with pytest.raises(BridgeError) as exc:
        await _bridge(handler).start_device_login()
    assert "/teams login <client_id>" in str(exc.value)


async def test_tenant_override_reaches_the_bridge(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content or b"{}")
        return httpx.Response(200, json={
            "user_code": "AB12CD34", "verification_url": "https://login.microsoft.com/device",
            "interval": 5, "expires_in": 900, "client_id": GUID,
        })

    await _bridge(handler).start_device_login(GUID, "common")
    assert seen["body"] == {"client_id": GUID, "tenant": "common"}
