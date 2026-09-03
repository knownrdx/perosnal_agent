"""HTTP clients for the Go bridges (WhatsApp / Teams).

The bridges own the protocol details; Python only speaks a small JSON API to
them.  Every call is authenticated with the shared BRIDGE_TOKEN, has a
timeout, and maps transport problems onto the agent's failure taxonomy so the
tool framework can decide whether to retry.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)


class BridgeError(Exception):
    """A bridge call failed. ``temporary`` drives the retry decision."""

    def __init__(self, message: str, *, temporary: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.temporary = temporary
        self.status = status


class BridgeClient:
    """Thin authenticated JSON client for one bridge service."""

    channel = "bridge"

    def __init__(self, base_url: str, token: str, timeout_s: int | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s or settings.bridge_timeout_s
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
                headers={"X-Bridge-Token": self.token, "Content-Type": "application/json"},
            )
        return self._client

    async def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.token:
            raise BridgeError("BRIDGE_TOKEN is not configured")
        try:
            response = await self._http().request(method, path, json=json, params=params)
        except httpx.ConnectError as exc:
            raise BridgeError(
                f"{self.channel} bridge unreachable at {self.base_url}", temporary=True
            ) from exc
        except httpx.TimeoutException as exc:
            raise BridgeError(f"{self.channel} bridge timed out", temporary=True) from exc
        except httpx.HTTPError as exc:
            raise BridgeError(f"{self.channel} bridge error: {exc}", temporary=True) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text[:300]}

        if response.status_code >= 400:
            detail = str(payload.get("error") or payload)[:300]
            raise BridgeError(
                detail,
                temporary=response.status_code in {429, 502, 503, 504},
                status=response.status_code,
            )
        return payload if isinstance(payload, dict) else {"result": payload}

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return {"ok": True, "channel": self.channel, **response.json()}
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "channel": self.channel, "error": str(exc)[:200]}

    async def status(self) -> dict[str, Any]:
        return await self.request("GET", "/status")

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class WhatsAppBridge(BridgeClient):
    channel = "whatsapp"

    async def send_text(self, to: str, text: str) -> dict[str, Any]:
        return await self.request("POST", "/send/text", json={"to": to, "text": text})

    async def send_file(self, to: str, path: str, caption: str = "") -> dict[str, Any]:
        return await self.request(
            "POST", "/send/file", json={"to": to, "path": path, "caption": caption}
        )

    async def login_qr(self) -> dict[str, Any]:
        return await self.request("POST", "/login/qr")

    async def login_qr_png(self) -> bytes:
        """Raw PNG of the pairing code, ready to send as a Telegram photo."""
        if not self.token:
            raise BridgeError("BRIDGE_TOKEN is not configured")
        try:
            response = await self._http().get("/login/qr.png")
        except httpx.HTTPError as exc:
            raise BridgeError(f"whatsapp bridge unreachable: {exc}", temporary=True) from exc
        if response.status_code >= 400:
            detail = "could not generate QR"
            try:
                detail = str(response.json().get("error", detail))
            except ValueError:
                pass
            raise BridgeError(detail, status=response.status_code)
        return response.content

    async def logout(self) -> dict[str, Any]:
        return await self.request("POST", "/logout")

    async def messages(self, limit: int = 20, since: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if since:
            params["since"] = since
        return await self.request("GET", "/messages", params=params)


class TeamsBridge(BridgeClient):
    channel = "teams"

    async def send_text(self, chat: str, text: str) -> dict[str, Any]:
        return await self.request("POST", "/send/text", json={"chat": chat, "text": text})

    async def messages(self, chat: str = "", limit: int = 20) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if chat:
            params["chat"] = chat
        return await self.request("GET", "/messages", params=params)

    async def chats(self, user: str, limit: int = 20) -> dict[str, Any]:
        return await self.request("GET", "/chats", params={"user": user, "limit": limit})

    async def download(self, url: str, path: str) -> dict[str, Any]:
        return await self.request("POST", "/download", json={"url": url, "path": path})

    async def configure(
        self, tenant_id: str, client_id: str, client_secret: str, default_chat: str = ""
    ) -> dict[str, Any]:
        """Set Graph credentials at runtime; the bridge verifies them."""
        return await self.request(
            "POST",
            "/config",
            json={
                "tenant_id": tenant_id,
                "client_id": client_id,
                "client_secret": client_secret,
                "default_chat": default_chat,
            },
        )

    async def disconnect(self) -> dict[str, Any]:
        return await self.request("POST", "/disconnect")

    async def start_device_login(
        self, client_id: str = "", tenant: str = ""
    ) -> dict[str, Any]:
        """Begin device-code sign-in; returns the code + URL to show the owner.

        ``client_id``/``tenant`` let the owner point the flow at their own app
        registration when tenant policy blocks the default public client.
        """
        payload: dict[str, Any] = {}
        if client_id:
            payload["client_id"] = client_id
        if tenant:
            payload["tenant"] = tenant
        return await self.request("POST", "/login/start", json=payload)

    async def login_with_token(self, access_token: str) -> dict[str, Any]:
        """Authenticate with a Graph access token pasted by the owner."""
        return await self.request(
            "POST", "/login/token", json={"access_token": access_token}
        )

    async def poll_device_login(self) -> dict[str, Any]:
        """One poll tick: {done, pending, account?, error?}."""
        return await self.request("POST", "/login/poll")


_whatsapp: WhatsAppBridge | None = None
_teams: TeamsBridge | None = None


def whatsapp_bridge() -> WhatsAppBridge:
    global _whatsapp
    if _whatsapp is None:
        settings = get_settings()
        _whatsapp = WhatsAppBridge(settings.whatsapp_bridge_url, settings.bridge_token)
    return _whatsapp


def teams_bridge() -> TeamsBridge:
    global _teams
    if _teams is None:
        settings = get_settings()
        _teams = TeamsBridge(settings.teams_bridge_url, settings.bridge_token)
    return _teams


def set_bridges(whatsapp: WhatsAppBridge | None = None, teams: TeamsBridge | None = None) -> None:
    """Injection point for tests."""
    global _whatsapp, _teams
    _whatsapp = whatsapp
    _teams = teams


async def close_bridges() -> None:
    global _whatsapp, _teams
    for bridge in (_whatsapp, _teams):
        if bridge is not None:
            try:
                await bridge.close()
            except Exception:  # noqa: BLE001
                pass
    _whatsapp = None
    _teams = None
