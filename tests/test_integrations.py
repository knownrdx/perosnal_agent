"""WhatsApp / Teams bridges: tools, inbound webhooks, auto-task, dedupe."""

from __future__ import annotations

import httpx
import pytest

from app.db import repo
from app.db.base import session_scope
from app.integrations import BridgeError, TeamsBridge, WhatsAppBridge, set_bridges
from app.integrations.inbound import handle_inbound


class FakeBridge:
    """Records calls; can be told to fail in specific ways."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.calls: list[tuple[str, dict]] = []
        self.fail: BridgeError | None = None
        self.next_id = 100

    async def _maybe_fail(self) -> None:
        if self.fail is not None:
            error, self.fail = self.fail, None
            raise error

    async def send_text(self, to_or_chat: str, text: str) -> dict:
        self.calls.append(("send_text", {"to": to_or_chat, "text": text}))
        await self._maybe_fail()
        self.next_id += 1
        return {"sent": True, "message_id": f"MSG{self.next_id}", "chat": to_or_chat}

    async def send_file(self, to: str, path: str, caption: str = "") -> dict:
        self.calls.append(("send_file", {"to": to, "path": path, "caption": caption}))
        await self._maybe_fail()
        self.next_id += 1
        return {"sent": True, "message_id": f"MSG{self.next_id}", "path": path}

    async def messages(self, chat: str = "", limit: int = 20, since=None) -> dict:
        self.calls.append(("messages", {"chat": chat, "limit": limit}))
        await self._maybe_fail()
        return {"count": 1, "chat": chat,
                "messages": [{"id": "1", "sender": "8801711111111", "text": "hi"}]}

    async def status(self) -> dict:
        await self._maybe_fail()
        return {"connected": True, "logged_in": True, "jid": "8801700000000@s.whatsapp.net"}

    async def download(self, url: str, path: str) -> dict:
        self.calls.append(("download", {"url": url, "path": path}))
        await self._maybe_fail()
        return {"downloaded": True, "path": path, "size_bytes": 2048}

    async def login_qr(self) -> dict:
        return {"logged_in": False, "qr": "2@abc", "qr_ascii": "##QR##"}

    async def close(self) -> None:
        return None


@pytest.fixture
def bridges(environment, monkeypatch):
    """Enable both integrations and swap in fake bridges."""
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("TEAMS_ENABLED", "true")
    monkeypatch.setenv("BRIDGE_TOKEN", "test-bridge-token")
    from app.config import reload_settings

    reload_settings()

    whatsapp, teams = FakeBridge("whatsapp"), FakeBridge("teams")
    set_bridges(whatsapp, teams)

    # messaging tools read the enable flags at import time -> re-register them.
    import importlib

    from app.tools import registry

    for name in [
        "whatsapp_send_message", "whatsapp_send_file", "whatsapp_read_messages",
        "whatsapp_status", "teams_send_message", "teams_read_messages",
        "teams_download_file", "teams_status",
    ]:
        registry.unregister(name)

    from app.tools import messaging_tools

    importlib.reload(messaging_tools)

    yield whatsapp, teams

    set_bridges(None, None)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
async def test_whatsapp_tools_are_registered_only_when_enabled(bridges):
    from app.tools import registry

    assert "whatsapp_send_message" in registry.names()
    assert "teams_send_message" in registry.names()


async def test_whatsapp_send_message_verified(bridges):
    whatsapp, _ = bridges
    from app.tools import registry
    from app.tools.base import ToolContext

    result = await registry.get("whatsapp_send_message").run(
        {"to": "8801712345678", "text": "hello"}, ToolContext(task_id="t1")
    )
    assert result.ok and result.data["message_id"].startswith("MSG")
    assert whatsapp.calls[0][1]["to"] == "8801712345678"


async def test_whatsapp_send_message_rejects_empty_text(bridges):
    from app.tools import registry
    from app.tools.base import ToolContext

    result = await registry.get("whatsapp_send_message").run(
        {"to": "8801712345678", "text": "   "}, ToolContext()
    )
    assert not result.ok and "empty" in result.error


async def test_whatsapp_send_file_blocks_traversal_and_empty(bridges, environment):
    from app.security import safe_path
    from app.tools import registry
    from app.tools.base import ToolContext

    tool = registry.get("whatsapp_send_file")

    escape = await tool.run({"to": "880171", "path": "../../etc/passwd"}, ToolContext())
    assert not escape.ok and "escapes workspace" in escape.error

    empty = safe_path("output/empty.bin")
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    result = await tool.run({"to": "880171", "path": "output/empty.bin"}, ToolContext())
    assert not result.ok and "empty file" in result.error

    good = safe_path("output/doc.txt")
    good.write_text("payload", encoding="utf-8")
    ok = await tool.run({"to": "880171", "path": "output/doc.txt"}, ToolContext())
    assert ok.ok and ok.data["message_id"]


async def test_bridge_temporary_error_is_retried(bridges):
    whatsapp, _ = bridges
    from app.tools import registry
    from app.tools.base import ToolContext

    tool = registry.get("whatsapp_send_message")
    tool.retry_backoff_s = 0.01
    whatsapp.fail = BridgeError("bridge unreachable", temporary=True)

    result = await tool.run({"to": "880171", "text": "retry me"}, ToolContext())
    assert result.ok and result.attempts == 2


async def test_bridge_permanent_error_is_not_retried(bridges):
    whatsapp, _ = bridges
    from app.tools import registry
    from app.tools.base import ToolContext

    tool = registry.get("whatsapp_send_message")
    tool.retry_backoff_s = 0.01
    whatsapp.fail = BridgeError("not linked", temporary=False, status=503)

    result = await tool.run({"to": "880171", "text": "no retry"}, ToolContext())
    assert not result.ok and result.attempts == 1


async def test_teams_send_and_read(bridges):
    _, teams = bridges
    from app.tools import registry
    from app.tools.base import ToolContext

    sent = await registry.get("teams_send_message").run(
        {"text": "standup done", "chat": "19:abc@thread.v2"}, ToolContext(task_id="t2")
    )
    assert sent.ok and sent.data["message_id"]

    read = await registry.get("teams_read_messages").run(
        {"chat": "19:abc@thread.v2", "limit": 5}, ToolContext()
    )
    assert read.ok and read.data["count"] == 1


async def test_teams_download_verifies_non_empty(bridges):
    from app.tools import registry
    from app.tools.base import ToolContext

    result = await registry.get("teams_download_file").run(
        {"url": "https://graph.microsoft.com/v1.0/x/content", "path": "downloads/teams/a.bin"},
        ToolContext(),
    )
    assert result.ok and result.data["size_bytes"] == 2048


# --------------------------------------------------------------------------- #
# Bridge HTTP client
# --------------------------------------------------------------------------- #
async def test_bridge_client_sends_token_and_maps_errors(environment):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers.get("X-Bridge-Token")
        if request.url.path == "/send/text":
            return httpx.Response(200, json={"sent": True, "message_id": "X1"})
        if request.url.path == "/status":
            return httpx.Response(401, json={"error": "invalid bridge token"})
        return httpx.Response(503, json={"error": "not linked"})

    client = WhatsAppBridge("http://bridge:8081", "secret")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://bridge:8081",
        headers={"X-Bridge-Token": "secret"},
    )

    result = await client.send_text("880171", "hi")
    assert result["message_id"] == "X1"
    assert seen["token"] == "secret"

    with pytest.raises(BridgeError) as exc:
        await client.status()
    assert exc.value.status == 401
    assert exc.value.temporary is False


async def test_bridge_client_marks_503_temporary(environment):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "starting up"})

    client = TeamsBridge("http://bridge:8082", "secret")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://bridge:8082"
    )
    with pytest.raises(BridgeError) as exc:
        await client.send_text("chat", "hi")
    assert exc.value.temporary is True


async def test_bridge_requires_token(environment):
    client = WhatsAppBridge("http://bridge:8081", "")
    with pytest.raises(BridgeError, match="BRIDGE_TOKEN"):
        await client.send_text("880171", "hi")


# --------------------------------------------------------------------------- #
# Inbound webhook handling
# --------------------------------------------------------------------------- #
WA_PAYLOAD = {
    "channel": "whatsapp",
    "message": {
        "id": "WA123", "chat": "8801711111111@s.whatsapp.net",
        "sender": "8801711111111", "sender_name": "Rahim",
        "text": "please send the report", "from_me": False, "timestamp": 1735000000,
    },
}


async def test_inbound_message_is_stored(environment):
    result = await handle_inbound("whatsapp", WA_PAYLOAD)
    assert result["stored"] is True

    async with session_scope() as session:
        rows = await repo.list_inbound(session)
    assert len(rows) == 1
    assert rows[0].sender == "8801711111111"
    assert rows[0].text == "please send the report"


async def test_inbound_duplicate_is_ignored(environment):
    first = await handle_inbound("whatsapp", WA_PAYLOAD)
    second = await handle_inbound("whatsapp", WA_PAYLOAD)
    assert first["stored"] is True
    assert second["stored"] is False and second.get("duplicate") is True

    async with session_scope() as session:
        rows = await repo.list_inbound(session)
    assert len(rows) == 1


async def test_own_and_empty_messages_ignored(environment):
    own = dict(WA_PAYLOAD)
    own["message"] = {**WA_PAYLOAD["message"], "id": "WA_OWN", "from_me": True}
    assert (await handle_inbound("whatsapp", own))["stored"] is False

    empty = dict(WA_PAYLOAD)
    empty["message"] = {**WA_PAYLOAD["message"], "id": "WA_EMPTY", "text": ""}
    assert (await handle_inbound("whatsapp", empty))["stored"] is False


async def test_unknown_channel_rejected(environment):
    assert (await handle_inbound("signal", WA_PAYLOAD))["stored"] is False


async def test_auto_task_only_for_allowlisted_sender(environment, monkeypatch):
    monkeypatch.setenv("INBOUND_AUTO_TASK", "true")
    monkeypatch.setenv("INBOUND_ALLOWED_SENDERS", "8801711111111")
    from app.config import reload_settings

    reload_settings()

    allowed = await handle_inbound("whatsapp", WA_PAYLOAD)
    assert allowed["auto_task"] is True and allowed["task_id"]

    stranger = {"channel": "whatsapp", "message": {
        **WA_PAYLOAD["message"], "id": "WA999", "sender": "8801799999999"}}
    blocked = await handle_inbound("whatsapp", stranger)
    assert blocked["stored"] is True
    assert blocked["auto_task"] is False, "unknown senders must not be able to run tasks"

    async with session_scope() as session:
        tasks = await repo.list_tasks(session, limit=10)
    assert len(tasks) == 1


async def test_auto_task_disabled_by_default_notifies_owner(environment, fake_telegram):
    from app.telegram.notifier import Notifier

    result = await handle_inbound("whatsapp", WA_PAYLOAD, notifier=Notifier())
    assert result["stored"] is True and result["auto_task"] is False
    texts = fake_telegram.sent_messages()
    assert texts and "New whatsapp message" in texts[0]
    assert "please send the report" in texts[0]


async def test_teams_inbound_normalised(environment):
    payload = {"channel": "teams", "message": {
        "id": "T1", "chat": "19:chat@thread.v2", "sender": "user-guid",
        "sender_name": "Karim", "text": "deploy finished", "created_at": "2026-01-01T00:00:00Z"}}
    result = await handle_inbound("teams", payload)
    assert result["stored"] is True

    async with session_scope() as session:
        rows = await repo.list_inbound(session, channel="teams")
    assert rows[0].sender_name == "Karim"


# --------------------------------------------------------------------------- #
# Sender allowlist matching
# --------------------------------------------------------------------------- #
def test_sender_allowlist_matching(environment, monkeypatch):
    monkeypatch.setenv("INBOUND_ALLOWED_SENDERS", "8801711111111,karim@corp.com")
    from app.config import reload_settings

    settings = reload_settings()
    assert settings.sender_allowed("8801711111111")
    assert settings.sender_allowed("+880 1711 111111")   # formatting ignored
    assert settings.sender_allowed("KARIM@corp.com")     # case-insensitive
    assert not settings.sender_allowed("8801799999999")
    assert not settings.sender_allowed("")


def test_empty_allowlist_denies_everyone(environment, monkeypatch):
    monkeypatch.setenv("INBOUND_ALLOWED_SENDERS", "")
    from app.config import reload_settings

    settings = reload_settings()
    assert not settings.sender_allowed("8801711111111")
