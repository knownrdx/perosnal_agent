"""Contacts/members: repo layer, tools, and Telegram commands."""

from __future__ import annotations

import pytest

from app.db import repo
from app.db.base import session_scope
from app.integrations import BridgeError, set_bridges


class FakeUserbot:
    def __init__(self, dialogs=None, linked: bool = True) -> None:
        self._dialogs = dialogs if dialogs is not None else [
            {"id": 111, "name": "Alice", "username": "alice", "unread": 0,
             "is_group": False, "is_channel": False, "is_user": True},
            {"id": 222, "name": "Work Group", "username": "", "unread": 0,
             "is_group": True, "is_channel": False, "is_user": False},
            {"id": 333, "name": "Bob", "username": "bobby", "unread": 0,
             "is_group": False, "is_channel": False, "is_user": True},
        ]
        self._linked = linked

    def has_session(self) -> bool:
        return self._linked

    async def list_dialogs(self, limit: int = 30):
        return self._dialogs


class FakeWhatsAppBridge:
    """Stands in for app.integrations.bridge_client.WhatsAppBridge."""

    def __init__(self, contacts=None, fail: BridgeError | None = None) -> None:
        self._contacts = contacts if contacts is not None else [
            {"jid": "8801711111111@s.whatsapp.net", "name": "Karim", "phone": "8801711111111"},
            {"jid": "8801722222222@s.whatsapp.net", "name": "Rahim", "phone": "8801722222222"},
        ]
        self._fail = fail

    async def request(self, method: str, path: str, **kwargs):
        if self._fail is not None:
            raise self._fail
        assert path == "/contacts"
        return {"contacts": self._contacts}

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_userbot(monkeypatch):
    from app.tools import contact_tools

    userbot = FakeUserbot()
    monkeypatch.setattr(
        "app.integrations.telegram_user.get_userbot", lambda: userbot
    )
    return userbot


@pytest.fixture
def fake_whatsapp(environment):
    bridge = FakeWhatsAppBridge()
    set_bridges(whatsapp=bridge)
    yield bridge
    set_bridges(None, None)


# --------------------------------------------------------------------------- #
# Repo layer
# --------------------------------------------------------------------------- #
async def test_upsert_contact_creates_then_updates(environment):
    async with session_scope() as session:
        created = await repo.upsert_contact(
            session, channel="telegram", external_id="111",
            display_name="Alice", username_or_phone="alice", seen_in=["Family"],
        )
        assert created.display_name == "Alice"

        updated = await repo.upsert_contact(
            session, channel="telegram", external_id="111",
            display_name="Alice W.", username_or_phone="alice", seen_in=["Work"],
        )
        assert updated.id == created.id
        assert updated.display_name == "Alice W."
        assert set(updated.seen_in) == {"Family", "Work"}

    async with session_scope() as session:
        rows = await repo.list_contacts(session)
    assert len(rows) == 1, "re-sync must upsert, not duplicate"


async def test_list_contacts_filters_by_channel(environment):
    async with session_scope() as session:
        await repo.upsert_contact(session, channel="telegram", external_id="1", display_name="A")
        await repo.upsert_contact(session, channel="whatsapp", external_id="2", display_name="B")

    async with session_scope() as session:
        telegram_only = await repo.list_contacts(session, channel="telegram")
        everyone = await repo.list_contacts(session)
    assert [c.display_name for c in telegram_only] == ["A"]
    assert len(everyone) == 2


async def test_search_contacts_matches_name_or_phone(environment):
    async with session_scope() as session:
        await repo.upsert_contact(
            session, channel="whatsapp", external_id="1",
            display_name="Karim Uddin", username_or_phone="8801711111111",
        )
        await repo.upsert_contact(
            session, channel="whatsapp", external_id="2",
            display_name="Nusrat", username_or_phone="8801799999999",
        )

    async with session_scope() as session:
        by_name = await repo.search_contacts(session, "karim")
        by_phone = await repo.search_contacts(session, "9999")
        none = await repo.search_contacts(session, "zzz")
    assert [c.display_name for c in by_name] == ["Karim Uddin"]
    assert [c.display_name for c in by_phone] == ["Nusrat"]
    assert none == []


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
async def test_contacts_tools_registered(environment):
    from app.tools import registry

    for name in ["contacts_sync", "contacts_list", "contacts_search"]:
        assert name in registry.names()


async def test_contacts_sync_pulls_telegram_and_whatsapp(environment, fake_userbot, fake_whatsapp):
    from app.tools.contact_tools import contacts_sync

    result = await contacts_sync()
    assert result["telegram"]["ok"] is True
    assert result["telegram"]["synced"] == 2, "groups/channels are not people"
    assert result["whatsapp"]["ok"] is True
    assert result["whatsapp"]["synced"] == 2
    assert result["total_synced"] == 4

    async with session_scope() as session:
        rows = await repo.list_contacts(session, limit=100)
    channels = {r.channel for r in rows}
    assert channels == {"telegram", "whatsapp"}


async def test_contacts_sync_degrades_gracefully_when_telegram_unlinked(
    environment, fake_whatsapp, monkeypatch
):
    from app.tools import contact_tools

    monkeypatch.setattr(
        "app.integrations.telegram_user.get_userbot", lambda: FakeUserbot(linked=False)
    )
    result = await contact_tools.contacts_sync()
    assert result["telegram"]["ok"] is False
    assert result["telegram"]["synced"] == 0
    assert result["whatsapp"]["ok"] is True, "WhatsApp half must still succeed"


async def test_contacts_sync_degrades_gracefully_when_whatsapp_unreachable(
    environment, fake_userbot
):
    from app.tools import contact_tools

    set_bridges(whatsapp=FakeWhatsAppBridge(fail=BridgeError("bridge unreachable", temporary=True)))
    try:
        result = await contact_tools.contacts_sync()
    finally:
        set_bridges(None, None)
    assert result["telegram"]["ok"] is True
    assert result["whatsapp"]["ok"] is False
    assert result["whatsapp"]["synced"] == 0


async def test_contacts_sync_survives_whatsapp_endpoint_not_implemented(
    environment, fake_userbot
):
    """/contacts may not exist yet on some bridge builds; must not raise."""
    from app.tools import contact_tools

    class Missing404(FakeWhatsAppBridge):
        async def request(self, method, path, **kwargs):
            raise BridgeError("not found", temporary=False, status=404)

    set_bridges(whatsapp=Missing404())
    try:
        result = await contact_tools.contacts_sync()
    finally:
        set_bridges(None, None)
    assert result["telegram"]["ok"] is True
    assert result["whatsapp"]["ok"] is False


async def test_contacts_list_and_search_tools(environment):
    async with session_scope() as session:
        await repo.upsert_contact(
            session, channel="telegram", external_id="1", display_name="Zara",
            username_or_phone="zara_x",
        )

    from app.tools.contact_tools import contacts_list, contacts_search

    listed = await contacts_list()
    assert listed["count"] == 1 and listed["contacts"][0]["display_name"] == "Zara"

    found = await contacts_search(query="zara")
    assert found["count"] == 1
    missed = await contacts_search(query="nope")
    assert missed["count"] == 0


# --------------------------------------------------------------------------- #
# Telegram commands
# --------------------------------------------------------------------------- #
async def test_members_command_registered(environment, fake_telegram):
    from app.telegram.bot import AgentBot

    async with session_scope() as session:
        await repo.upsert_contact(session, channel="telegram", external_id="1", display_name="Zara")

    bot = AgentBot()
    try:
        commands = {
            filt.callback.commands
            for handler in bot.dp.message.handlers
            for filt in handler.filters
            if hasattr(filt.callback, "commands")
        }
        flat = {c for group in commands for c in group}
        assert "members" in flat and "syncmembers" in flat and "skills" in flat
    finally:
        await bot.stop()


async def test_syncmembers_and_members_handlers_registered(environment):
    from aiogram import Dispatcher

    from app.telegram.member_commands import register_member_handlers

    dp = Dispatcher()

    async def guard(message):
        return True

    register_member_handlers(dp, guard)
    assert len(dp.message.handlers) >= 3
