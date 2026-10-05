"""A revoked Telegram session must not stay cached until the next restart.

Telethon keeps the socket open after Telegram revokes the authorisation
(owner pressed "terminate session", or the account was logged out
elsewhere), so ``is_connected()`` stays true and the dead client used to be
handed out on every call. Auth errors now drop it; network blips do not.
"""

from __future__ import annotations

import pytest

from app.integrations.telegram_user import (
    API_HASH_KEY,
    API_ID_KEY,
    BACKEND_PYROGRAM,
    SESSION_KEY,
    TELEGRAM_USER_BACKEND_KEY,
    TelegramUserbot,
    UserbotError,
    set_userbot,
)
from app.security.vault import get_vault, reset_vault

telethon_errors = pytest.importorskip("telethon.errors")


class FakeMe:
    id = 789019025
    username = "owner"
    first_name = "Arif"
    last_name = ""
    phone = "8801712345678"


class FakeClient:
    """A Telethon-shaped client whose next action raises ``fail_with``."""

    def __init__(self, *, authorized: bool = True) -> None:
        self.connected = False
        self.authorized = authorized
        self.fail_with: BaseException | None = None
        self.sent: list[str] = []

    async def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    async def disconnect(self) -> None:
        self.connected = False

    async def is_user_authorized(self) -> bool:
        return self.authorized

    async def get_me(self):
        if self.fail_with is not None:
            raise self.fail_with
        return FakeMe()

    async def send_message(self, entity, text, reply_to=None):
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append(text)

        class Msg:
            id = 1

        return Msg()


@pytest.fixture
async def linked(environment, monkeypatch):
    """A userbot with a stored Telethon session and a factory of fake clients."""
    reset_vault()
    vault = get_vault()
    await vault.set(SESSION_KEY, "S" * 60)
    await vault.set(API_ID_KEY, "12345")
    await vault.set(API_HASH_KEY, "hash")

    built: list[FakeClient] = []
    next_authorized = {"value": True}

    def factory(*_a, **_k):
        client = FakeClient(authorized=next_authorized["value"])
        built.append(client)
        return client

    monkeypatch.setattr("app.integrations.telegram_user.TelegramClient", factory)
    monkeypatch.setattr("app.integrations.telegram_user.StringSession", lambda s="": s)
    yield TelegramUserbot(), built, next_authorized
    reset_vault()
    set_userbot(None)


@pytest.mark.parametrize(
    "error",
    [
        telethon_errors.AuthKeyUnregisteredError(request=None),
        telethon_errors.SessionRevokedError(request=None),
        telethon_errors.UserDeactivatedError(request=None),
        telethon_errors.AuthKeyDuplicatedError(request=None),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_revoked_session_drops_the_cached_client(linked, error):
    userbot, built, next_authorized = linked
    await userbot.send_message("me", "first")
    assert len(built) == 1

    built[0].fail_with = error
    with pytest.raises(UserbotError, match="/tglogin"):
        await userbot.send_message("me", "second")

    assert built[0].connected is False, "the dead client must be disconnected"

    # Telegram now refuses the stored session too: the next call rebuilds the
    # client rather than reusing the dead one, and says how to recover.
    next_authorized["value"] = False
    with pytest.raises(UserbotError, match="/tglogin"):
        await userbot.send_message("me", "third")
    assert len(built) == 2


async def test_relinked_session_works_without_restart(linked):
    userbot, built, _ = linked
    await userbot.send_message("me", "first")
    built[0].fail_with = telethon_errors.AuthKeyUnregisteredError(request=None)
    with pytest.raises(UserbotError):
        await userbot.send_message("me", "second")

    # The owner runs /tglogin again (a fresh authorised session in the vault).
    await userbot.send_message("me", "third")
    assert len(built) == 2
    assert built[1].sent == ["third"]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("network unreachable"),
        TimeoutError("timed out"),
        telethon_errors.FloodWaitError(request=None, capture=5),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_transient_errors_keep_the_client(linked, error):
    userbot, built, _ = linked
    await userbot.send_message("me", "first")

    built[0].fail_with = error
    with pytest.raises(type(error)):
        await userbot.send_message("me", "second")

    built[0].fail_with = None
    await userbot.send_message("me", "third")
    assert len(built) == 1, "a network blip must not force a reconnect"
    assert built[0].sent == ["first", "third"]


async def test_status_reports_revocation_and_drops_client(linked):
    userbot, built, _ = linked
    assert (await userbot.status())["linked"] is True

    built[0].fail_with = telethon_errors.AuthKeyUnregisteredError(request=None)
    status = await userbot.status()
    assert status["linked"] is False
    assert "/tglogin" in status["error"]
    assert userbot._client is None


async def test_revoked_session_is_a_permanent_tool_failure(linked):
    """Agent tools must not retry a dead session as if it were a network blip."""
    from app.tools import registry
    from app.tools.base import FailureKind, ToolContext

    userbot, built, _ = linked
    set_userbot(userbot)
    await userbot.send_message("me", "first")
    built[0].fail_with = telethon_errors.AuthKeyUnregisteredError(request=None)

    result = await registry.get("tg_send_message").run(
        {"to": "me", "text": "hi"}, ToolContext()
    )
    assert not result.ok
    assert result.kind is FailureKind.PERMANENT
    assert "/tglogin" in result.error


async def test_pyrogram_revoked_session_drops_the_cached_client(environment, monkeypatch):
    pyrogram_errors = pytest.importorskip("pyrogram.errors")

    class FakePyro:
        def __init__(self, **_kw) -> None:
            self._connected = False
            self.fail_with: BaseException | None = None

        @property
        def is_connected(self) -> bool:
            return self._connected

        async def start(self) -> None:
            self._connected = True

        async def stop(self, block: bool = True) -> None:
            self._connected = False

        async def get_me(self):
            return FakeMe()

        async def send_message(self, chat_id, text, reply_to_message_id=None):
            if self.fail_with is not None:
                raise self.fail_with

            class Msg:
                id = 2

            return Msg()

    built: list[FakePyro] = []

    def factory(**kw):
        client = FakePyro(**kw)
        built.append(client)
        return client

    monkeypatch.setattr("app.integrations.telegram_user.PyrogramClient", factory)
    reset_vault()
    vault = get_vault()
    await vault.set(SESSION_KEY, "P" * 60)
    await vault.set(API_ID_KEY, "12345")
    await vault.set(API_HASH_KEY, "hash")
    await vault.set(TELEGRAM_USER_BACKEND_KEY, BACKEND_PYROGRAM)

    try:
        userbot = TelegramUserbot()
        await userbot.send_message("me", "first")
        built[0].fail_with = pyrogram_errors.AuthKeyUnregistered()
        with pytest.raises(UserbotError, match="/tglogin"):
            await userbot.send_message("me", "second")
        assert built[0].is_connected is False

        await userbot.send_message("me", "third")
        assert len(built) == 2
    finally:
        reset_vault()
