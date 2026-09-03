"""Telegram user account (userbot): login flow, session storage, tools."""

from __future__ import annotations

import pytest

from app.integrations.telegram_user import (
    API_HASH_KEY,
    API_ID_KEY,
    SESSION_KEY,
    TelegramUserbot,
    TwoFactorRequired,
    UserbotError,
    set_userbot,
)
from app.security.vault import get_vault, reset_vault


class FakeMe:
    id = 789019025
    username = "owner"
    first_name = "Arif"
    last_name = ""
    phone = "8801712345678"


class FakeSession:
    def __init__(self, value: str = "SESSION-STRING-ABC") -> None:
        self.value = value

    def save(self) -> str:
        return self.value


class FakeTelethonClient:
    """Stands in for telethon.TelegramClient."""

    def __init__(self, *, needs_2fa: bool = False, bad_code: bool = False) -> None:
        self.session = FakeSession()
        self.connected = False
        self.authorized = True
        self.needs_2fa = needs_2fa
        self.bad_code = bad_code
        self.sent: list[tuple[str, str]] = []
        self.logged_out = False

    async def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    async def disconnect(self) -> None:
        self.connected = False

    async def is_user_authorized(self) -> bool:
        return self.authorized

    async def send_code_request(self, phone: str):
        class Sent:
            phone_code_hash = "HASH123"

        return Sent()

    async def sign_in(self, phone=None, code=None, phone_code_hash=None, password=None):
        if password is not None:
            return FakeMe()
        if self.bad_code:
            raise UserbotError("that code is not correct")
        if self.needs_2fa:
            from app.integrations.telegram_user import SessionPasswordNeededError

            raise SessionPasswordNeededError(request=None)
        return FakeMe()

    async def get_me(self):
        return FakeMe()

    async def send_message(self, entity, text):
        self.sent.append((str(entity), text))

        class Msg:
            id = 4242

        return Msg()

    async def log_out(self) -> bool:
        self.logged_out = True
        return True


@pytest.fixture
def vault(environment):
    reset_vault()
    yield get_vault()
    reset_vault()
    set_userbot(None)


def _telethon_available() -> bool:
    from app.integrations.telegram_user import TELETHON_AVAILABLE

    return TELETHON_AVAILABLE


pytestmark = pytest.mark.skipif(
    not _telethon_available(), reason="telethon is not installed in this environment"
)


# --------------------------------------------------------------------------- #
# Login flow
# --------------------------------------------------------------------------- #
async def test_login_stores_encrypted_session(vault, monkeypatch):
    fake = FakeTelethonClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )

    userbot = TelegramUserbot()
    result = await userbot.start_login(1234567, "hash-abc", "8801712345678")
    assert result["code_sent"] is True
    assert result["phone"] == "+8801712345678"

    linked = await userbot.submit_code("12345")
    assert linked["linked"] is True
    assert linked["user_id"] == FakeMe.id

    # Session persisted, and encrypted at rest.
    assert vault.get(SESSION_KEY) == "SESSION-STRING-ABC"
    assert vault.get(API_ID_KEY) == "1234567"
    assert vault.get(API_HASH_KEY) == "hash-abc"

    from app.db import repo
    from app.db.base import session_scope

    async with session_scope() as session:
        rows = await repo.list_credentials(session)
    stored = {r.name: r.value for r in rows}
    assert "SESSION-STRING-ABC" not in stored[SESSION_KEY]


async def test_two_factor_flow(vault, monkeypatch):
    fake = FakeTelethonClient(needs_2fa=True)
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )

    userbot = TelegramUserbot()
    await userbot.start_login(1234567, "hash", "+8801712345678")

    with pytest.raises(TwoFactorRequired):
        await userbot.submit_code("12345")

    result = await userbot.submit_password("my-2fa-password")
    assert result["linked"] is True
    assert vault.get(SESSION_KEY) == "SESSION-STRING-ABC"


async def test_code_without_login_is_rejected(vault):
    with pytest.raises(UserbotError, match="no login in progress"):
        await TelegramUserbot().submit_code("12345")


async def test_2fa_without_pending_is_rejected(vault):
    with pytest.raises(UserbotError, match="no 2FA step"):
        await TelegramUserbot().submit_password("x")


async def test_client_requires_link(vault):
    with pytest.raises(UserbotError, match="not linked"):
        await TelegramUserbot().client()


async def test_status_reports_unlinked(vault):
    status = await TelegramUserbot().status()
    assert status["available"] is True
    assert status["linked"] is False


async def test_status_after_link(vault, monkeypatch):
    fake = FakeTelethonClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )
    userbot = TelegramUserbot()
    await userbot.start_login(1, "h", "+880171")
    await userbot.submit_code("1")

    status = await userbot.status()
    assert status["linked"] is True
    assert status["user_id"] == FakeMe.id


async def test_logout_clears_stored_session(vault, monkeypatch):
    fake = FakeTelethonClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )
    userbot = TelegramUserbot()
    await userbot.start_login(1, "h", "+880171")
    await userbot.submit_code("1")

    assert await userbot.logout() is True
    assert vault.get(SESSION_KEY) == ""
    assert fake.logged_out is True


async def test_session_survives_restart(vault, monkeypatch):
    fake = FakeTelethonClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )
    await (u := TelegramUserbot()).start_login(1, "h", "+880171")
    await u.submit_code("1")

    from app.security.vault import CredentialVault

    fresh = CredentialVault()
    await fresh.load()
    assert fresh.get(SESSION_KEY) == "SESSION-STRING-ABC"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
async def test_tools_registered(environment):
    from app.tools import registry

    for name in ["tg_send_message", "tg_read_messages", "tg_list_chats",
                 "tg_bot_admin", "tg_account_status"]:
        assert name in registry.names()


async def test_bot_admin_is_high_risk(environment):
    from app.security import Permission
    from app.tools import registry

    assert registry.get("tg_bot_admin").permission is Permission.HIGH_RISK


async def test_tool_reports_missing_link_clearly(environment, vault):
    from app.tools import registry
    from app.tools.base import ToolContext

    set_userbot(TelegramUserbot())
    result = await registry.get("tg_send_message").run(
        {"to": "me", "text": "hi"}, ToolContext()
    )
    assert not result.ok
    assert "/tglogin" in result.error


async def test_send_message_through_tool(environment, vault, monkeypatch):
    fake = FakeTelethonClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )
    userbot = TelegramUserbot()
    await userbot.start_login(1, "h", "+880171")
    await userbot.submit_code("1")
    set_userbot(userbot)

    from app.tools import registry
    from app.tools.base import ToolContext

    result = await registry.get("tg_send_message").run(
        {"to": "me", "text": "note to self"}, ToolContext(task_id="t1")
    )
    assert result.ok and result.data["message_id"] == 4242
    assert fake.sent == [("me", "note to self")]


async def test_bot_admin_requires_slash_command(environment, vault):
    from app.tools import registry
    from app.tools.base import ToolContext

    set_userbot(TelegramUserbot())
    result = await registry.get("tg_bot_admin").run(
        {"command": "mybots"}, ToolContext()
    )
    assert not result.ok and "start" in result.error
