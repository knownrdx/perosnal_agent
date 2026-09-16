"""Telegram user account (userbot): login flow, session storage, tools."""

from __future__ import annotations

import pytest

from app.integrations.telegram_user import (
    API_HASH_KEY,
    API_ID_KEY,
    SESSION_KEY,
    TELEGRAM_USER_BACKEND_KEY,
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

    async def send_message(self, entity, text, reply_to=None):
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


async def test_login_works_under_production_logging_level(vault, monkeypatch, caplog):
    """Regression guard: /tglogin -> /tgcode -> /tg2fa must complete a real login.

    app/security/vault.py used to log credential_set/credential_deleted with
    extra={"name": ...}, and "name" is a reserved LogRecord attribute -
    stdlib logging raises KeyError for that at INFO level, which is what
    setup_logging("INFO") uses in production. That crash landed inside
    _finish_login() AFTER the DB write but before the in-memory client/backend
    were set, so the owner saw success messages in the bot but /tgstatus kept
    saying "not linked". Reproduce at the real production log level so this
    exact class of bug (a reserved LogRecord key in ANY vault.set/delete call)
    cannot silently regress.
    """
    import logging

    caplog.set_level(logging.INFO)

    fake = FakeTelethonClient(needs_2fa=True)
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", lambda *a, **k: fake
    )

    userbot = TelegramUserbot()
    await userbot.start_login(1234567, "hash", "+880****5678")
    with pytest.raises(TwoFactorRequired):
        await userbot.submit_code("12345")
    result = await userbot.submit_password("my-2fa-password")

    assert result["linked"] is True
    # The real regression: login "succeeded" but the vault write silently
    # never landed / status still reported unlinked.
    assert vault.get(SESSION_KEY) == "SESSION-STRING-ABC"
    status = await userbot.status()
    assert status["linked"] is True


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


# --------------------------------------------------------------------------- #
# Pyrogram backend
# --------------------------------------------------------------------------- #
def _pyrogram_available() -> bool:
    from app.integrations.telegram_user import PYROGRAM_AVAILABLE

    return PYROGRAM_AVAILABLE


pyrogram_only = pytest.mark.skipif(
    not _pyrogram_available(), reason="pyrogram is not installed in this environment"
)


class FakePyroUser:
    id = 900123
    username = "owner_pg"
    first_name = "Arif"
    last_name = ""
    phone_number = "8801799999999"


class FakePyroChat:
    def __init__(self, chat_id, title="", username="", first_name="", last_name="", chat_type="private"):
        self.id = chat_id
        self.title = title
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.type = chat_type


class FakePyroDialog:
    def __init__(self, chat, unread=0):
        self.chat = chat
        self.unread_messages_count = unread


class FakePyroMessage:
    def __init__(self, msg_id, text="", chat=None, from_user=None, date=None, outgoing=False):
        self.id = msg_id
        self.text = text
        self.caption = ""
        self.chat = chat
        self.from_user = from_user
        self.date = date
        self.outgoing = outgoing


class FakePyrogramClient:
    """Stands in for pyrogram.Client."""

    def __init__(self, name=None, session_string=None, api_id=None, api_hash=None, in_memory=None, **kwargs):
        self.name = name
        self.session_string = session_string or "PYRO-SESSION-STRING-XYZ"
        self.api_id = api_id
        self.api_hash = api_hash
        self.in_memory = in_memory
        self._is_connected = False
        self.sent: list[tuple] = []
        self.sent_documents: list[tuple] = []
        self.logged_out = False
        self.reject_start = False

    @property
    def is_connected(self):
        return self._is_connected

    async def start(self):
        if self.reject_start:
            raise RuntimeError("AUTH_KEY_UNREGISTERED")
        self._is_connected = True

    async def stop(self, block: bool = True):
        self._is_connected = False

    async def get_me(self):
        return FakePyroUser()

    async def send_message(self, chat_id, text, reply_to_message_id=None):
        self.sent.append((chat_id, text))

        class Msg:
            id = 5151

        return Msg()

    async def send_document(self, chat_id, document, caption="", reply_to_message_id=None):
        self.sent_documents.append((chat_id, document, caption))

        class Msg:
            id = 6161

        return Msg()

    async def get_chat_history(self, chat_id, limit=0):
        for msg in [
            FakePyroMessage(1, text="hello", from_user=FakePyroUser(), outgoing=False),
            FakePyroMessage(2, text="world", from_user=FakePyroUser(), outgoing=True),
        ][:limit or None]:
            yield msg

    async def get_dialogs(self, limit=0):
        dialogs = [
            FakePyroDialog(FakePyroChat(1, title="Group A", chat_type="group"), unread=3),
            FakePyroDialog(FakePyroChat(2, username="bob", first_name="Bob", chat_type="private"), unread=0),
        ]
        for d in dialogs[: limit or None]:
            yield d

    async def search_global(self, query, limit=0):
        yield FakePyroMessage(9, text="found it", chat=FakePyroChat(1, title="Group A"))

    async def log_out(self):
        self.logged_out = True
        return True


def _no_telethon_match(*args, **kwargs):
    raise RuntimeError("not a valid telethon string")


async def test_link_string_session_falls_back_to_pyrogram(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )

    userbot = TelegramUserbot()
    pasted_string = "PYRO-SESSION-STRING-XYZ" + "0" * 30
    result = await userbot.link_string_session(pasted_string, api_id=123, api_hash="hash")
    assert result["linked"] is True
    assert result["method"] == "string_session"
    assert result["user_id"] == FakePyroUser.id
    assert vault.get(TELEGRAM_USER_BACKEND_KEY) == "pyrogram"
    assert vault.get(SESSION_KEY) == pasted_string


async def test_link_string_session_raises_when_both_backends_reject(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )

    class RejectingPyrogram(FakePyrogramClient):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.reject_start = True

    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: RejectingPyrogram(**kw)
    )

    userbot = TelegramUserbot()
    with pytest.raises(UserbotError, match="not recognised"):
        await userbot.link_string_session(
            "NOT-A-VALID-STRING" + "0" * 30, api_id=123, api_hash="hash"
        )


async def test_pyrogram_backend_persists_across_restart(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )

    userbot = TelegramUserbot()
    await userbot.link_string_session(
        "PYRO-SESSION-STRING-XYZ" + "0" * 30, api_id=123, api_hash="hash"
    )

    # A brand-new instance (simulating a process restart) should read the
    # backend from the vault and reconnect with Pyrogram, not Telethon.
    fresh_fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fresh_fake
    )
    fresh = TelegramUserbot()
    assert fresh._stored_backend() == "pyrogram"

    client = await fresh.client()
    assert client is fresh_fake
    assert fresh._backend == "pyrogram"


async def test_pyrogram_send_message(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )
    userbot = TelegramUserbot()
    await userbot.link_string_session(
        "PYRO-SESSION-STRING-XYZ" + "0" * 30, api_id=123, api_hash="hash"
    )

    result = await userbot.send_message("me", "hi from pyrogram")
    assert result == {"sent": True, "message_id": 5151, "to": "me"}
    assert fake.sent == [("me", "hi from pyrogram")]


async def test_pyrogram_read_messages(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )
    userbot = TelegramUserbot()
    await userbot.link_string_session(
        "PYRO-SESSION-STRING-XYZ" + "0" * 30, api_id=123, api_hash="hash"
    )

    messages = await userbot.read_messages("me", limit=2)
    assert len(messages) == 2
    assert messages[0]["id"] == 1
    assert messages[0]["text"] == "hello"
    assert messages[0]["out"] is False
    assert messages[1]["out"] is True
    assert all("sender_id" in m for m in messages)


async def test_pyrogram_list_dialogs(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )
    userbot = TelegramUserbot()
    await userbot.link_string_session(
        "PYRO-SESSION-STRING-XYZ" + "0" * 30, api_id=123, api_hash="hash"
    )

    dialogs = await userbot.list_dialogs(limit=5)
    assert len(dialogs) == 2
    assert dialogs[0]["id"] == 1
    assert dialogs[0]["name"] == "Group A"
    assert dialogs[0]["is_group"] is True
    assert dialogs[0]["unread"] == 3
    assert dialogs[1]["username"] == "bob"
    assert dialogs[1]["is_user"] is True


async def test_pyrogram_botfather_raises_userbot_error(vault, monkeypatch):
    monkeypatch.setattr(
        "app.integrations.telegram_user.TelegramClient", _no_telethon_match
    )
    fake = FakePyrogramClient()
    monkeypatch.setattr(
        "app.integrations.telegram_user.PyrogramClient", lambda **kw: fake
    )
    userbot = TelegramUserbot()
    await userbot.link_string_session(
        "PYRO-SESSION-STRING-XYZ" + "0" * 30, api_id=123, api_hash="hash"
    )

    with pytest.raises(UserbotError, match="Telethon-linked session"):
        await userbot.botfather("/mybots")
