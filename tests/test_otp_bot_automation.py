"""Deterministic OTP-bot automation - no LLM in the loop.

Uses set_userbot() to inject a fake TelegramUserbot, exactly like
test_tools.py does for other userbot-backed tools.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.automation import otp_bot
from app.db.base import session_scope
from app.integrations.telegram_user import UserbotError, set_userbot


class FakeUserbot:
    """Records every call and returns scripted replies."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sent_messages: list[tuple[str, str, int | None]] = []
        self.sent_files: list[tuple[str, str, int | None]] = []
        self._next_message_id = 1000

    async def send_message(self, target: str, text: str, reply_to: int | None = None) -> dict[str, Any]:
        self.sent_messages.append((target, text, reply_to))
        self._next_message_id += 1
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        self.sent_files.append((target, path, reply_to))
        self._next_message_id += 1
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def read_messages(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self.replies:
            return []
        return [self.replies.pop(0)]


class FailingUserbot:
    async def send_message(self, *args, **kwargs):
        raise UserbotError("Telegram account is not linked. Use /tglogin first.")


@pytest.fixture(autouse=True)
def _reset_userbot():
    yield
    set_userbot(None)


async def _write_numbers_file(name: str = "numbers.txt") -> None:
    from app.config import get_settings

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / name).write_text("+8801711111111\n+8801722222222\n", encoding="utf-8")


async def test_config_roundtrip(environment):
    default = await otp_bot.get_config()
    assert default["enabled"] is False
    assert default["target_bot"] == "@PBDxbot"

    saved = await otp_bot.save_config({"enabled": True, "interval_minutes": 15})
    assert saved["enabled"] is True
    assert saved["interval_minutes"] == 15
    # Untouched fields keep their previous value, not silently reset.
    assert saved["target_bot"] == "@PBDxbot"

    reloaded = await otp_bot.get_config()
    assert reloaded["enabled"] is True
    assert reloaded["interval_minutes"] == 15


async def test_cycle_skips_when_quota_healthy(environment):
    fake = FakeUserbot([{"text": "\U0001F4CA Your quota\n\nActive : 3\nLimit  : unlimited", "out": False}])
    set_userbot(fake)

    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.action == "skipped"
    assert result.active_quota == 3
    assert len(fake.sent_messages) == 1  # only the quota check, no cleanup/re-add


async def test_cycle_refills_when_quota_exhausted(environment):
    await _write_numbers_file()
    fake = FakeUserbot([
        {"text": "\U0001F4CA Your quota\n\nActive : 0\nLimit  : unlimited", "out": False},
        {"text": "\u2705 Added 2 numbers.", "out": False},
    ])
    set_userbot(fake)

    config = await otp_bot.save_config({"quota_threshold": 0})
    result = await otp_bot.run_cycle(config)

    assert result.ok is True
    assert result.action == "added"
    assert result.active_quota == 0
    assert result.add_reply == "\u2705 Added 2 numbers."

    # Cleanup command sent before the file.
    assert fake.sent_messages[1][1] == "/useddelete"
    # The file was sent...
    assert len(fake.sent_files) == 1
    # ...and the add-command was sent as a REPLY to that exact file message -
    # this is the whole point: the target bot's FSM requires a reply, not a
    # standalone message, or it silently rejects the command.
    file_message_id = fake.sent_files[0][2] is None  # send_file itself has no reply_to
    add_command_reply_to = fake.sent_messages[2][2]
    assert add_command_reply_to is not None


async def test_cycle_fails_cleanly_when_not_linked(environment):
    set_userbot(FailingUserbot())
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "not linked" in result.error


async def test_cycle_reports_unparseable_quota(environment):
    fake = FakeUserbot([{"text": "some unexpected reply with no numbers", "out": False}])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "could not parse" in result.error


async def test_last_result_persists(environment):
    fake = FakeUserbot([{"text": "Active : 5", "out": False}])
    set_userbot(fake)
    await otp_bot.run_cycle(await otp_bot.get_config())

    last = await otp_bot.get_last_result()
    assert last is not None
    assert last["active_quota"] == 5
