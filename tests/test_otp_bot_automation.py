"""Deterministic OTP-bot automation - no LLM in the loop.

Uses set_userbot() to inject a fake TelegramUserbot, exactly like
test_tools.py does for other userbot-backed tools.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.automation import otp_bot
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
def _reset_userbot(monkeypatch):
    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)
    yield
    set_userbot(None)


OTP_CHAT_ID = 4242


async def _bind_otp_thread(chat_id: int = OTP_CHAT_ID) -> str:
    """Put `chat_id` into the dedicated OTP thread, as /otpchat would."""
    return await otp_bot.ensure_thread(chat_id)


async def _write_numbers_file(name: str = "numbers.txt") -> str:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text("+880****1111\n+880****2222\n", encoding="utf-8")
    return rel_path(path)


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


# --------------------------------------------------------------------------- #
# Dedicated chat thread - answers "where do I send the files?" structurally
# --------------------------------------------------------------------------- #
async def test_ensure_thread_creates_and_binds_once(environment):
    assert await otp_bot.get_thread_id() == ""

    thread_id = await otp_bot.ensure_thread(OTP_CHAT_ID)
    assert thread_id
    assert await otp_bot.get_thread_id() == thread_id
    assert await otp_bot.is_otp_thread(OTP_CHAT_ID) is True

    # Calling it again reuses the same thread instead of spawning a new one.
    again = await otp_bot.ensure_thread(OTP_CHAT_ID)
    assert again == thread_id


async def test_is_otp_thread_false_in_another_conversation(environment):
    from app.db import repo
    from app.db.base import session_scope

    await otp_bot.ensure_thread(OTP_CHAT_ID)
    assert await otp_bot.is_otp_thread(OTP_CHAT_ID) is True

    # Owner starts an unrelated conversation - the OTP triggers must not fire
    # there, otherwise a plain "start" hijacks a normal chat.
    async with session_scope() as session:
        await repo.reset_session(session, OTP_CHAT_ID)
    assert await otp_bot.is_otp_thread(OTP_CHAT_ID) is False


async def test_thread_is_titled_so_it_is_findable(environment):
    from app.db import repo
    from app.db.base import session_scope

    thread_id = await otp_bot.ensure_thread(OTP_CHAT_ID)
    async with session_scope() as session:
        threads = await repo.list_threads(session, OTP_CHAT_ID, limit=10)
    match = [t for t in threads if t["thread_id"] == thread_id]
    assert match, "the dedicated thread must show up in the thread list"


# --------------------------------------------------------------------------- #
# Autonomous tag decisions - the agent should decide on its own whenever it
# reasonably can, and only ever ask when it genuinely has nothing to go on.
# --------------------------------------------------------------------------- #
async def test_enqueue_infers_tag_from_filename(environment):
    rel = await _write_numbers_file("numbers_BD_batch2.txt")
    entry = await otp_bot.enqueue_file(rel, "numbers_BD_batch2.txt")
    assert entry["tag"] == "BD"
    # No prompt needed - it decided on its own.
    assert await otp_bot.get_awaiting_tag_entry() is None


async def test_enqueue_uses_default_tag_when_configured(environment):
    await otp_bot.save_config({"default_tag": "MyCampaign"})
    rel = await _write_numbers_file("plain.txt")
    entry = await otp_bot.enqueue_file(rel, "plain.txt")
    assert entry["tag"] == "MyCampaign"


async def test_enqueue_reuses_last_tag(environment):
    rel1 = await _write_numbers_file("numbers_IN.txt")
    entry1 = await otp_bot.enqueue_file(rel1, "numbers_IN.txt")
    assert entry1["tag"] == "IN"
    await otp_bot.set_queue_tag(entry1["id"], entry1["tag"])

    rel2 = await _write_numbers_file("plain_batch2.txt")
    entry2 = await otp_bot.enqueue_file(rel2, "plain_batch2.txt")
    assert entry2["tag"] == "IN"  # reused, no prompt


async def test_enqueue_leaves_untagged_when_nothing_to_go_on(environment):
    rel = await _write_numbers_file("plain.txt")
    entry = await otp_bot.enqueue_file(rel, "plain.txt")
    assert entry["tag"] is None


# --------------------------------------------------------------------------- #
# Start/stop lifecycle
# --------------------------------------------------------------------------- #
async def test_start_requires_a_queue(environment):
    result = await otp_bot.start_automation()
    assert result["ok"] is False
    assert "no files queued" in result["error"]


async def test_start_asks_for_missing_tags_before_running(environment):
    rel = await _write_numbers_file("plain.txt")
    await otp_bot.enqueue_file(rel, "plain.txt")  # tag stays None
    set_userbot(FakeUserbot([]))

    result = await otp_bot.start_automation()
    assert result["ok"] is False
    assert "missing_tags" in result and len(result["missing_tags"]) == 1


async def test_start_sends_cleanup_then_reply_based_add_per_file(environment):
    rel = await _write_numbers_file("numbers_BD.txt")
    entry = await otp_bot.enqueue_file(rel, "numbers_BD.txt")
    assert entry["tag"] == "BD"  # inferred, start() should proceed with no prompt

    fake = FakeUserbot([{"text": "\u2705 Added 2 numbers.", "out": False}])
    set_userbot(fake)

    result = await otp_bot.start_automation()
    assert result["ok"] is True
    assert result["files"][0]["tag"] == "BD"

    # cleanup command sent first
    assert fake.sent_messages[0][1] == "/useddelete"
    # file sent, then add-command sent as a REPLY to that file's message
    assert len(fake.sent_files) == 1
    add_reply_to = fake.sent_messages[1][2]
    assert add_reply_to is not None

    # Queue is now empty, file moved to active, monitor turned on.
    assert await otp_bot.get_queue() == []
    active = await otp_bot.get_active_files()
    assert len(active) == 1 and active[0]["name"] == "numbers_BD.txt"
    assert (await otp_bot.get_config())["enabled"] is True


async def test_start_reports_bot_rejection_as_failure_not_success(environment):
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")
    fake = FakeUserbot([{"text": "\u274C No valid phone numbers found.", "out": False}])
    set_userbot(fake)

    result = await otp_bot.start_automation()
    assert result["ok"] is False
    assert "rejected" in result["error"]
    # A rejected file must not silently become "active".
    assert await otp_bot.get_active_files() == []


async def test_stop_disables_monitor_but_keeps_active_files(environment):
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()

    stopped = await otp_bot.stop_automation()
    assert stopped["enabled"] is False
    assert len(await otp_bot.get_active_files()) == 1  # untouched


# --------------------------------------------------------------------------- #
# Deterministic chat triggers (start/stop/tag-answer/resume) - no LLM
# --------------------------------------------------------------------------- #
async def test_start_trigger_word_detection():
    assert otp_bot.is_start_trigger("start")
    assert otp_bot.is_start_trigger("done")
    assert otp_bot.is_start_trigger(" Start ")
    assert not otp_bot.is_start_trigger("start doing something else entirely")


async def test_stop_trigger_word_detection():
    assert otp_bot.is_stop_trigger("stop")
    assert otp_bot.is_stop_trigger("off")
    assert not otp_bot.is_stop_trigger("stop doing that thing")


async def test_handle_start_trigger_asks_one_tag_at_a_time(environment):
    rel1 = await _write_numbers_file("plain1.txt")
    rel2 = await _write_numbers_file("plain2.txt")
    await otp_bot.enqueue_file(rel1, "plain1.txt")
    await otp_bot.enqueue_file(rel2, "plain2.txt")

    reply1 = await otp_bot.handle_start_trigger()
    assert "plain1.txt" in reply1
    assert await otp_bot.get_awaiting_tag_entry() is not None

    fake = FakeUserbot([{"text": "Added.", "out": False}, {"text": "Added.", "out": False}])
    set_userbot(fake)

    reply2 = await otp_bot.handle_tag_answer("BD")
    assert "plain2.txt" in reply2  # asks about the second file next
    assert await otp_bot.get_awaiting_tag_entry() is not None

    reply3 = await otp_bot.handle_tag_answer("IN")
    assert "Shuru hoye geche" in reply3  # both tagged now, actually started
    assert await otp_bot.get_awaiting_tag_entry() is None
    assert len(await otp_bot.get_active_files()) == 2


async def test_handle_start_trigger_resumes_active_files_with_no_new_upload(environment):
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await otp_bot.stop_automation()

    # Owner just says "start" again with nothing new queued - the agent
    # should decide to resume the same file on its own, not ask which one.
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    reply = await otp_bot.handle_start_trigger()
    assert "resumed" in reply.lower() or "Shuru hoye geche" in reply
    assert (await otp_bot.get_config())["enabled"] is True


# --------------------------------------------------------------------------- #
# The periodic refill cycle (scheduler-driven)
# --------------------------------------------------------------------------- #
async def test_cycle_requires_active_files(environment):
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "no active files" in result.error


async def _start_with_one_active_file() -> FakeUserbot:
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")
    fake = FakeUserbot([{"text": "Added.", "out": False}])
    set_userbot(fake)
    await otp_bot.start_automation()
    return fake


async def test_cycle_skips_when_quota_healthy(environment):
    await _start_with_one_active_file()
    fake = FakeUserbot([{"text": "\U0001F4CA Your quota\n\nActive : 3\nLimit  : unlimited", "out": False}])
    set_userbot(fake)

    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.action == "skipped"
    assert result.active_quota == 3
    assert len(fake.sent_messages) == 1  # only the quota check, no cleanup/re-add


async def test_cycle_refills_when_quota_exhausted(environment):
    await _start_with_one_active_file()
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
    assert "Added 2 numbers" in result.add_reply
    assert result.files_processed == ["numbers_BD.txt"]

    # Cleanup command sent before the file.
    assert fake.sent_messages[1][1] == "/useddelete"
    assert len(fake.sent_files) == 1
    add_command_reply_to = fake.sent_messages[2][2]
    assert add_command_reply_to is not None


async def test_cycle_fails_cleanly_when_not_linked(environment):
    await _start_with_one_active_file()
    set_userbot(FailingUserbot())
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "not linked" in result.error


async def test_cycle_reports_unparseable_quota(environment):
    await _start_with_one_active_file()
    fake = FakeUserbot([{"text": "some unexpected reply with no numbers", "out": False}])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "could not parse" in result.error


async def test_last_result_persists(environment):
    await _start_with_one_active_file()
    fake = FakeUserbot([{"text": "Active : 5", "out": False}])
    set_userbot(fake)
    await otp_bot.run_cycle(await otp_bot.get_config())

    last = await otp_bot.get_last_result()
    assert last is not None
    assert last["active_quota"] == 5
