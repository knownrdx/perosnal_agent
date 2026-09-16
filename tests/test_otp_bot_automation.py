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
    """Records every call and returns scripted replies.

    Replies carry incrementing ids because the automation waits for a bot
    message NEWER than the last one it saw. The script advances on SEND, not
    on read: the real code reads once before sending (to anchor on the last
    message) and then polls until something newer appears, so advancing on
    read would consume the script out of order.
    """

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sent_messages: list[tuple[str, str, int | None]] = []
        self.sent_files: list[tuple[str, str, int | None]] = []
        self._next_message_id = 1000
        self._reply_id = 0
        self._current: dict[str, Any] | None = None

    def _advance(self) -> None:
        """Move to the next scripted reply, or re-issue the last one under a
        fresh id when the script runs out - the automation waits for a NEWER
        message, so a fake that stops producing ids would hang instead of
        simply running out of interesting things to say.
        """
        self._reply_id += 1
        if self.replies:
            nxt = dict(self.replies.pop(0))
        elif self._current is not None:
            nxt = dict(self._current)
        else:
            return
        nxt["id"] = self._reply_id
        self._current = nxt

    async def send_message(self, target: str, text: str, reply_to: int | None = None) -> dict[str, Any]:
        self.sent_messages.append((target, text, reply_to))
        self._next_message_id += 1
        self._advance()
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        self.sent_files.append((target, path, reply_to))
        self._next_message_id += 1
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def read_messages(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        if self._current is None:
            return []
        return [self._current]


class FailingUserbot:
    """An unlinked account: every call raises, including the anchor read."""

    async def send_message(self, *args, **kwargs):
        raise UserbotError("Telegram account is not linked. Use /tglogin first.")

    async def send_file(self, *args, **kwargs):
        raise UserbotError("Telegram account is not linked. Use /tglogin first.")

    async def read_messages(self, *args, **kwargs):
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


async def _write_numbers_file(name: str = "numbers.txt", prefix: str = "+880") -> str:
    """One country's worth of numbers, unless a test asks for something else."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(f"{prefix}1711111111\n{prefix}1722222222\n", encoding="utf-8")
    return rel_path(path)


async def _write_mixed_country_file(name: str = "mixed.txt") -> str:
    """Bangladesh + Dominican Republic in one file, as real exports do."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(
        "\n".join([
            "+8801711111111",
            "+8801722222222",
            "+8801733333333",
            "+18091234567",
            "+18491234567",
        ]) + "\n",
        encoding="utf-8",
    )
    return rel_path(path)


async def _enqueue_single(rel: str, name: str) -> dict[str, Any]:
    """Queue a single-country file and hand back its one entry."""
    analysis = await otp_bot.enqueue_file(rel, name)
    return analysis["entries"][0]


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


async def test_sleep_helper_actually_sleeps_and_does_not_recurse():
    """Regression: _sleep once called itself, so every real run died with
    "maximum recursion depth exceeded". The autouse fixture monkeypatches
    _sleep away, so this test reloads a clean copy of the module to get the
    REAL one - otherwise the whole suite can pass while production is
    completely broken.
    """
    import importlib
    import time

    pristine = importlib.reload(importlib.import_module("app.automation.otp_bot"))
    started = time.monotonic()
    await pristine._sleep(0.02)
    assert time.monotonic() - started >= 0.01


async def test_start_reports_a_crash_as_a_crash_not_a_bot_rejection(environment, monkeypatch):
    """A genuine exception must not be mislabelled as "the bot rejected it"."""
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")

    class ExplodingUserbot:
        async def send_message(self, *args, **kwargs):
            raise ValueError("something genuinely broke")

    set_userbot(ExplodingUserbot())
    result = await otp_bot.start_automation()

    assert result["ok"] is False
    assert "unexpected error (ValueError)" in result["error"]
    assert "rejected" not in result["error"]


# --------------------------------------------------------------------------- #
# Multi-country files: one upload can hold several countries, and the target
# bot keeps stock per country, so each must be added under its own tag.
# --------------------------------------------------------------------------- #
async def test_mixed_country_file_is_split_into_one_entry_per_country(environment):
    rel = await _write_mixed_country_file("export.txt")
    analysis = await otp_bot.enqueue_file(rel, "export.txt")

    assert analysis["countries"] == {"Bangladesh": 3, "Dominican Republic": 2}
    queue = await otp_bot.get_queue()
    assert len(queue) == 2

    by_country = {e["country"]: e for e in queue}
    assert by_country["Bangladesh"]["count"] == 3
    assert by_country["Dominican Republic"]["count"] == 2
    # Both entries came from the same upload.
    assert len({e["batch_id"] for e in queue}) == 1


async def test_split_files_contain_only_their_own_country(environment):
    from app.security import safe_path

    rel = await _write_mixed_country_file("export.txt")
    await otp_bot.enqueue_file(rel, "export.txt")

    for entry in await otp_bot.get_queue():
        content = safe_path(entry["path"], must_exist=True).read_text(encoding="utf-8")
        numbers = [n for n in content.splitlines() if n.strip()]
        assert len(numbers) == entry["count"]
        # Every number in the split file really is that country's.
        from app.automation import phone_countries

        assert {phone_countries.country_of(n) for n in numbers} == {entry["country"]}


async def test_single_country_file_is_not_split(environment):
    rel = await _write_numbers_file("bd_only.txt", prefix="+880")
    analysis = await otp_bot.enqueue_file(rel, "bd_only.txt")

    assert list(analysis["countries"]) == ["Bangladesh"]
    queue = await otp_bot.get_queue()
    assert len(queue) == 1
    # No pointless copy of a file that needed no splitting.
    assert queue[0]["path"] == rel


async def test_tag_is_remembered_per_country_not_globally(environment):
    """A Nigeria answer must not silently become Bangladesh's tag."""
    rel_ng = await _write_numbers_file("ng.txt", prefix="+234")
    entry_ng = await _enqueue_single(rel_ng, "ng.txt")
    await otp_bot.set_queue_tag(entry_ng["id"], "WhatsApp")

    rel_bd = await _write_numbers_file("bd.txt", prefix="+880")
    entry_bd = await _enqueue_single(rel_bd, "bd.txt")
    # Bangladesh has never been tagged, so it must not inherit Nigeria's.
    assert entry_bd["tag"] != "WhatsApp" or entry_bd["country"] == "Nigeria"

    await otp_bot.set_queue_tag(entry_bd["id"], "Telegram")

    # A second Bangladesh file now answers itself.
    rel_bd2 = await _write_numbers_file("bd2.txt", prefix="+880")
    entry_bd2 = await _enqueue_single(rel_bd2, "bd2.txt")
    assert entry_bd2["country"] == "Bangladesh"
    assert entry_bd2["tag"] == "Telegram"

    # And so does a second Nigeria file, with its own answer.
    rel_ng2 = await _write_numbers_file("ng2.txt", prefix="+234")
    entry_ng2 = await _enqueue_single(rel_ng2, "ng2.txt")
    assert entry_ng2["country"] == "Nigeria"
    assert entry_ng2["tag"] == "WhatsApp"


async def test_tag_question_names_the_country_and_count(environment):
    rel = await _write_mixed_country_file("export.txt")
    await otp_bot.enqueue_file(rel, "export.txt")

    question = await otp_bot.handle_start_trigger()
    # The owner must be able to see WHAT they are naming a service for.
    assert "Bangladesh" in question or "Dominican Republic" in question
    assert "service" in question.lower() or "tag" in question.lower()


async def test_add_command_can_reference_the_country(environment):
    """The add template exposes {country}, for bots whose command needs it."""
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = await _enqueue_single(rel, "bd.txt")
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    await otp_bot.save_config({"add_command_template": "/fan -t {tag} -c {country}"})

    fake = FakeUserbot([{"text": "Added.", "out": False}])
    set_userbot(fake)
    result = await otp_bot.start_automation()

    assert result["ok"] is True
    add_commands = [t for _, t, reply_to in fake.sent_messages if reply_to is not None]
    assert add_commands and "Bangladesh" in add_commands[0]


# --------------------------------------------------------------------------- #
# /frcd force-delete: wipe a country's stock before re-adding it
# --------------------------------------------------------------------------- #
async def test_force_delete_is_skipped_unless_configured(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = await _enqueue_single(rel, "bd.txt")
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")

    fake = FakeUserbot([{"text": "Added.", "out": False}])
    set_userbot(fake)
    await otp_bot.start_automation()

    sent = [t for _, t, _ in fake.sent_messages]
    assert not any(t.startswith("/frcd") for t in sent)


async def test_force_delete_runs_per_country_before_the_add(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = await _enqueue_single(rel, "bd.txt")
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    await otp_bot.save_config({
        "force_delete_before_add": True,
        "force_delete_uid": "789019025",
    })

    fake = FakeUserbot([{"text": "Deleted.", "out": False}, {"text": "Added.", "out": False}])
    set_userbot(fake)
    result = await otp_bot.start_automation()

    assert result["ok"] is True
    sent = [t for _, t, _ in fake.sent_messages]
    frcd = [t for t in sent if t.startswith("/frcd")]
    assert frcd == ["/frcd Bangladesh 789019025"]
    # It must happen BEFORE the add command for that file.
    add_index = next(i for i, t in enumerate(sent) if t.startswith("/fan"))
    assert sent.index(frcd[0]) < add_index


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
    # A nameless "New chat" entry would defeat the whole point - the owner
    # has to be able to spot this thread in the list without guessing.
    assert match[0]["title"].startswith(otp_bot.THREAD_TITLE[:20])
    assert match[0]["title"] != "New chat"


async def test_ensure_thread_backfills_a_missing_title(environment):
    """A thread bound before titling existed must not stay a nameless entry."""
    from app.db import repo
    from app.db.base import session_scope

    # Simulate the pre-fix state: a bound thread with no user message in it.
    async with session_scope() as session:
        await repo.ensure_session(session, OTP_CHAT_ID)
        untitled = await repo.reset_session(session, OTP_CHAT_ID)
    await otp_bot.save_config({"thread_id": untitled})

    async with session_scope() as session:
        before = await repo.list_threads(session, OTP_CHAT_ID, limit=10)
    assert [t for t in before if t["thread_id"] == untitled][0]["title"] == "New chat"

    reused = await otp_bot.ensure_thread(OTP_CHAT_ID)
    assert reused == untitled, "must reuse, not replace, the bound thread"

    async with session_scope() as session:
        after = await repo.list_threads(session, OTP_CHAT_ID, limit=10)
    assert [t for t in after if t["thread_id"] == untitled][0]["title"] != "New chat"


# --------------------------------------------------------------------------- #
# Autonomous tag decisions - the agent should decide on its own whenever it
# reasonably can, and only ever ask when it genuinely has nothing to go on.
# --------------------------------------------------------------------------- #
async def test_enqueue_infers_tag_from_filename(environment):
    rel = await _write_numbers_file("numbers_BD_batch2.txt")
    entry = await _enqueue_single(rel, "numbers_BD_batch2.txt")
    assert entry["tag"] == "BD"
    # No prompt needed - it decided on its own.
    assert await otp_bot.get_awaiting_tag_entry() is None


async def test_enqueue_uses_default_tag_when_configured(environment):
    await otp_bot.save_config({"default_tag": "MyCampaign"})
    rel = await _write_numbers_file("plain.txt")
    entry = await _enqueue_single(rel, "plain.txt")
    assert entry["tag"] == "MyCampaign"


async def test_enqueue_reuses_last_tag(environment):
    rel1 = await _write_numbers_file("numbers_IN.txt")
    entry1 = await _enqueue_single(rel1, "numbers_IN.txt")
    assert entry1["tag"] == "IN"
    await otp_bot.set_queue_tag(entry1["id"], entry1["tag"])

    rel2 = await _write_numbers_file("plain_batch2.txt")
    entry2 = await _enqueue_single(rel2, "plain_batch2.txt")
    assert entry2["tag"] == "IN"  # reused, no prompt


async def test_enqueue_leaves_untagged_when_nothing_to_go_on(environment):
    rel = await _write_numbers_file("plain.txt")
    entry = await _enqueue_single(rel, "plain.txt")
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
    entry = await _enqueue_single(rel, "numbers_BD.txt")
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


async def test_add_waits_for_a_new_reply_not_the_previous_one(environment, monkeypatch):
    """Regression: a slow add returned the PREVIOUS bot message as its result.

    Live, a large file took longer than the old fixed 3s sleep, so the add
    was reported with the /useddelete reply ("Deleted 4,796 used numbers")
    instead of its own confirmation.
    """
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")

    class SlowBot:
        """Answers the add only after several polls, as a busy bot would."""

        def __init__(self) -> None:
            self.polls_until_answer = 3
            self.current = {"text": "\U0001F5D1 Deleted 4,796 used numbers.", "id": 10, "out": False}

        async def send_message(self, target, text, reply_to=None):
            return {"sent": True, "message_id": 500}

        async def send_file(self, target, path, caption="", reply_to=None):
            return {"sent": True, "message_id": 501}

        async def read_messages(self, target, limit=20):
            if self.polls_until_answer > 0:
                self.polls_until_answer -= 1
            elif self.current["id"] == 10:
                self.current = {"text": "\u26A1 53412 added", "id": 11, "out": False}
            return [self.current]

    set_userbot(SlowBot())
    result = await otp_bot.start_automation()

    assert result["ok"] is True
    reply = result["files"][0]["reply"]
    assert "53412 added" in reply
    assert "Deleted" not in reply, "must not report the previous message as this add's result"


async def test_add_fails_when_the_bot_never_answers(environment, monkeypatch):
    rel = await _write_numbers_file("numbers_BD.txt")
    await otp_bot.enqueue_file(rel, "numbers_BD.txt")

    class SilentBot:
        async def send_message(self, target, text, reply_to=None):
            return {"sent": True, "message_id": 500}

        async def send_file(self, target, path, caption="", reply_to=None):
            return {"sent": True, "message_id": 501}

        async def read_messages(self, target, limit=20):
            # Never produces anything newer than the anchor.
            return [{"text": "an older message", "id": 10, "out": False}]

    set_userbot(SilentBot())
    result = await otp_bot.start_automation()

    assert result["ok"] is False
    assert "no reply" in result["error"]
    assert await otp_bot.get_active_files() == []


async def test_last_bot_message_picks_the_newest_not_the_oldest(environment):
    """Regression: read_messages returns NEWEST-first (both Telethon's
    iter_messages and Pyrogram's get_chat_history do), but the code iterated
    in reverse and so returned the oldest message in the window. Live, that
    left a quota check reporting "no reply" while "Active : 50047" was
    already sitting in the chat.
    """

    class HistoryBot:
        async def read_messages(self, target, limit=20):
            return [
                {"id": 42135, "text": "Active : 50047", "out": False},
                {"id": 42134, "text": "/myquota", "out": True},
                {"id": 42133, "text": "Active : 51484", "out": False},
                {"id": 42131, "text": "\u26A1 Fast Add Complete!", "out": False},
            ]

    set_userbot(HistoryBot())
    latest = await otp_bot._last_bot_message("@PBDxbot")

    assert latest is not None
    assert latest["id"] == 42135
    assert "50047" in latest["text"]


async def test_quota_parser_handles_the_real_reply_formats():
    """Exact shapes seen from the live bot, including grouped thousands."""
    assert otp_bot._parse_quota("\U0001F4CA Your quota\n\nActive : 0\nLimit  : unlimited") == 0
    assert otp_bot._parse_quota("Active : 4,796") == 4796
    assert otp_bot._parse_quota("active-12") == 12
    assert otp_bot._parse_quota("\u26A1 Fast Add Complete! 53412 added") is None


async def test_cycle_skips_the_bots_own_add_notice_when_reading_quota(environment):
    """Regression: a stray "Fast Add Complete!" was taken as the quota reply.

    It is newer than the anchor but is not an answer to /myquota, so the
    cycle reported "could not parse an 'Active' count" while the real reply
    was still on its way.
    """
    await _start_with_one_active_file()

    class ChattyBot:
        """Posts an unrelated notice first, then the actual quota."""

        def __init__(self) -> None:
            self.reads = 0
            self.messages = [
                {"text": "\u26A1 Fast Add Complete! 53412 added", "id": 20, "out": False},
                {"text": "\u26A1 Fast Add Complete! 53412 added", "id": 20, "out": False},
                {"text": "\U0001F4CA Your quota\n\nActive : 7", "id": 21, "out": False},
            ]

        async def send_message(self, target, text, reply_to=None):
            return {"sent": True, "message_id": 900}

        async def send_file(self, target, path, caption="", reply_to=None):
            return {"sent": True, "message_id": 901}

        async def read_messages(self, target, limit=20):
            index = min(self.reads, len(self.messages) - 1)
            self.reads += 1
            return [self.messages[index]]

    set_userbot(ChattyBot())
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.active_quota == 7
    assert result.action == "skipped"


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


async def test_cycle_reports_no_quota_reply_when_the_bot_only_talks_nonsense(environment):
    """Unparseable chatter is treated as "not the answer yet", not as the
    answer - so the cycle waits out its window and reports no reply rather
    than acting on a number it never actually read.
    """
    await _start_with_one_active_file()
    fake = FakeUserbot([{"text": "some unexpected reply with no numbers", "out": False}])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())
    assert result.ok is False
    assert "no reply" in result.error
    assert result.active_quota is None


async def test_last_result_persists(environment):
    await _start_with_one_active_file()
    fake = FakeUserbot([{"text": "Active : 5", "out": False}])
    set_userbot(fake)
    await otp_bot.run_cycle(await otp_bot.get_config())

    last = await otp_bot.get_last_result()
    assert last is not None
    assert last["active_quota"] == 5
