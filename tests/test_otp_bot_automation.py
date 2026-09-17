"""Deterministic OTP-bot automation - no LLM in the loop.

Uses set_userbot() to inject a fake TelegramUserbot, exactly like
test_tools.py does for other userbot-backed tools.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
# /st: one command, every country's live stock
# --------------------------------------------------------------------------- #
REAL_ST_REPLY = (
    "\U0001F4CA Bot Statistics\n"
    "\n"
    "\u23F1\uFE0F Uptime: 38m 36s\n"
    "\U0001F465 Users: 43024\n"
    "\U0001F4F1 Numbers:\n"
    "  \u2022 Available: 416036\n"
    "  \u2022 Taken: 1076\n"
    "  \u2022 Used: 11221\n"
    "\U0001F4E8 OTPs (24h): 24411\n"
    "\n"
    "\U0001F30D Country Stock (yours):\n"
    "  \U0001F1E8\U0001F1EB Central African Republic: 90712 (+416 taken)\n"
    "  \U0001F1E7\U0001F1E9 Bangladesh: 0\n"
)


def test_stock_parser_reads_the_real_st_reply():
    stock = otp_bot._parse_country_stock(REAL_ST_REPLY)
    assert stock == {"Central African Republic": 90712, "Bangladesh": 0}


def test_stock_parser_ignores_the_global_numbers_block():
    """"Available: 416036" sits above the country header and is a total, not
    a country - reading it as one would mask an empty country behind a
    healthy-looking global figure.
    """
    stock = otp_bot._parse_country_stock(REAL_ST_REPLY)
    assert "Available" not in stock
    assert "Taken" not in stock
    assert "Users" not in stock


def test_stock_parser_rejects_lookalike_messages():
    """The bot posts add-notices that also contain "Country: number".
    Accepting one as a stock report would act on a stale number.
    """
    assert otp_bot._parse_country_stock(
        "\u26A1 Fast Add Complete!\n\n\U0001F1E8\U0001F1EB Central African Republic: 100000 added"
    ) == {}
    assert otp_bot._parse_country_stock("\U0001F4CA Your quota\n\nActive : 96336") == {}
    assert otp_bot._parse_country_stock("") == {}


def test_stock_lookup_tolerates_name_differences():
    stock = {"Central African Republic": 5}
    assert otp_bot._stock_for(stock, "central african republic") == 5
    assert otp_bot._stock_for(stock, "  Central African Republic ") == 5
    assert otp_bot._stock_for(stock, "Bangladesh") is None


async def test_cycle_refills_only_the_empty_country(environment):
    """The whole point of /st: one check, and only the country that actually
    ran out gets re-added. Previously any empty quota re-added everything.
    """
    rel_caf = await _write_numbers_file("caf.txt", prefix="+236")
    rel_bd = await _write_numbers_file("bd.txt", prefix="+880")
    for rel, name in ((rel_caf, "caf.txt"), (rel_bd, "bd.txt")):
        entry = (await otp_bot.enqueue_file(rel, name))["entries"][0]
        await otp_bot.set_queue_tag(entry["id"], "WhatsApp")

    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    # CAR still has stock; Bangladesh is at zero.
    fake = FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u2705 Added 5 numbers.", "out": False},
    ])
    set_userbot(fake)

    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.country_stock["Central African Republic"] == 90712
    assert result.country_stock["Bangladesh"] == 0
    # Only the empty one was touched.
    assert result.files_processed == ["bd.txt"]


async def test_cycle_skips_when_every_country_still_has_stock(environment):
    rel = await _write_numbers_file("caf.txt", prefix="+236")
    entry = (await otp_bot.enqueue_file(rel, "caf.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert "skipped" in result.action
    assert result.files_processed == []


async def test_a_country_missing_from_the_stock_list_counts_as_empty(environment):
    """Absence means the bot holds none of it - that is exactly when a
    refill is needed, so it must not be read as "unknown, leave it alone".
    """
    rel = await _write_numbers_file("ng.txt", prefix="+234")
    entry = (await otp_bot.enqueue_file(rel, "ng.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    # Nigeria is not in the reply at all.
    fake = FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u2705 Added 5 numbers.", "out": False},
    ])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.files_processed == ["ng.txt"]


async def test_myquota_style_reply_still_works(environment):
    """An owner who kept /myquota configured must not be stranded by the
    switch to /st.
    """
    await _start_with_one_active_file()
    fake = FakeUserbot([
        {"text": "\U0001F4CA Your quota\n\nActive : 0", "out": False},
        {"text": "\u2705 Added 2 numbers.", "out": False},
    ])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.save_config({"quota_command": "/myquota"}))

    assert result.ok is True
    assert result.action.startswith("added")
    assert result.active_quota == 0


def test_added_count_parser_distinguishes_worked_from_contributed():
    """"0 added (20000 dup)" is a SUCCESSFUL command that changed nothing.
    Treating it as success without reading the count is what would leave a
    dead file cycling forever.
    """
    assert otp_bot._parse_added_count("\u2705 53412 added, 46588 duplicates skipped") == 53412
    assert otp_bot._parse_added_count("Central African Republic: 0 added (20000 dup)") == 0
    assert otp_bot._parse_added_count("\u2705 1,204 added") == 1204
    assert otp_bot._parse_added_count("Added successfully") is None
    assert otp_bot._parse_added_count("") is None


async def test_cycle_flags_a_country_whose_file_is_spent(environment):
    """Stock empty AND nothing new added = the owner must send a new file."""
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    fake = FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},           # Bangladesh: 0
        {"text": "\u26A1 Fast Add Complete!\n\nBangladesh: 0 added (5 dup)", "out": False},
    ])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert len(result.exhausted) == 1
    assert result.exhausted[0]["country"] == "Bangladesh"
    assert result.exhausted[0]["name"] == "bd.txt"


async def test_a_file_with_numbers_left_is_not_flagged_as_spent(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    fake = FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u2705 4821 added, 12 duplicates skipped", "out": False},
    ])
    set_userbot(fake)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.ok is True
    assert result.exhausted == []


# --------------------------------------------------------------------------- #
# The owner-facing alert: a country whose stock is gone needs a new file, and
# nothing the automation can do will fix it by itself.
# --------------------------------------------------------------------------- #
class _CapturingNotifier:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, chat_id: int, text: str, *, dedupe_key: str | None = None) -> dict:
        self.sent.append({"chat_id": chat_id, "text": text, "dedupe_key": dedupe_key})
        return {"sent": True}


async def _run_scheduler_once(notifier: _CapturingNotifier) -> None:
    """Drive one OTP pass through the real scheduler method."""
    from datetime import timedelta as _td

    from app.workers.scheduler_worker import SchedulerRunner

    runner = SchedulerRunner(notifier=notifier)
    # First call only arms the timer by design, so pre-arm it into the past.
    runner._next_otp_check = datetime.now(timezone.utc) - _td(minutes=1)
    await runner.maybe_run_otp_automation()


async def test_owner_is_told_which_country_ran_out_and_what_to_do(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    set_userbot(FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u26A1 Fast Add Complete!\n\nBangladesh: 0 added (5 dup)", "out": False},
    ]))

    notifier = _CapturingNotifier()
    await _run_scheduler_once(notifier)

    assert notifier.sent, "the owner must be told - they are the only one who can fix it"
    message = notifier.sent[-1]["text"]
    assert "Bangladesh" in message
    assert "shesh" in message            # says the stock is gone
    assert "file" in message.lower()     # says a new file is what is needed


async def test_the_same_dead_country_does_not_re_alert_every_cycle(environment):
    """Re-alerting on a timer trains the owner to ignore the alert, which is
    worse than not sending it. Keyed on the country, so a DIFFERENT country
    running dry still gets through.
    """
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()

    notifier = _CapturingNotifier()
    for _ in range(2):
        await _make_everything_due()
        set_userbot(FakeUserbot([
            {"text": REAL_ST_REPLY, "out": False},
            {"text": "\u26A1 Fast Add Complete!\n\nBangladesh: 0 added (5 dup)", "out": False},
        ]))
        await _run_scheduler_once(notifier)

    keys = [n["dedupe_key"] for n in notifier.sent]
    # Same key both times, so the notifier's own dedupe suppresses the repeat.
    assert len(set(keys)) == 1
    assert "Bangladesh" in keys[0]


async def test_a_healthy_refill_does_not_raise_the_stock_alarm(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    await _make_everything_due()

    set_userbot(FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u2705 4821 added, 12 duplicates skipped", "out": False},
    ]))

    notifier = _CapturingNotifier()
    await _run_scheduler_once(notifier)

    joined = " ".join(n["text"] for n in notifier.sent)
    assert "stock shesh" not in joined


# --------------------------------------------------------------------------- #
# The country list comes from the BOT, not from our prefix table
# --------------------------------------------------------------------------- #
async def test_countries_are_learned_from_the_bots_own_reply(environment):
    new = await otp_bot.learn_countries({"Central African Republic": 90712, "Bangladesh": 0})
    assert sorted(new) == ["Bangladesh", "Central African Republic"]

    known = (await otp_bot.get_known_countries())["names"]
    assert known["bangladesh"]["name"] == "Bangladesh"
    assert known["bangladesh"]["last_stock"] == 0


async def test_only_genuinely_new_countries_are_reported_as_new(environment):
    await otp_bot.learn_countries({"Bangladesh": 5})
    new = await otp_bot.learn_countries({"Bangladesh": 3, "Nigeria": 7})
    assert new == ["Nigeria"]


async def test_our_guessed_name_is_replaced_by_the_bots_spelling(environment):
    """Our prefix table is a guess; /frcd and /setlimit key on the bot's own
    name, so a mismatch would silently never match its stock line.
    """
    await otp_bot.learn_countries({"Congo (DRC)": 12})

    assert await otp_bot.canonical_country("Congo") == "Congo (DRC)"
    assert await otp_bot.canonical_country("congo (drc)") == "Congo (DRC)"


async def test_an_unknown_country_keeps_our_guess(environment):
    """A brand-new country still has to work - it just gets corrected the
    first time the bot reports it.
    """
    assert await otp_bot.canonical_country("Bangladesh") == "Bangladesh"


async def test_uploads_adopt_the_bots_country_name(environment):
    await otp_bot.learn_countries({"Central African Rep.": 5})

    rel = await _write_numbers_file("caf.txt", prefix="+236")
    analysis = await otp_bot.enqueue_file(rel, "caf.txt")

    # Our table says "Central African Republic"; the bot says otherwise.
    assert analysis["entries"][0]["country"] == "Central African Rep."


async def test_a_cycle_learns_the_country_list(environment):
    await _start_with_one_active_file()
    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    known = (await otp_bot.get_known_countries())["names"]
    assert "central african republic" in known
    assert "bangladesh" in known
    assert sorted(result.new_countries) == ["Bangladesh", "Central African Republic"]


async def test_refresh_reports_the_bots_countries(environment):
    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    outcome = await otp_bot.refresh_countries_from_bot()

    assert outcome["ok"] is True
    assert outcome["countries"]["Central African Republic"] == 90712
    assert "Bangladesh" in outcome["new"]


async def test_refresh_fails_cleanly_when_the_bot_says_nothing_useful(environment):
    set_userbot(FakeUserbot([{"text": "\u26A1 Fast Add Complete!", "out": False}]))
    outcome = await otp_bot.refresh_countries_from_bot()

    assert outcome["ok"] is False
    assert outcome["countries"] == {}


async def test_check_now_ignores_the_per_country_timers(environment):
    """A human pressing "Check now" is asking for a check NOW. Answering
    "not due yet" is what made the button look broken.
    """
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    # start_automation just armed the timer, so an unforced cycle must wait.

    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    lazy = await otp_bot.run_cycle(await otp_bot.get_config())
    assert lazy.action == "not due yet"

    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    forced = await otp_bot.run_cycle(await otp_bot.get_config(), force=True)
    assert forced.action != "not due yet"
    assert forced.country_stock, "a forced check must actually read the stock"


async def test_a_forced_check_rearms_the_timer(environment):
    """Otherwise a manual check leaves the country due, and the next
    scheduler tick immediately repeats the work.
    """
    from app.automation import otp_schedule

    await _start_with_one_active_file()
    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    await otp_bot.run_cycle(await otp_bot.get_config(), force=True)

    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))
    following = await otp_bot.run_cycle(await otp_bot.get_config())
    assert following.action == "not due yet"
    assert await otp_schedule.get_due_at("Bangladesh") is not None


async def test_no_reply_error_names_the_command_it_sent(environment):
    """"no reply to the quota command" was unhelpful once the command became
    configurable - it never said which command or which bot.
    """
    await _start_with_one_active_file()
    set_userbot(FakeUserbot([{"text": "some chatter with no stock", "out": False}]))

    result = await otp_bot.run_cycle(await otp_bot.get_config(), force=True)

    assert result.ok is False
    assert "/st" in result.error
    assert "@PBDxbot" in result.error


# --------------------------------------------------------------------------- #
# Editing settings from chat
# --------------------------------------------------------------------------- #
def test_setting_values_are_validated_before_they_are_stored():
    """A bad value stored silently is worse than a rejected one: the
    automation would keep running with settings the owner never intended.
    """
    assert otp_bot.coerce_setting("interval_minutes", "15") == 15
    assert otp_bot.coerce_setting("force_delete_before_add", "on") is True
    assert otp_bot.coerce_setting("force_delete_before_add", "bondho") is False

    with pytest.raises(ValueError):
        otp_bot.coerce_setting("interval_minutes", "abc")
    with pytest.raises(ValueError):
        otp_bot.coerce_setting("interval_minutes", "0")       # below the minimum
    with pytest.raises(ValueError):
        otp_bot.coerce_setting("no_such_key", "1")
    with pytest.raises(ValueError):
        otp_bot.coerce_setting("target_bot", "PBDxbot")       # missing the @


def test_command_templates_must_keep_their_placeholders():
    """A template that loses {tag} still "works" - it just sends the wrong
    command forever, which is the hardest kind of bug to notice.
    """
    with pytest.raises(ValueError):
        otp_bot.coerce_setting("add_command_template", "/fan -l 4")
    with pytest.raises(ValueError):
        otp_bot.coerce_setting("force_delete_command", "/frcd {uid}")

    assert otp_bot.coerce_setting("add_command_template", "/fan -t {tag}") == "/fan -t {tag}"


async def test_setting_a_value_from_chat_persists_it(environment):
    reply = await otp_bot.set_setting_from_chat("interval_minutes", "25")
    assert "25" in reply
    assert (await otp_bot.get_config())["interval_minutes"] == 25


async def test_per_country_setting_from_chat_only_touches_that_country(environment):
    from app.automation import otp_schedule

    await otp_bot.set_country_setting_from_chat("Bangladesh", "quota_threshold", "200")

    cfg = await otp_bot.get_config()
    assert (await otp_schedule.effective_config("Bangladesh", cfg))["quota_threshold"] == 200
    assert (await otp_schedule.effective_config("Nigeria", cfg))["quota_threshold"] == cfg[
        "quota_threshold"
    ]


async def test_per_country_setting_uses_the_bots_spelling(environment):
    from app.automation import otp_schedule

    await otp_bot.learn_countries({"Central African Republic": 5})
    reply = await otp_bot.set_country_setting_from_chat(
        "central african republic", "interval_minutes", "5"
    )

    assert "Central African Republic" in reply
    settings = await otp_schedule.get_country_settings("Central African Republic")
    assert settings["interval_minutes"] == 5


async def test_a_global_only_field_is_rejected_per_country(environment):
    """target_bot is a property of the bot being driven, not of one country -
    accepting it per country would store a setting that never applies.
    """
    with pytest.raises(ValueError):
        await otp_bot.set_country_setting_from_chat("Bangladesh", "target_bot", "@other")


async def test_settings_listing_shows_current_values(environment):
    await otp_bot.save_config({"interval_minutes": 42})
    text = await otp_bot.describe_settings()

    assert "interval_minutes = 42" in text
    assert "/otpset" in text
    for key in otp_bot.SETTABLE_FIELDS:
        assert key in text


# --------------------------------------------------------------------------- #
# Finishing on its own: N re-adds, or N minutes, then optionally delete
# --------------------------------------------------------------------------- #
async def _active_country(name: str = "bd.txt", prefix: str = "+880") -> dict:
    rel = await _write_numbers_file(name, prefix=prefix)
    entry = (await otp_bot.enqueue_file(rel, name))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    return entry


def _empty_stock_bot() -> FakeUserbot:
    """Bot with zero Bangladesh stock, so every cycle wants to refill."""
    return FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\u2705 500 added", "out": False},
    ])


async def test_a_country_stops_after_its_refill_limit(environment):
    """"Run it 2 times" must mean 2, not 3 - the check happens before the
    re-add, not after.
    """
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings("Bangladesh", {"max_refills": 2})

    for _ in range(2):
        await _make_everything_due()
        set_userbot(_empty_stock_bot())
        result = await otp_bot.run_cycle(await otp_bot.get_config())
        assert result.finished == []

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    third = await otp_bot.run_cycle(await otp_bot.get_config())

    assert [f["country"] for f in third.finished] == ["Bangladesh"]
    assert "2 bar" in third.finished[0]["reason"]
    # And it stops being monitored.
    assert await otp_bot.get_active_files() == []


async def test_a_country_stops_after_its_time_limit(environment):
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings("Bangladesh", {"run_minutes": 30})

    # Backdate the run so the limit has passed.
    state = await otp_schedule._get_run_state()
    state["bangladesh"]["started_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=31)
    ).isoformat()
    await otp_schedule._save_run_state(state)

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert [f["country"] for f in result.finished] == ["Bangladesh"]
    assert "30 min" in result.finished[0]["reason"]


async def test_no_limits_means_it_keeps_running(environment):
    """The default must stay "run until told to stop"."""
    await _active_country()

    for _ in range(3):
        await _make_everything_due()
        set_userbot(_empty_stock_bot())
        result = await otp_bot.run_cycle(await otp_bot.get_config())
        assert result.finished == []

    assert len(await otp_bot.get_active_files()) == 1


async def test_finishing_deletes_the_numbers_when_asked(environment):
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings(
        "Bangladesh", {"max_refills": 1, "delete_when_done": True}
    )
    await otp_bot.save_config({"force_delete_uid": "12345"})

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    await otp_bot.run_cycle(await otp_bot.get_config())

    await _make_everything_due()
    bot = FakeUserbot([
        {"text": REAL_ST_REPLY, "out": False},
        {"text": "\U0001F5D1 Deleted 500 numbers", "out": False},
    ])
    set_userbot(bot)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.finished[0]["deleted"] is True
    sent = " ".join(text for _, text, *_rest in bot.sent_messages)
    assert "/frcd Bangladesh 12345" in sent


async def test_finishing_does_not_delete_by_default(environment):
    """Deleting is irreversible on the bot, so a run merely ending must not
    trigger it.
    """
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings("Bangladesh", {"max_refills": 1})

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    await otp_bot.run_cycle(await otp_bot.get_config())

    await _make_everything_due()
    bot = _empty_stock_bot()
    set_userbot(bot)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.finished[0]["deleted"] is False
    assert "/frcd" not in " ".join(text for _, text, *_r in bot.sent_messages)


async def test_delete_without_a_uid_reports_it_instead_of_sending_junk(environment):
    """Sending "/frcd Bangladesh {uid}" literally is a silent no-op that
    looks like it worked.
    """
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings(
        "Bangladesh", {"max_refills": 1, "delete_when_done": True}
    )
    await otp_bot.save_config({"force_delete_uid": ""})

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    await otp_bot.run_cycle(await otp_bot.get_config())

    await _make_everything_due()
    bot = _empty_stock_bot()
    set_userbot(bot)
    result = await otp_bot.run_cycle(await otp_bot.get_config())

    assert result.finished[0]["deleted"] is False
    assert "user id" in result.finished[0]["error"]
    assert "{uid}" not in " ".join(text for _, text, *_r in bot.sent_messages)
    # The run still ends - a failed delete must not leave it cycling.
    assert await otp_bot.get_active_files() == []


async def test_limits_are_per_country(environment):
    """One country finishing must not retire a sibling that is still going."""
    from app.automation import otp_schedule

    # Both countries queued BEFORE starting: start_automation replaces the
    # active set, so starting twice would drop the first country.
    for name, prefix in (("bd.txt", "+880"), ("ng.txt", "+234")):
        rel = await _write_numbers_file(name, prefix=prefix)
        entry = (await otp_bot.enqueue_file(rel, name))["entries"][0]
        await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    assert len(await otp_bot.get_active_files()) == 2

    await otp_schedule.set_country_settings("Bangladesh", {"max_refills": 0})
    await otp_schedule.set_country_settings("Nigeria", {"max_refills": 1})

    for _ in range(2):
        await _make_everything_due()
        set_userbot(FakeUserbot([
            {"text": REAL_ST_REPLY, "out": False},
            {"text": "\u2705 500 added", "out": False},
        ]))
        await otp_bot.run_cycle(await otp_bot.get_config())

    remaining = {e.get("country") for e in await otp_bot.get_active_files()}
    assert "Bangladesh" in remaining
    assert "Nigeria" not in remaining


async def test_the_owner_is_told_when_a_run_finishes(environment):
    from app.automation import otp_schedule

    await _active_country()
    await otp_schedule.set_country_settings("Bangladesh", {"max_refills": 1})

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    await otp_bot.run_cycle(await otp_bot.get_config())

    await _make_everything_due()
    set_userbot(_empty_stock_bot())
    notifier = _CapturingNotifier()
    await _run_scheduler_once(notifier)

    joined = " ".join(n["text"] for n in notifier.sent)
    assert "shesh" in joined
    assert "Bangladesh" in joined


# --------------------------------------------------------------------------- #
# Keeping the chat clean: the stock check is bookkeeping, not conversation
# --------------------------------------------------------------------------- #
class _DeletingBot(FakeUserbot):
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        super().__init__(replies)
        self.deleted: list[int] = []

    async def delete_messages(self, target: str, message_ids: list[int]) -> dict:
        self.deleted.extend(message_ids)
        return {"deleted": len(message_ids)}


async def test_the_stock_command_and_its_reply_are_deleted(environment):
    """On a 2-minute interval this traffic buries the chat the owner reads."""
    await _start_with_one_active_file()
    bot = _DeletingBot([{"text": REAL_ST_REPLY, "out": False}])
    set_userbot(bot)

    result = await otp_bot.run_cycle(await otp_bot.get_config(), force=True)

    # Both sides of the exchange go: the command we sent and the bot's reply.
    assert len(bot.deleted) == 2
    # And the data was still read before deleting.
    assert result.country_stock


async def test_tidying_can_be_turned_off(environment):
    await _start_with_one_active_file()
    await otp_bot.save_config({"tidy_stock_messages": False})
    bot = _DeletingBot([{"text": REAL_ST_REPLY, "out": False}])
    set_userbot(bot)

    await otp_bot.run_cycle(await otp_bot.get_config(), force=True)
    assert bot.deleted == []


async def test_a_userbot_that_cannot_delete_still_works(environment):
    """Older backends have no delete_messages; tidying is cosmetic and must
    never cost a cycle.
    """
    await _start_with_one_active_file()
    set_userbot(FakeUserbot([{"text": REAL_ST_REPLY, "out": False}]))

    result = await otp_bot.run_cycle(await otp_bot.get_config(), force=True)
    assert result.ok is True


async def test_a_failing_delete_does_not_fail_the_cycle(environment):
    class _BrokenDeleter(FakeUserbot):
        async def delete_messages(self, target, message_ids):
            raise RuntimeError("cannot delete")

    await _start_with_one_active_file()
    set_userbot(_BrokenDeleter([{"text": REAL_ST_REPLY, "out": False}]))

    result = await otp_bot.run_cycle(await otp_bot.get_config(), force=True)
    assert result.ok is True
    assert result.country_stock


# --------------------------------------------------------------------------- #
# Saying when a run will end - the overnight failure was a silent stop
# --------------------------------------------------------------------------- #
async def test_setting_a_run_limit_warns_that_it_ends_the_run(environment):
    """"run_minutes 3" reads naturally as "every 3 minutes". It actually
    means "stop after 3 minutes", which is how an overnight run ended
    minutes after it started.
    """
    reply = await otp_bot.set_setting_from_chat("run_minutes", "3")
    assert "BONDHO" in reply
    assert "run_minutes 0" in reply  # tells you how to undo it

    reply = await otp_bot.set_setting_from_chat("max_refills", "3")
    assert "BONDHO" in reply


async def test_clearing_a_limit_does_not_warn(environment):
    reply = await otp_bot.set_setting_from_chat("run_minutes", "0")
    assert "BONDHO" not in reply


async def test_delete_when_done_warns_that_numbers_will_be_removed(environment):
    reply = await otp_bot.set_setting_from_chat("delete_when_done", "on")
    assert "MUCHE" in reply


async def test_start_says_when_the_run_will_end(environment):
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    await otp_bot.save_config({"run_minutes": 3, "max_refills": 2})
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))

    result = await otp_bot.start_automation()
    text = otp_bot._format_start_success(result)

    assert "3 min" in text
    assert "2 bar" in text
    assert "BONDHO" in text


async def test_start_says_when_there_are_no_limits(environment):
    """The common case needs to be equally explicit, or "it will keep
    running" is just an assumption.
    """
    rel = await _write_numbers_file("bd.txt", prefix="+880")
    entry = (await otp_bot.enqueue_file(rel, "bd.txt"))["entries"][0]
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    await otp_bot.save_config({"run_minutes": 0, "max_refills": 0})
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))

    result = await otp_bot.start_automation()
    text = otp_bot._format_start_success(result)

    assert "limit nai" in text
    assert "BONDHO" not in text


# --------------------------------------------------------------------------- #
# Removing things the owner decided against
# --------------------------------------------------------------------------- #
async def test_skip_during_the_tag_question_drops_that_country(environment):
    """"skip"/"bad dao" at the tag prompt is how a wrong country gets out."""
    rel = await _write_mixed_country_file("export.txt")
    await otp_bot.enqueue_file(rel, "export.txt")

    first = await otp_bot.handle_start_trigger()
    assert "Bangladesh" in first or "Dominican Republic" in first
    skipped_country = (await otp_bot.get_awaiting_tag_entry())["country"]

    reply = await otp_bot.handle_tag_answer("bad dao")

    assert "bad deoa holo" in reply
    remaining = {e["country"] for e in await otp_bot.get_queue()}
    assert skipped_country not in remaining
    # And it moved straight on to asking about the other country.
    assert await otp_bot.get_awaiting_tag_entry() is not None


async def test_remove_by_country_takes_it_out_of_queue_and_active(environment):
    rel_bd = await _write_numbers_file("bd.txt", prefix="+880")
    entry_bd = await _enqueue_single(rel_bd, "bd.txt")
    await otp_bot.set_queue_tag(entry_bd["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()
    assert len(await otp_bot.get_active_files()) == 1

    # A second country arrives and is still only queued.
    rel_ng = await _write_numbers_file("ng.txt", prefix="+234")
    await _enqueue_single(rel_ng, "ng.txt")

    removed = await otp_bot.remove_by_country("bangladesh")  # case-insensitive
    assert len(removed) == 1
    assert await otp_bot.get_active_files() == []
    # Removing the last active file also stops the monitor.
    assert (await otp_bot.get_config())["enabled"] is False
    # The unrelated queued country is untouched.
    assert [e["country"] for e in await otp_bot.get_queue()] == ["Nigeria"]


async def test_remove_active_file_leaves_the_others_running(environment):
    rel = await _write_mixed_country_file("export.txt")
    await otp_bot.enqueue_file(rel, "export.txt")
    for entry in await otp_bot.get_queue():
        await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()

    active = await otp_bot.get_active_files()
    assert len(active) == 2

    removed = await otp_bot.remove_active_file(active[0]["id"])
    assert removed is not None
    assert len(await otp_bot.get_active_files()) == 1
    # Still work left to do, so the monitor stays on.
    assert (await otp_bot.get_config())["enabled"] is True


async def test_clear_queue_does_not_touch_active_files(environment):
    rel_bd = await _write_numbers_file("bd.txt", prefix="+880")
    entry_bd = await _enqueue_single(rel_bd, "bd.txt")
    await otp_bot.set_queue_tag(entry_bd["id"], "WhatsApp")
    set_userbot(FakeUserbot([{"text": "Added.", "out": False}]))
    await otp_bot.start_automation()

    rel_ng = await _write_numbers_file("ng.txt", prefix="+234")
    await _enqueue_single(rel_ng, "ng.txt")

    cleared = await otp_bot.clear_queue()
    assert cleared == 1
    assert await otp_bot.get_queue() == []
    # Running work is deliberately left alone.
    assert len(await otp_bot.get_active_files()) == 1
    assert (await otp_bot.get_config())["enabled"] is True


async def test_removing_the_entry_being_asked_about_clears_the_prompt(environment):
    """A dangling prompt would wedge the conversation: every later message
    gets read as a tag answer for an entry that no longer exists.
    """
    rel = await _write_mixed_country_file("export.txt")
    await otp_bot.enqueue_file(rel, "export.txt")
    await otp_bot.handle_start_trigger()

    pending = await otp_bot.get_awaiting_tag_entry()
    assert pending is not None

    await otp_bot.remove_from_queue(pending["id"])
    config = await otp_bot.get_config()
    assert config["awaiting_tag_entry_id"] is None


async def test_removal_phrases_are_recognised():
    assert otp_bot.is_skip_trigger("skip")
    assert otp_bot.is_skip_trigger("bad dao")
    assert otp_bot.is_clear_trigger("sob bad dao")
    assert otp_bot.country_to_remove("Bangladesh bad dao") == "bangladesh"
    assert otp_bot.country_to_remove("remove Nigeria") == "nigeria"
    # An ordinary sentence must never be read as a removal.
    assert otp_bot.country_to_remove("Bangladesh e koyta number ache") is None
    assert otp_bot.country_to_remove("start") is None


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
    # start_automation arms each country's own timer, so a cycle run
    # immediately afterwards would correctly report "not due yet". These
    # tests are about what a cycle DOES when it fires, so make it due.
    await _make_everything_due()
    return fake


async def _make_everything_due() -> None:
    """Backdate every country's next-check time so the next cycle fires."""
    from app.automation import otp_schedule

    due = await otp_schedule._get_due_map()
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    await otp_schedule._save_due_map({key: past for key in due})


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
    assert result.action.startswith("added")
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
