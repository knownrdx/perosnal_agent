"""The owner's required behaviour for the OTP-bot automation, end to end.

Each test is one scenario the owner described, driven through the real entry
points (enqueue_file / maybe_auto_start / start_automation / run_cycle) with a
fake @PBDxbot and a fake clock:

  - The clock is otp_schedule._now, which every schedule read goes through.
    Dubai is UTC+4, so "10:00 Dubai" is 06:00 UTC.
  - The fake bot answers by COMMAND rather than by position in a script (the
    reply still advances on SEND, with a fresh id, like FakeUserbot in
    test_otp_bot_automation.py). That keeps these scenarios independent of
    exactly how many commands an implementation sends in which order - what
    they assert is what reaches the target bot, not the route there.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.integrations.telegram_user import set_userbot

DUBAI = otp_schedule.DUBAI_TZ

BD_FLAG = "\U0001F1E7\U0001F1E9"
STOCK_HEADER = "\U0001F30D Country Stock (yours):"
FAST_ADD = "⚡ Fast Add Complete!\n\nBangladesh: 3501 added (16499 dup)"
SPENT_ADD = "⚡ Fast Add Complete!\n\nBangladesh: 0 added (20000 dup)"

STOCKED = 1234          # comfortably above the default quota_threshold of 0
NEXT_CHECK = 11         # minutes: one past the default 10-minute interval


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #
def dubai(day: int, hour: int, minute: int = 0) -> datetime:
    """An aware UTC instant for a Dubai wall-clock time on 2026-10-<day>."""
    return datetime(2026, 10, day, hour, minute, tzinfo=DUBAI).astimezone(timezone.utc)


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def set(self, when: datetime) -> None:
        self.now = when

    def advance(self, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock(monkeypatch) -> Clock:
    """Starts at 10:00 Dubai on 5 October - the owner's daytime upload."""
    fake = Clock(dubai(5, 10, 0))
    monkeypatch.setattr(otp_schedule, "_now", fake)
    # Timestamps written by otp_bot itself (uploaded_at etc.) follow the same
    # clock, where that helper still exists.
    monkeypatch.setattr(otp_bot, "_now_iso", lambda: fake.now.isoformat(), raising=False)
    return fake


@pytest.fixture(autouse=True)
def _no_sleep_and_reset_userbot(monkeypatch):
    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)
    yield
    set_userbot(None)


# --------------------------------------------------------------------------- #
# Fake @PBDxbot
# --------------------------------------------------------------------------- #
class PBDxBot:
    """Answers like the real bot and records every send, in order.

    The current reply advances on SEND (never on read), each with a fresh id,
    because the automation anchors on the last message and then waits for a
    newer one. Every call yields to the event loop, as real network I/O does,
    so two concurrent cycles get the chance to interleave if nothing stops
    them.
    """

    def __init__(self, bangladesh: int = 0, add_reply: str = FAST_ADD) -> None:
        self.stock: dict[str, int] = {"Bangladesh": bangladesh}
        self.add_reply = add_reply
        # ("msg" | "file", text-or-path, reply_to)
        self.log: list[tuple[str, str, int | None]] = []
        self.deleted: list[int] = []
        self._next_message_id = 1000
        self._reply_id = 0
        self._current: dict[str, Any] | None = None
        # Concurrency probe (off unless a test sets it): the first stock
        # check is held open for up to this many seconds, released early the
        # moment a SECOND stock check is sent. Without mutual exclusion the
        # second cycle always gets in while the first is parked here, so the
        # interleaving is deterministic instead of depending on DB timing;
        # with it, the second cycle cannot send and the hold simply expires.
        self.hold_first_stock_check: float = 0.0
        self._stock_checks = 0
        self._second_stock_check = asyncio.Event()

    def stock_text(self) -> str:
        lines = [
            "\U0001F4CA Bot Statistics",
            "",
            "\U0001F4F1 Numbers:",
            "  • Available: 416036",
            "",
            STOCK_HEADER,
        ]
        for country, count in self.stock.items():
            flag = BD_FLAG if country == "Bangladesh" else "\U0001F3F3️"
            lines.append(f"  {flag} {country}: {count} (+5 taken)")
        return "\n".join(lines)

    def _answer(self, text: str, reply_to: int | None) -> str:
        command = text.strip()
        head = command.split()[0] if command else ""
        if reply_to is not None or head == "/fan":
            return self.add_reply
        if head == "/st":
            return self.stock_text()
        if head == "/frcd":
            return "\U0001F5D1 Force-deleted Bangladesh numbers"
        if head == "/useddelete":
            return "\U0001F9F9 Deleted 0 used numbers"
        return "✅ OK"

    async def send_message(self, target: str, text: str, reply_to: int | None = None) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.log.append(("msg", text, reply_to))
        self._next_message_id += 1
        self._reply_id += 1
        self._current = {"id": self._reply_id, "text": self._answer(text, reply_to), "out": False}
        if self.hold_first_stock_check and text.strip().split()[:1] == ["/st"]:
            self._stock_checks += 1
            if self._stock_checks == 1:
                try:
                    await asyncio.wait_for(
                        self._second_stock_check.wait(), self.hold_first_stock_check
                    )
                except asyncio.TimeoutError:
                    pass
            else:
                self._second_stock_check.set()
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.log.append(("file", str(path), reply_to))
        self._next_message_id += 1
        return {"sent": True, "message_id": self._next_message_id, "to": target}

    async def read_messages(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        return [] if self._current is None else [dict(self._current)]

    async def delete_messages(self, target: str, message_ids: list[int]) -> dict[str, Any]:
        self.deleted.extend(message_ids)
        return {"deleted": len(message_ids)}

    # -- inspection ----------------------------------------------------- #
    def mark(self) -> int:
        return len(self.log)

    def messages(self, since: int = 0) -> list[str]:
        return [text for kind, text, _ in self.log[since:] if kind == "msg"]

    def files(self, since: int = 0) -> list[str]:
        """Names of the files sent, e.g. ["bd.txt"]."""
        return [Path(p).name for kind, p, _ in self.log[since:] if kind == "file"]

    def add_commands(self, since: int = 0) -> list[str]:
        return [t for kind, t, reply_to in self.log[since:] if kind == "msg" and reply_to is not None]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _upload(name: str = "bd.txt", caption: str = "whatsapp 20h") -> dict[str, Any]:
    """Queue a Bangladesh file. The default caption names the service and the
    run length but NO time, so the global 04:00 start applies."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(
        "+8801711111111\n+8801722222222\n+8801733333333\n", encoding="utf-8"
    )
    return await otp_bot.enqueue_file(rel_path(path), name, caption)


async def _upload_and_auto_start(name: str = "bd.txt", caption: str = "whatsapp 20h") -> dict[str, Any]:
    """What a file dropped in the OTP thread does: queue it, start by itself."""
    await _upload(name, caption)
    result = await otp_bot.maybe_auto_start()
    assert result is not None and result["ok"], result
    return result


async def _upload_and_start_now(name: str = "bd.txt", caption: str = "whatsapp 20h") -> dict[str, Any]:
    """What pressing Start does: begin immediately."""
    await _upload(name, caption)
    result = await otp_bot.start_automation()
    assert result["ok"], result
    return result


async def _has_started(country: str = "Bangladesh") -> bool:
    return await otp_schedule.has_started(country, await otp_bot.get_config())


async def _active_bangladesh() -> list[dict[str, Any]]:
    return [e for e in await otp_bot.get_active_files() if e.get("country") == "Bangladesh"]


async def _quota_command() -> str:
    return (await otp_bot.get_config())["quota_command"]


def _countries(items: Any) -> list[str]:
    """Country names out of a result list, whether it holds dicts or names."""
    return [
        item.get("country") if isinstance(item, dict) else str(item)
        for item in (items or [])
    ]


def _assert_no_cleanup_sent(bot: PBDxBot, since: int = 0) -> None:
    sent = [text.strip() for text in bot.messages(since)]
    assert not any(text.startswith("/useddelete") for text in sent), sent
    # Blank cleanup_command means "send nothing", not "send an empty message".
    assert all(sent), f"a blank message was sent to the bot: {sent}"


# --------------------------------------------------------------------------- #
# 1. No cleanup command
# --------------------------------------------------------------------------- #
async def test_cleanup_command_defaults_to_blank(environment):
    assert otp_bot.DEFAULT_CONFIG["cleanup_command"] == ""
    assert (await otp_bot.get_config())["cleanup_command"] == ""


async def test_start_automation_sends_no_cleanup_command(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)

    await _upload_and_start_now()

    assert bot.files() == ["bd.txt"], "precondition: the empty country was added"
    _assert_no_cleanup_sent(bot)


async def test_a_scheduled_start_sends_no_cleanup_command(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_auto_start()

    clock.set(dubai(6, 4, 0))
    await otp_bot.run_cycle()

    assert bot.files(), "precondition: the 04:00 start added the file"
    _assert_no_cleanup_sent(bot)


async def test_a_refill_cycle_sends_no_cleanup_command(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_start_now()

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    await otp_bot.run_cycle()

    assert bot.files(mark) == ["bd.txt"], "precondition: the refill happened"
    _assert_no_cleanup_sent(bot)


# --------------------------------------------------------------------------- #
# 2. A scheduled start holds a stocked country - it never wipes it
# --------------------------------------------------------------------------- #
async def test_a_scheduled_upload_touches_nothing_before_four_am(clock):
    """Uploaded at 10:00 with the default 04:00 start: until 04:00 no file,
    no add, no /frcd and no cleanup reach the bot (a stock check is the only
    thing tolerated), however often the cycle runs or "Check now" is pressed.
    """
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await otp_bot.save_config({"force_delete_before_add": True, "force_delete_uid": "789019025"})

    await _upload_and_auto_start()
    for when in (dubai(5, 10, 30), dubai(5, 18, 0), dubai(6, 3, 59)):
        clock.set(when)
        await otp_bot.run_cycle()
        await otp_bot.run_cycle(force=True)

    quota = await _quota_command()
    assert bot.files() == []
    assert [t for t in bot.messages() if t.strip() != quota] == []
    assert await _has_started() is False


async def test_a_scheduled_start_holds_a_stocked_country_instead_of_wiping_it(clock):
    bot = PBDxBot(bangladesh=STOCKED)
    set_userbot(bot)
    # Even with wipe-before-add switched on: a start is not a refill.
    await otp_bot.save_config({"force_delete_before_add": True, "force_delete_uid": "789019025"})
    await _upload_and_auto_start()

    clock.set(dubai(6, 4, 0))
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    sent = [t.strip() for t in bot.messages(mark)]
    assert not any(t.startswith("/frcd") for t in sent), sent
    assert bot.files(mark) == []
    assert bot.add_commands(mark) == []
    assert "Bangladesh" in _countries(getattr(result, "held_at_start", None))
    # Held, but started: from here on it is an ordinary running country.
    assert await _has_started() is True
    assert len(await _active_bangladesh()) == 1


async def test_a_file_held_at_the_scheduled_start_is_added_once_the_country_runs_empty(clock):
    bot = PBDxBot(bangladesh=STOCKED)
    set_userbot(bot)
    await _upload_and_auto_start()

    clock.set(dubai(6, 4, 0))
    await otp_bot.run_cycle()
    assert bot.files() == [], "the stocked country should have been held at 04:00"

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    await otp_bot.run_cycle()
    assert bot.files(mark) == [], "still stocked, so still held"

    bot.stock["Bangladesh"] = 0
    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    assert bot.files(mark) == ["bd.txt"], (result.action, result.error)
    assert len(bot.add_commands(mark)) == 1


# --------------------------------------------------------------------------- #
# 3. After a scheduled start, refills keep happening
# --------------------------------------------------------------------------- #
async def test_refills_keep_happening_after_a_scheduled_start(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_auto_start()

    clock.set(dubai(6, 4, 0))
    mark = bot.mark()
    await otp_bot.run_cycle()
    assert len(bot.files(mark)) >= 1, "the 04:00 start added nothing"

    for refill in range(1, 4):
        clock.advance(NEXT_CHECK)
        mark = bot.mark()
        result = await otp_bot.run_cycle()
        assert bot.files(mark) == ["bd.txt"], (
            f"refill #{refill} after the 04:00 start did not happen: "
            f"{result.action} {result.error}"
        )
    assert await _has_started() is True


# --------------------------------------------------------------------------- #
# 4. Pressing "Now" on a country still waiting for 04:00
# --------------------------------------------------------------------------- #
async def test_pressing_now_releases_a_country_waiting_for_four_am(clock):
    set_userbot(PBDxBot(bangladesh=0))
    await _upload_and_auto_start()
    assert await _has_started() is False, "precondition: waiting for 04:00"

    clock.set(dubai(5, 10, 5))
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": "now"})

    assert await _has_started() is True


async def test_after_pressing_now_the_next_cycle_adds_an_empty_country(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_auto_start()

    clock.set(dubai(5, 10, 5))
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": "now"})
    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    assert bot.files(mark) == ["bd.txt"], (result.action, result.error)


async def test_after_pressing_now_the_next_cycle_holds_a_stocked_country(clock):
    bot = PBDxBot(bangladesh=STOCKED)
    set_userbot(bot)
    await _upload_and_auto_start()

    clock.set(dubai(5, 10, 5))
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": "now"})
    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    quota = await _quota_command()
    assert quota in [t.strip() for t in bot.messages(mark)], (
        f"the released country was not checked: {result.action} {result.error}"
    )
    assert bot.files(mark) == []
    assert len(await _active_bangladesh()) == 1
    assert await _has_started() is True


# --------------------------------------------------------------------------- #
# 5. Changing the start time of a running country does not pause it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("new_start_at", ["05:00", None])
async def test_changing_the_start_time_of_a_running_country_does_not_pause_it(clock, new_start_at):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_start_now()

    clock.advance(30)
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": new_start_at})
    assert await _has_started() is True

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()
    assert bot.files(mark) == ["bd.txt"], (result.action, result.error)


@pytest.mark.parametrize("new_start_at", ["05:00", None])
async def test_changing_the_start_time_after_a_scheduled_start_does_not_pause_it(clock, new_start_at):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_auto_start()
    clock.set(dubai(6, 4, 0))
    await otp_bot.run_cycle()
    assert await _has_started() is True, "precondition: started at 04:00"

    clock.set(dubai(6, 4, 30))
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": new_start_at})
    assert await _has_started() is True

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()
    assert bot.files(mark) == ["bd.txt"], (result.action, result.error)


# --------------------------------------------------------------------------- #
# 6. A new file for a running country is not gated until 04:00
# --------------------------------------------------------------------------- #
async def test_a_new_file_for_a_running_country_is_not_gated_until_four_am(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_start_now("bd.txt")

    clock.set(dubai(5, 12, 0))
    bot.stock["Bangladesh"] = STOCKED
    await _upload_and_auto_start("bd2.txt")          # caption names no time

    assert await _has_started() is True

    # Still checked on its own interval, not left alone until tomorrow 04:00.
    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()
    quota = await _quota_command()
    assert quota in [t.strip() for t in bot.messages(mark)], (result.action, result.error)


async def test_a_new_file_for_a_running_country_replaces_the_old_one_and_waits_for_empty(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_start_now("bd.txt")
    assert bot.files() == ["bd.txt"], "precondition: first file added"

    clock.set(dubai(5, 12, 0))
    bot.stock["Bangladesh"] = STOCKED
    mark = bot.mark()
    await _upload_and_auto_start("bd2.txt")

    assert bot.files(mark) == [], "stock is still there, so the new file is held"
    assert [e["name"] for e in await _active_bangladesh()] == ["bd2.txt"]

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    await otp_bot.run_cycle()
    assert bot.files(mark) == []

    bot.stock["Bangladesh"] = 0
    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    result = await otp_bot.run_cycle()
    assert bot.files(mark) == ["bd2.txt"], (result.action, result.error)


# --------------------------------------------------------------------------- #
# 7. Run length: finish on time, and a finish HOLDS the country
# --------------------------------------------------------------------------- #
async def _start_sixty_minute_run() -> None:
    """Bangladesh, started now (T), run_minutes=60, checked every minute."""
    await _upload("bd.txt")
    await otp_schedule.set_country_settings(
        "Bangladesh", {"run_minutes": 60, "interval_minutes": 1}
    )
    result = await otp_bot.start_automation()
    assert result["ok"], result


async def test_a_run_still_refills_one_minute_before_its_end(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _start_sixty_minute_run()

    clock.advance(59)
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    assert result.finished == []
    assert bot.files(mark) == ["bd.txt"], (result.action, result.error)


@pytest.mark.parametrize("stock_at_end", [STOCKED, 0])
async def test_a_run_finishes_on_time_and_is_held_not_removed(clock, stock_at_end):
    from app.security import safe_path

    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _start_sixty_minute_run()
    clock.advance(59)
    await otp_bot.run_cycle()                       # last refill inside the hour

    bot.stock["Bangladesh"] = stock_at_end
    clock.advance(2)                                # T+61
    mark = bot.mark()
    result = await otp_bot.run_cycle()

    finished = [f for f in result.finished if f.get("country") == "Bangladesh"]
    assert finished, f"not finished at T+61: {result.action} {result.error}"
    assert finished[0].get("held") is True
    assert bot.files(mark) == []

    entries = await _active_bangladesh()
    assert len(entries) == 1, "a finished country stays in the active set"
    assert entries[0].get("finished_at")
    assert await otp_schedule.is_paused("Bangladesh") is True
    assert safe_path(entries[0]["path"]).exists()

    # Nothing more is added, even once the bot runs dry.
    bot.stock["Bangladesh"] = 0
    for _ in range(3):
        clock.advance(2)
        mark = bot.mark()
        await otp_bot.run_cycle()
        assert bot.files(mark) == []


# --------------------------------------------------------------------------- #
# 8. delete_when_done is off by default: finishing never sends /frcd
# --------------------------------------------------------------------------- #
async def test_delete_when_done_is_off_by_default(environment):
    assert otp_bot.DEFAULT_CONFIG["delete_when_done"] is False
    assert (await otp_bot.get_config())["delete_when_done"] is False


@pytest.mark.parametrize("stock_at_end", [0, STOCKED])
async def test_finishing_never_sends_frcd_by_default(clock, stock_at_end):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    # A uid is configured, so /frcd WOULD be possible - it must still not run.
    await otp_bot.save_config({"force_delete_uid": "789019025"})
    await _start_sixty_minute_run()

    bot.stock["Bangladesh"] = stock_at_end
    clock.advance(61)
    result = await otp_bot.run_cycle()

    assert _countries(result.finished) == ["Bangladesh"], (result.action, result.error)
    assert not result.finished[0].get("deleted")
    assert not any(t.strip().startswith("/frcd") for t in bot.messages()), bot.messages()


# --------------------------------------------------------------------------- #
# 9. An exhausted file is not re-sent every cycle
# --------------------------------------------------------------------------- #
async def _exhaust_bangladesh(bot: PBDxBot, clock: Clock) -> None:
    """Start Bangladesh on a 1-minute interval, then let a refill report
    "0 added" so its file is known to be spent."""
    await _upload("bd.txt")
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 1})
    result = await otp_bot.start_automation()
    assert result["ok"], result

    bot.add_reply = SPENT_ADD
    clock.advance(2)
    mark = bot.mark()
    await otp_bot.run_cycle()
    assert bot.files(mark) == ["bd.txt"], "precondition: the refill that finds it spent"


async def test_a_refill_that_adds_nothing_marks_the_file_exhausted(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _exhaust_bangladesh(bot, clock)

    entries = await _active_bangladesh()
    assert len(entries) == 1
    assert entries[0].get("exhausted_at")


async def test_an_exhausted_file_is_not_re_sent_every_minute(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _exhaust_bangladesh(bot, clock)

    for minute in range(1, 6):
        clock.advance(1)
        mark = bot.mark()
        await otp_bot.run_cycle()
        assert bot.files(mark) == [], f"exhausted file re-sent {minute} min later"
        assert bot.add_commands(mark) == []


async def test_uploading_a_new_file_clears_the_exhausted_mark(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _exhaust_bangladesh(bot, clock)
    assert (await _active_bangladesh())[0].get("exhausted_at"), "precondition"

    bot.add_reply = FAST_ADD
    clock.advance(1)
    await _upload("bd2.txt")
    result = await otp_bot.start_automation()
    assert result["ok"], result

    entries = await _active_bangladesh()
    assert [e["name"] for e in entries] == ["bd2.txt"]
    assert not entries[0].get("exhausted_at")

    clock.advance(2)
    mark = bot.mark()
    result = await otp_bot.run_cycle()
    assert bot.files(mark) == ["bd2.txt"], (result.action, result.error)


# --------------------------------------------------------------------------- #
# 10. Stock parsing: names with brackets, accents and ampersands
# --------------------------------------------------------------------------- #
ODD_NAMES_ST_REPLY = (
    "\U0001F4CA Bot Statistics\n"
    "\n"
    "\U0001F4F1 Numbers:\n"
    "  • Available: 416036\n"
    "\n"
    f"{STOCK_HEADER}\n"
    "  \U0001F1E8\U0001F1E9 Congo (DRC): 120 (+3 taken)\n"
    "  \U0001F1E8\U0001F1EE Côte d'Ivoire: 50\n"
    "  \U0001F1E7\U0001F1E6 Bosnia & Herzegovina: 7 (+1 taken)\n"
    "  \U0001F1F9\U0001F1F7 Türkiye: 9\n"
    f"  {BD_FLAG} Bangladesh: 1234 (+5 taken)\n"
)


def test_stock_parser_reads_names_with_brackets_accents_and_ampersands():
    assert otp_bot._parse_country_stock(ODD_NAMES_ST_REPLY) == {
        "Congo (DRC)": 120,
        "Côte d'Ivoire": 50,
        "Bosnia & Herzegovina": 7,
        "Türkiye": 9,
        "Bangladesh": 1234,
    }


def test_stock_lookup_finds_congo_drc_as_the_bot_spells_it():
    stock = otp_bot._parse_country_stock(ODD_NAMES_ST_REPLY)
    assert otp_bot._stock_for(stock, "Congo (DRC)") == 120


# --------------------------------------------------------------------------- #
# 11. Concurrent cycles never interleave their sends
# --------------------------------------------------------------------------- #
def _cycle_blocks(tokens: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    for token in tokens:
        if token == "ST" or not blocks:
            blocks.append([token])
        else:
            blocks[-1].append(token)
    return blocks


async def test_concurrent_cycles_never_interleave_their_sends(clock):
    bot = PBDxBot(bangladesh=0)
    set_userbot(bot)
    await _upload_and_start_now()

    clock.advance(NEXT_CHECK)
    mark = bot.mark()
    bot.hold_first_stock_check = 1.5
    await asyncio.gather(
        otp_bot.run_cycle(force=True),
        otp_bot.run_cycle(force=True),
    )

    quota = await _quota_command()
    tokens: list[str] = []
    for kind, text, reply_to in bot.log[mark:]:
        if kind == "file":
            tokens.append("FILE")
        elif reply_to is not None:
            tokens.append("ADD")
        elif text.strip() == quota:
            tokens.append("ST")
        else:
            tokens.append(f"MSG {text.strip()}")

    # Every cycle here finds Bangladesh empty, so a whole cycle is: stock
    # check, file, add command. One cycle's sends must all land before the
    # other's begin; a second cycle that declines to run at all is fine too.
    blocks = _cycle_blocks(tokens)
    assert tokens, "neither cycle sent anything"
    assert 1 <= len(blocks) <= 2, f"sends interleaved: {tokens}"
    for block in blocks:
        assert block[0] == "ST" and block[-1] == "ADD" and block.count("FILE") == 1, (
            f"sends interleaved: {tokens}"
        )
