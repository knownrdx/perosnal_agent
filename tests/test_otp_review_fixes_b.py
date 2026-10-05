"""Review fixes, part B: add results credited to the right country, and no
Telegram call (or lock) allowed to hang the automation forever.

The fake @PBDxbot here runs on a SIMULATED clock: otp_bot._sleep advances it,
so "the result arrives 30 s after the add" means 30 s of the automation's own
polling, not 30 real seconds. Bot posts are scheduled at a time and only show
up in read_messages once the clock has reached it - which is what lets these
tests reproduce a slow add whose answer lands during the NEXT country's wait.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.integrations.telegram_user import UserbotError, set_userbot

DUBAI = otp_schedule.DUBAI_TZ
STOCK_HEADER = "\U0001F30D Country Stock (yours):"
FLAGS = {"Bangladesh": "\U0001F1E7\U0001F1E9", "Central African Republic": "\U0001F1E8\U0001F1EB"}
CAR = "Central African Republic"
NEXT_CHECK = 11         # minutes: one past the default 10-minute interval

PROGRESS = "⏳ Processing..."
BD_SPENT = "⚡ Fast Add Complete!\n\nBangladesh: 0 added (5 dup)"
BD_FRESH = "⚡ Fast Add Complete!\n\nBangladesh: 3501 added (16499 dup)"
CAR_FRESH = "⚡ Fast Add Complete!\n\n\U0001F1E8\U0001F1EB Central African Republic: 100000 added"
PLAIN_FRESH = "✅ 53412 added, 46588 duplicates skipped"
PLAIN_SPENT = "✅ 0 added, 100000 duplicates skipped"


# --------------------------------------------------------------------------- #
# Clock (schedule side) - same as test_otp_owner_scenarios
# --------------------------------------------------------------------------- #
def dubai(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=DUBAI).astimezone(timezone.utc)


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock(dubai(5, 10, 0))
    monkeypatch.setattr(otp_schedule, "_now", fake)
    monkeypatch.setattr(otp_bot, "_now_iso", lambda: fake.now.isoformat(), raising=False)
    return fake


# --------------------------------------------------------------------------- #
# Fake @PBDxbot on a simulated clock
# --------------------------------------------------------------------------- #
class TimelineBot:
    """Answers like the real bot, with every bot post scheduled in time.

    ``add_scripts[country]`` is what the bot says after an add for that
    country: a list of (delay_s, text) posts, or (delay_s, text, "edit") to
    edit the script's previous post in place. A country without a script
    gets an immediate "<country>: 100 added".
    """

    def __init__(self, stock: dict[str, int] | None = None) -> None:
        self.now = 0.0
        self.stock = dict(stock or {"Bangladesh": 0, CAR: 0})
        self.chat: list[dict[str, Any]] = []        # oldest first
        self.add_scripts: dict[str, list[tuple]] = {}
        self.hang: set[str] = set()                 # methods that never return
        self.calls: list[str] = []
        self.files_sent: list[str] = []
        self._ids = 0
        self._seq = 0
        self._due: list[dict[str, Any]] = []
        self._file_country: str | None = None

    # -- clock ----------------------------------------------------------- #
    async def advance(self, seconds: float) -> None:
        self.now += seconds
        self._release()

    def _new_id(self) -> int:
        self._ids += 1
        return self._ids

    def _schedule(self, script: list[tuple]) -> None:
        holder: dict[str, int | None] = {"id": None}
        for item in script:
            self._seq += 1
            self._due.append({
                "at": self.now + item[0],
                "seq": self._seq,
                "text": item[1],
                "edit": len(item) > 2 and item[2] == "edit",
                "holder": holder,
            })

    def _release(self) -> None:
        ready = sorted(
            (d for d in self._due if d["at"] <= self.now),
            key=lambda d: (d["at"], d["seq"]),
        )
        for item in ready:
            self._due.remove(item)
            holder = item["holder"]
            if item["edit"] and holder["id"] is not None:
                for message in self.chat:
                    if message["id"] == holder["id"]:
                        message["text"] = item["text"]
            else:
                holder["id"] = self._new_id()
                self.chat.append({"id": holder["id"], "text": item["text"], "out": False})

    def stock_text(self) -> str:
        lines = ["\U0001F4CA Bot Statistics", "", "  • Available: 416036", "", STOCK_HEADER]
        for country, count in self.stock.items():
            lines.append(f"  {FLAGS.get(country, '')} {country}: {count} (+5 taken)")
        return "\n".join(lines)

    async def _maybe_hang(self, method: str) -> None:
        self.calls.append(method)
        await asyncio.sleep(0)
        if method in self.hang:
            await asyncio.Event().wait()

    # -- userbot surface -------------------------------------------------- #
    async def send_message(self, target: str, text: str, reply_to: int | None = None) -> dict[str, Any]:
        await self._maybe_hang("send_message")
        self._release()
        message_id = self._new_id()
        self.chat.append({"id": message_id, "text": text, "out": True})
        head = text.strip().split()[0] if text.strip() else ""
        if reply_to is not None or head == "/fan":
            country = self._file_country or "Bangladesh"
            if country in self.add_scripts:
                self._schedule(self.add_scripts[country])
            else:
                self._schedule([(0, f"⚡ Fast Add Complete!\n\n{country}: 100 added")])
        elif head == "/st":
            self._schedule([(0, self.stock_text())])
        else:
            self._schedule([(0, "✅ OK")])
        self._release()
        return {"sent": True, "message_id": message_id, "to": target}

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        await self._maybe_hang("send_file")
        self._release()
        name = Path(path).name
        self.files_sent.append(name)
        stem = name.rsplit(".", 1)[0]
        self._file_country = stem.split("__", 1)[1].replace("_", " ") if "__" in stem else "Bangladesh"
        message_id = self._new_id()
        self.chat.append({"id": message_id, "text": f"[file {name}]", "out": True})
        return {"sent": True, "message_id": message_id, "to": target}

    async def read_messages(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        await self._maybe_hang("read_messages")
        self._release()
        return [dict(m) for m in reversed(self.chat)][:limit]

    async def delete_messages(self, target: str, message_ids: list[int]) -> dict[str, Any]:
        await self._maybe_hang("delete_messages")
        gone = set(message_ids)
        self.chat = [m for m in self.chat if m["id"] not in gone]
        return {"deleted": len(gone)}


@pytest.fixture
def bot(monkeypatch) -> TimelineBot:
    fake = TimelineBot()
    set_userbot(fake)
    monkeypatch.setattr(otp_bot, "_sleep", fake.advance)
    yield fake
    set_userbot(None)


async def _upload_mixed(name: str = "mix.txt") -> list[dict[str, Any]]:
    """One file holding Bangladesh and Central African Republic numbers -
    split by enqueue_file into one queue entry per country."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(
        "+8801711111111\n+8801722222222\n+23670111111\n+23672222222\n",
        encoding="utf-8",
    )
    return await otp_bot.enqueue_file(rel_path(path), name, "whatsapp 20h")


async def _start_both(bot: TimelineBot) -> None:
    await _upload_mixed()
    result = await otp_bot.start_automation()
    assert result["ok"], result
    countries = sorted(e.get("country") for e in await otp_bot.get_active_files())
    assert countries == ["Bangladesh", CAR], countries


async def _entry(country: str) -> dict[str, Any]:
    entries = [e for e in await otp_bot.get_active_files() if e.get("country") == country]
    assert len(entries) == 1, entries
    return entries[0]


async def _refill_both(bot: TimelineBot, clock: Clock) -> otp_bot.CycleResult:
    clock.advance(NEXT_CHECK)
    before = len(bot.files_sent)
    result = await otp_bot.run_cycle()
    sent = bot.files_sent[before:]
    assert len(sent) == 2, (sent, result.action, result.error)
    return result


# --------------------------------------------------------------------------- #
# 1a. A progress line that sits unchanged is not the add's result
# --------------------------------------------------------------------------- #
async def test_a_slow_add_is_credited_to_its_own_country(bot, clock):
    """Bangladesh's real answer comes 30 s after its progress line; the old
    20 s settle took "Processing..." as the result, and the late "Bangladesh:
    0 added" was then read as Central African Republic's answer - marking a
    good file spent and leaving the spent one to be re-sent forever."""
    await _start_both(bot)
    bot.add_scripts = {
        "Bangladesh": [(0, PROGRESS), (30, BD_SPENT)],
        CAR: [(10, CAR_FRESH)],
    }

    result = await _refill_both(bot, clock)

    assert (await _entry("Bangladesh")).get("exhausted_at"), result.add_reply
    assert not (await _entry(CAR)).get("exhausted_at"), result.add_reply
    assert [e["country"] for e in result.exhausted] == ["Bangladesh"]
    assert f"Bangladesh: {BD_SPENT}" in result.add_reply, result.add_reply
    assert f"{CAR}: {CAR_FRESH}" in result.add_reply, result.add_reply


async def test_a_progress_line_edited_into_the_result_is_read_as_the_result(bot, clock):
    await _start_both(bot)
    bot.add_scripts = {
        "Bangladesh": [(0, PROGRESS), (30, BD_FRESH, "edit")],
        CAR: [(0, CAR_FRESH)],
    }

    result = await _refill_both(bot, clock)

    assert "3501 added" in result.add_reply, result.add_reply
    assert not (await _entry("Bangladesh")).get("exhausted_at")


async def test_add_one_file_waits_past_an_unchanged_progress_line(bot, clock):
    await _start_both(bot)
    bot.add_scripts = {"Bangladesh": [(0, PROGRESS), (40, BD_FRESH)]}
    entry = await _entry("Bangladesh")

    reply = await otp_bot._add_one_file("@PBDxbot", entry, await otp_bot.get_config())

    assert reply == BD_FRESH


async def test_an_add_that_never_finishes_still_returns_the_last_thing_said(bot, clock):
    await _start_both(bot)
    bot.add_scripts = {"Bangladesh": [(0, PROGRESS)]}
    entry = await _entry("Bangladesh")

    reply = await otp_bot._add_one_file("@PBDxbot", entry, await otp_bot.get_config())

    assert reply == PROGRESS


# --------------------------------------------------------------------------- #
# 1b. The add matcher knows which country it is waiting for
# --------------------------------------------------------------------------- #
async def test_another_countrys_late_result_is_not_taken_as_this_ones(bot, clock):
    await _start_both(bot)
    # Bangladesh's answer to an EARLIER add lands first, then CAR's own.
    bot.add_scripts = {CAR: [(0, BD_SPENT), (15, CAR_FRESH)]}
    entry = await _entry(CAR)

    reply = await otp_bot._add_one_file("@PBDxbot", entry, await otp_bot.get_config())

    assert reply == CAR_FRESH


async def test_a_result_without_per_country_lines_matches_any_country(bot, clock):
    await _start_both(bot)
    bot.add_scripts = {CAR: [(0, PROGRESS), (6, PLAIN_FRESH)]}
    entry = await _entry(CAR)

    reply = await otp_bot._add_one_file("@PBDxbot", entry, await otp_bot.get_config())

    assert reply == PLAIN_FRESH


async def test_a_plain_zero_added_reply_still_marks_the_file_spent(bot, clock):
    await _start_both(bot)
    bot.add_scripts = {"Bangladesh": [(3, PLAIN_SPENT)], CAR: [(3, PLAIN_FRESH)]}

    await _refill_both(bot, clock)

    assert (await _entry("Bangladesh")).get("exhausted_at")
    assert not (await _entry(CAR)).get("exhausted_at")


# --------------------------------------------------------------------------- #
# 1c. Exhaustion is decided from THIS country's count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "country", "expected"),
    [
        (BD_SPENT, "Bangladesh", 0),
        (BD_FRESH, "Bangladesh", 3501),
        (CAR_FRESH, CAR, 100000),
        (CAR_FRESH, "Bangladesh", None),
        (BD_SPENT, CAR, None),
        (PLAIN_FRESH, "Bangladesh", 53412),
        (PLAIN_SPENT, CAR, 0),
        ("Bangladesh: 0 added (5 dup)\n\U0001F1E8\U0001F1EB Central African Republic: 1,200 added", CAR, 1200),
        (PROGRESS, "Bangladesh", None),
    ],
)
def test_added_for_reads_this_countrys_line(text, country, expected):
    assert otp_bot._added_for(text, country) == expected


async def test_another_countrys_zero_never_marks_this_file_spent(bot, clock):
    """Bangladesh's add never finishes inside its window; its "0 added" turns
    up while CAR is waiting and CAR's own answer never comes. CAR is handed
    that text as a last resort - but it is not CAR's count, so CAR's file
    must not be written off."""
    await _start_both(bot)
    bot.add_scripts = {
        "Bangladesh": [(0, PROGRESS), (100, BD_SPENT)],
        CAR: [],
    }

    result = await _refill_both(bot, clock)

    assert not (await _entry(CAR)).get("exhausted_at"), result.add_reply
    assert CAR not in [e["country"] for e in result.exhausted]


# --------------------------------------------------------------------------- #
# 2. Deadlines: no Telegram call and no lock wait is unbounded
# --------------------------------------------------------------------------- #
def test_deadline_defaults():
    assert otp_bot._BOT_CALL_TIMEOUT_S == 120
    assert otp_bot._SEND_FILE_TIMEOUT_S == 300
    assert otp_bot._CYCLE_LOCK_WAIT_S == 600


async def test_bot_call_turns_a_timeout_into_a_userbot_error():
    async def never() -> None:
        await asyncio.Event().wait()

    with pytest.raises(UserbotError, match="Telegram did not answer within"):
        await otp_bot._bot_call(never(), 0.01)


async def test_bot_call_passes_results_and_errors_through():
    async def answer() -> int:
        return 7

    async def broken() -> None:
        raise UserbotError("not linked")

    assert await otp_bot._bot_call(answer(), 1) == 7
    with pytest.raises(UserbotError, match="not linked"):
        await otp_bot._bot_call(broken(), 1)


async def test_every_userbot_call_goes_through_the_deadline(bot, clock, monkeypatch):
    seen: list[tuple[str, float]] = []
    real = otp_bot._bot_call

    async def spy(awaitable, timeout_s):
        seen.append((getattr(awaitable, "__qualname__", "").rsplit(".", 1)[-1], timeout_s))
        return await real(awaitable, timeout_s)

    monkeypatch.setattr(otp_bot, "_bot_call", spy)
    await _start_both(bot)
    await otp_bot.refresh_countries_from_bot()
    clock.advance(NEXT_CHECK)
    await otp_bot.run_cycle()

    assert len(seen) == len(bot.calls), (seen, bot.calls)
    assert {name for name, _ in seen} >= {"send_file", "send_message", "read_messages", "delete_messages"}
    for name, timeout_s in seen:
        assert timeout_s == (300 if name == "send_file" else 120), (name, timeout_s)


async def test_a_hung_call_in_refresh_gives_up_and_frees_the_lock(bot, monkeypatch):
    monkeypatch.setattr(otp_bot, "_BOT_CALL_TIMEOUT_S", 0.05)
    bot.hang.add("send_message")

    result = await asyncio.wait_for(otp_bot.refresh_countries_from_bot(), 5)

    assert result["ok"] is False
    assert "did not answer" in result["error"], result
    assert not otp_bot._conversation_lock().locked()


async def test_a_hung_call_in_a_cycle_gives_up_and_frees_the_lock(bot, clock, monkeypatch):
    await _start_both(bot)
    monkeypatch.setattr(otp_bot, "_BOT_CALL_TIMEOUT_S", 0.05)
    bot.hang.add("read_messages")

    result = await asyncio.wait_for(otp_bot.run_cycle(force=True), 5)

    assert result.ok is False
    assert "did not answer" in result.error, result.error
    assert not otp_bot._conversation_lock().locked()


async def test_a_hung_file_upload_in_start_gives_up_and_keeps_the_queue(bot, monkeypatch):
    monkeypatch.setattr(otp_bot, "_SEND_FILE_TIMEOUT_S", 0.05)
    bot.hang.add("send_file")
    await _upload_mixed()

    result = await asyncio.wait_for(otp_bot.start_automation(), 5)

    assert result["ok"] is False
    assert "did not answer" in result["error"], result
    assert len(await otp_bot.get_queue()) == 2
    assert not otp_bot._conversation_lock().locked()


async def test_a_cycle_reports_busy_instead_of_waiting_forever(bot, clock, monkeypatch):
    await _start_both(bot)
    monkeypatch.setattr(otp_bot, "_CYCLE_LOCK_WAIT_S", 0.05)
    sent_before = len(bot.calls)

    async with otp_bot._conversation_lock():
        result = await asyncio.wait_for(otp_bot.run_cycle(force=True), 5)

    assert result.ok is False
    assert result.error == "busy: another operation with the bot is still running"
    assert bot.calls[sent_before:] == []
    last = await otp_bot.get_last_result()
    assert last and last["error"].startswith("busy"), last
    assert not otp_bot._conversation_lock().locked()


async def test_a_cycle_that_gets_the_lock_in_time_runs_normally(bot, clock, monkeypatch):
    await _start_both(bot)
    monkeypatch.setattr(otp_bot, "_CYCLE_LOCK_WAIT_S", 5)
    lock = otp_bot._conversation_lock()
    await lock.acquire()
    clock.advance(NEXT_CHECK)
    task = asyncio.create_task(otp_bot.run_cycle())
    await asyncio.sleep(0.05)
    assert not task.done()
    lock.release()

    result = await asyncio.wait_for(task, 5)

    assert result.ok, (result.action, result.error)
    assert not lock.locked()


async def test_start_still_waits_for_the_lock_however_long(bot, monkeypatch):
    """start/refresh are owner-initiated: they wait their turn rather than
    being turned away by the cycle's limit."""
    monkeypatch.setattr(otp_bot, "_CYCLE_LOCK_WAIT_S", 0.01)
    await _upload_mixed()
    lock = otp_bot._conversation_lock()
    await lock.acquire()
    task = asyncio.create_task(otp_bot.start_automation())
    await asyncio.sleep(0.1)
    assert not task.done()
    lock.release()

    result = await asyncio.wait_for(task, 5)

    assert result["ok"], result
    assert not lock.locked()
