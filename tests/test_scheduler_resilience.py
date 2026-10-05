"""The scheduler loop around the OTP automation.

The OTP automation is the one job that has to keep running unattended, so the
loop that drives it must: survive its neighbours crashing, never hang on one
stuck Telegram call, look often enough that per-country timers are honoured,
and talk to the owner in clean English without crying wolf.

run_cycle() is replaced by a scripted fake throughout - what is under test is
the scheduler's reaction to a result, not the cycle itself (that lives in
test_otp_bot_automation.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.workers import scheduler_worker
from app.workers.scheduler_worker import SchedulerRunner


class _CapturingNotifier:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, chat_id: int, text: str, *, dedupe_key: str | None = None) -> dict:
        self.sent.append({"chat_id": chat_id, "text": text, "dedupe_key": dedupe_key})
        return {"sent": True}

    @property
    def texts(self) -> list[str]:
        return [n["text"] for n in self.sent]


class _Clock:
    """A hand-cranked utcnow() so pacing and reminder windows are testable."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(scheduler_worker, "utcnow", fake)
    return fake


@pytest.fixture
def otp_enabled(monkeypatch) -> None:
    async def enabled_config() -> dict[str, Any]:
        return {**otp_bot.DEFAULT_CONFIG, "enabled": True, "target_bot": "@otp_test_bot"}

    monkeypatch.setattr(otp_bot, "get_config", enabled_config)


def _script_cycles(monkeypatch, make_result) -> list[Any]:
    """Make run_cycle() return make_result() each time; returns the call log."""
    calls: list[Any] = []

    async def fake_run_cycle(config=None, *, force=False):
        calls.append(config)
        return make_result()

    monkeypatch.setattr(otp_bot, "run_cycle", fake_run_cycle)
    return calls


def _failed() -> otp_bot.CycleResult:
    return otp_bot.CycleResult(
        ok=False,
        action="error",
        error="telegram account not linked or errored: Use /tglogin first.",
    )


def _refilled() -> otp_bot.CycleResult:
    return otp_bot.CycleResult(
        ok=True,
        action="added (1 country)",
        country_stock={"Bangladesh": 0},
        add_reply="Bangladesh: ✅ 4821 added, 12 duplicates skipped",
    )


def _not_due() -> otp_bot.CycleResult:
    return otp_bot.CycleResult(ok=True, action="not due yet")


@dataclass
class _HeldResult(otp_bot.CycleResult):
    """CycleResult plus held_at_start, which otp_bot is gaining separately.

    CycleResult uses slots, so the field cannot simply be set on an instance;
    once otp_bot declares it, this redeclaration is harmless.
    """

    held_at_start: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. One crashing step must not starve the others
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("broken", ["tick", "maybe_send_briefing"])
async def test_a_crashing_step_does_not_starve_the_otp_automation(monkeypatch, broken):
    runner = SchedulerRunner()
    runner.interval = 0
    passes = {"count": 0, "otp": 0}

    async def fine() -> int:
        return 0

    async def crash() -> int:
        raise RuntimeError("malformed row")

    async def otp() -> bool:
        passes["otp"] += 1
        passes["count"] += 1
        if passes["count"] >= 3:
            runner._stop.set()
        return True

    async def counting_crash() -> int:
        # Also stops the loop, so the old shared-try code terminates and FAILS
        # instead of spinning forever.
        passes["count"] += 1
        if passes["count"] >= 3:
            runner._stop.set()
        raise RuntimeError("malformed row")

    monkeypatch.setattr(runner, "tick", fine)
    monkeypatch.setattr(runner, "maybe_send_briefing", fine)
    monkeypatch.setattr(runner, broken, counting_crash)
    monkeypatch.setattr(runner, "maybe_run_otp_automation", otp)

    await asyncio.wait_for(runner._loop(), timeout=5)

    assert passes["otp"] >= 1, f"a crashing {broken} must not stop the OTP automation"


async def test_cancellation_still_stops_the_loop(monkeypatch):
    runner = SchedulerRunner()
    runner.interval = 0

    async def cancelled() -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(runner, "tick", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(runner._loop(), timeout=5)


# --------------------------------------------------------------------------- #
# 2. A hung cycle is cut off, not allowed to freeze the scheduler
# --------------------------------------------------------------------------- #
async def test_the_cycle_timeout_is_generous_enough_for_a_real_cycle():
    # A cycle legitimately waits up to ~90 s per country file, ~15 countries.
    assert scheduler_worker.OTP_CYCLE_TIMEOUT_S >= 15 * 90


async def test_a_hung_cycle_is_cut_off_and_reported_as_a_failure(monkeypatch, otp_enabled, clock):
    monkeypatch.setattr(scheduler_worker, "OTP_CYCLE_TIMEOUT_S", 0.05)

    async def hung(config=None, *, force=False):
        await asyncio.sleep(3600)

    monkeypatch.setattr(otp_bot, "run_cycle", hung)
    notifier = _CapturingNotifier()
    runner = SchedulerRunner(notifier=notifier)

    ok = await asyncio.wait_for(runner.maybe_run_otp_automation(), timeout=5)

    assert ok is False
    assert len(notifier.sent) == 1
    assert "did not finish" in notifier.texts[0]


# --------------------------------------------------------------------------- #
# 3. Pacing: look every minute, starting on the first tick
# --------------------------------------------------------------------------- #
async def test_otp_is_looked_at_every_minute_from_the_first_tick(monkeypatch, otp_enabled, clock):
    async def thirty_minutes(base) -> int:
        return 30

    # Even with a 30-minute shortest interval the look stays at one minute:
    # the per-country due times decide what actually runs.
    monkeypatch.setattr(otp_schedule, "shortest_interval_minutes", thirty_minutes)
    calls = _script_cycles(monkeypatch, _not_due)
    runner = SchedulerRunner()

    await runner.maybe_run_otp_automation()
    assert len(calls) == 1, "the first tick after boot should already look"

    clock.advance(seconds=30)
    await runner.maybe_run_otp_automation()
    assert len(calls) == 1, "but not more than once a minute"

    clock.advance(seconds=31)
    await runner.maybe_run_otp_automation()
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# 4. Failure alerts: first one, a reminder every 30 min, one recovery note
# --------------------------------------------------------------------------- #
async def test_a_failure_streak_alerts_once_reminds_half_hourly_and_announces_recovery(
    monkeypatch, otp_enabled, clock
):
    state = {"ok": False}
    _script_cycles(monkeypatch, lambda: _refilled() if state["ok"] else _failed())
    notifier = _CapturingNotifier()
    runner = SchedulerRunner(notifier=notifier)

    async def minute() -> None:
        await runner.maybe_run_otp_automation()
        clock.advance(minutes=1)

    await minute()                                   # t=0: first failure
    assert len(notifier.sent) == 1
    assert "not linked" in notifier.texts[0]

    for _ in range(29):                              # t=1..29: silence
        await minute()
    assert len(notifier.sent) == 1, "a failure every minute must not mean an alert every minute"

    await minute()                                   # t=30: reminder
    assert len(notifier.sent) == 2
    assert "failing for 30 min" in notifier.texts[1]

    for _ in range(29):
        await minute()
    assert len(notifier.sent) == 2

    await minute()                                   # t=60: second reminder
    assert len(notifier.sent) == 3
    assert "failing for 1h" in notifier.texts[2]

    state["ok"] = True
    await minute()                                   # recovered
    recovered = [t for t in notifier.texts[3:] if t.startswith("✅")]
    assert len(recovered) == 1
    assert "running normally again" in recovered[0]

    before = len(notifier.sent)
    await minute()                                   # still fine: no second recovery note
    assert not [t for t in notifier.texts[before:] if t.startswith("✅")]

    state["ok"] = False
    before = len(notifier.sent)
    await minute()                                   # a NEW streak alerts straight away
    assert len(notifier.sent) == before + 1


async def test_an_idle_not_due_pass_is_not_mistaken_for_recovery(monkeypatch, otp_enabled, clock):
    """Between failing cycles the countries are simply not due; that proves
    nothing is fixed, and treating it as recovery would flap alert/recovered.
    """
    results = iter([_failed(), _not_due(), _not_due(), _failed()])
    _script_cycles(monkeypatch, lambda: next(results))
    notifier = _CapturingNotifier()
    runner = SchedulerRunner(notifier=notifier)

    for _ in range(4):
        await runner.maybe_run_otp_automation()
        clock.advance(minutes=1)

    assert len(notifier.sent) == 1
    assert not any(t.startswith("✅") for t in notifier.texts)


# --------------------------------------------------------------------------- #
# 5. A finished country that is HELD says how to resume it
# --------------------------------------------------------------------------- #
async def _notify_for(monkeypatch, result) -> list[str]:
    _script_cycles(monkeypatch, lambda: result)
    notifier = _CapturingNotifier()
    await SchedulerRunner(notifier=notifier).maybe_run_otp_automation()
    return notifier.texts


async def test_a_held_finish_says_the_file_is_kept_and_how_to_resume(monkeypatch, otp_enabled, clock):
    result = otp_bot.CycleResult(
        ok=True,
        action="skipped - every due country still has stock",
        finished=[{"country": "Senegal", "reason": "720 min shomoy shesh", "held": True,
                   "deleted": False, "error": ""}],
    )
    [message] = await _notify_for(monkeypatch, result)
    lines = message.split("\n")

    assert lines[0] == "\U0001F3C1 Run finished — file kept"
    assert lines[1] == ""
    assert "• Senegal: 12h run time reached" in lines
    assert "Nothing more will be added." in lines[-1]
    assert "▶️ Resume" in lines[-1] and "/otpbot" in lines[-1]
    assert "Per country" in lines[-1] and "Senegal" in lines[-1]
    # Sending the file again would duplicate a run that is merely paused.
    assert 'say "start"' not in message


async def test_a_finish_without_hold_keeps_the_send_a_file_instruction(monkeypatch, otp_enabled, clock):
    result = otp_bot.CycleResult(
        ok=True,
        action="skipped - every due country still has stock",
        finished=[{"country": "Nigeria", "reason": "3 bar re-add shesh", "deleted": True, "error": ""}],
    )
    [message] = await _notify_for(monkeypatch, result)
    lines = message.split("\n")

    assert lines[0] == "\U0001F3C1 Run finished"
    assert "• Nigeria: 3 re-adds done" in lines
    assert "Resume" not in message
    assert lines[-1] == 'To run it again, send the file and say "start".'


# --------------------------------------------------------------------------- #
# 6. held_at_start: start time arrived but stock is still there
# --------------------------------------------------------------------------- #
async def test_a_start_held_for_stock_is_reported_once(monkeypatch, otp_enabled, clock):
    result = _HeldResult(
        ok=True,
        action="skipped - every due country still has stock",
        held_at_start=[{"country": "Malaysia", "stock": 11068, "count": 4}],
    )

    [message] = await _notify_for(monkeypatch, result)

    assert message.startswith("⏸")
    lines = message.split("\n")
    assert lines[0].endswith("Scheduled start — file held")
    assert lines[1:] == [
        "",
        "• Malaysia: 11,068 numbers still in stock",
        "",
        "It will be added automatically once stock runs low.",
    ]


async def test_a_result_without_held_at_start_still_works(monkeypatch, otp_enabled, clock):
    class _Bare:
        """A result shaped like today's CycleResult: no held_at_start at all."""

        ok = True
        action = "skipped - every due country still has stock"
        error = ""
        active_quota = None
        started_now: list = []
        finished: list = []
        exhausted: list = []
        ran_at = "2026-10-05T12:00:00+00:00"

    assert not hasattr(_Bare(), "held_at_start")
    assert await _notify_for(monkeypatch, _Bare()) == []


# --------------------------------------------------------------------------- #
# Every notification is clean English with the same layout
# --------------------------------------------------------------------------- #
_BANGLISH = re.compile(
    r"\b(shesh|hoyeche|hoyni|kora|holo|abar|chalate|ekhono|rakha|bolo|dao|"
    r"nai|ache|lagbe|pathao|onujayi|shuru|korbo|dibo|chesta|theke|hoye|geche|"
    r"shomoy|bar)\b",
    re.IGNORECASE,
)


def _assert_clean_english(message: str) -> None:
    # Quoted text is a command the owner types back, not prose.
    prose = re.sub(r'"[^"]*"', "", message)
    assert not _BANGLISH.search(prose), message
    lines = message.split("\n")
    assert lines[1] == "", f"header then a blank line: {message!r}"
    assert any(line.startswith("• ") for line in lines), message


async def test_every_success_notification_is_tidy_english(monkeypatch, otp_enabled, clock):
    result = _HeldResult(
        ok=True,
        action="added (2 countries)",
        country_stock={"Bangladesh": 0, "Nigeria": 0},
        held_at_start=[{"country": "Malaysia", "stock": 11068, "count": 4}],
        exhausted=[{"country": "Nigeria", "name": "ng.txt", "had_stock": 1234}],
        finished=[
            {"country": "Senegal", "reason": "720 min shomoy shesh", "held": True,
             "deleted": True, "error": ""},
            {"country": "Ghana", "reason": "18:00 (Dubai) time hoye geche", "deleted": False,
             "error": "no delete command configured"},
        ],
        started_now=[
            {"country": "Kenya", "count": 5000, "tag": "WhatsApp", "added": True, "error": ""},
            {"country": "Peru", "count": 20, "tag": None, "added": False, "error": "flood wait"},
        ],
    )

    messages = await _notify_for(monkeypatch, result)

    assert len(messages) == 4      # started, held, finished, exhausted
    for message in messages:
        _assert_clean_english(message)
    joined = "\n".join(messages)
    assert "• Kenya: 5,000 numbers added (tag: WhatsApp)" in joined
    assert "1,234" in joined
    assert '"remove Nigeria"' in joined


async def test_the_refill_notification_is_tidy_english(monkeypatch, otp_enabled, clock):
    [message] = await _notify_for(monkeypatch, _refilled())

    _assert_clean_english(message)
    assert "• Bangladesh: 4,821 added, 12 duplicates skipped" in message.split("\n")
    assert "@otp_test_bot" in message


async def test_a_multi_line_bot_reply_still_gives_one_bullet_per_country(monkeypatch, otp_enabled, clock):
    result = otp_bot.CycleResult(
        ok=True,
        action="added (2 countries)",
        country_stock={"Bangladesh": 0, "Nigeria": 3},
        add_reply=(
            "Bangladesh: ⚡ Fast Add Complete!\n\nBangladesh: 500 added (5 dup)\n"
            "Nigeria: ⚡ Fast Add Complete!\n\nNigeria: 1200 added (0 dup)"
        ),
    )
    [message] = await _notify_for(monkeypatch, result)

    bullets = [line for line in message.split("\n") if line.startswith("•")]
    assert bullets == [
        "• Bangladesh: 500 added, 5 duplicates skipped",
        "• Nigeria: 1,200 added",
    ]


async def test_failure_reminder_and_recovery_are_tidy_english(monkeypatch, otp_enabled, clock):
    state = {"ok": False}
    _script_cycles(monkeypatch, lambda: _refilled() if state["ok"] else _failed())
    notifier = _CapturingNotifier()
    runner = SchedulerRunner(notifier=notifier)

    for _ in range(31):
        await runner.maybe_run_otp_automation()
        clock.advance(minutes=1)
    state["ok"] = True
    await runner.maybe_run_otp_automation()

    alert, reminder, recovery = notifier.texts[:3]
    for message in (alert, reminder, recovery):
        _assert_clean_english(message)
    assert recovery.startswith("✅ OTP-bot is running normally again")


async def test_a_partial_failure_still_sends_the_per_country_notices(monkeypatch, otp_enabled, clock):
    """ok is False as soon as ONE country's add fails, while the rest of the
    cycle still ran and persisted exhausted_at / finished_at. Returning early
    on ok=False swallowed the "file used up" and "run finished" notices for
    good - nothing would ever announce them again.
    """
    result = otp_bot.CycleResult(
        ok=False,
        action="error",
        error="Peru: flood wait",
        exhausted=[{"country": "Nigeria", "name": "ng.txt", "had_stock": 0}],
        finished=[{"country": "Senegal", "reason": "30 min run time reached", "held": True,
                   "deleted": False, "error": ""}],
    )
    _script_cycles(monkeypatch, lambda: result)
    notifier = _CapturingNotifier()

    ok = await SchedulerRunner(notifier=notifier).maybe_run_otp_automation()

    assert ok is False
    texts = notifier.texts
    assert len(texts) == 3, texts
    failure = [t for t in texts if "OTP-bot automation failed" in t.split("\n")[0]]
    assert len(failure) == 1 and "flood wait" in failure[0]
    finished = [t for t in texts if t.startswith("\U0001F3C1 Run finished")]
    assert len(finished) == 1
    assert "• Senegal: 30 min run time reached" in finished[0].split("\n")
    used_up = [t for t in texts if "new file needed" in t.split("\n")[0]]
    assert len(used_up) == 1 and "Nigeria" in used_up[0]
    for message in texts:
        _assert_clean_english(message)


@pytest.mark.parametrize(
    ("reason", "shown"),
    [
        # Today's English reasons pass through untouched...
        ("2 re-adds done", "2 re-adds done"),
        ("23:30 (Dubai) stop time reached", "23:30 (Dubai) stop time reached"),
        ("30 min run time reached", "30 min run time reached"),
        # ...and a run that finished before the switch still reads as English.
        ("2 bar re-add shesh", "2 re-adds done"),
        ("1 bar re-add shesh", "1 re-add done"),
        ("23:30 (Dubai) time hoye geche", "23:30 (Dubai) stop time reached"),
        ("720 min shomoy shesh", "12h run time reached"),
    ],
)
def test_finish_reasons_always_read_as_english(reason, shown):
    assert scheduler_worker._english_reason(reason) == shown


# --------------------------------------------------------------------------- #
# 7. Health: a dead scheduler loop is not "running"
# --------------------------------------------------------------------------- #
class _Holder:
    def __init__(self, task: asyncio.Task | None) -> None:
        self._task = task


async def _dead_task() -> asyncio.Task:
    async def crash() -> None:
        raise RuntimeError("loop blew up")

    task = asyncio.create_task(crash())
    with contextlib.suppress(RuntimeError):
        await task
    return task


async def test_a_dead_scheduler_loop_is_not_reported_as_running(monkeypatch):
    from app import monitoring

    monkeypatch.setitem(monitoring.RUNTIME, "scheduler", _Holder(await _dead_task()))

    assert "scheduler" not in monitoring.workers_state()["running"]
    snapshot = await monitoring.health_snapshot()
    assert snapshot["checks"]["scheduler"]["ok"] is False
    assert "loop blew up" in snapshot["checks"]["scheduler"]["error"]
    assert snapshot["ok"] is False


async def test_a_live_scheduler_loop_is_reported_as_running(monkeypatch):
    from app import monitoring

    alive = asyncio.create_task(asyncio.sleep(3600))
    try:
        monkeypatch.setitem(monitoring.RUNTIME, "scheduler", _Holder(alive))
        assert "scheduler" in monitoring.workers_state()["running"]
        snapshot = await monitoring.health_snapshot()
        assert snapshot["checks"]["scheduler"]["ok"] is True
    finally:
        alive.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await alive


async def test_a_scheduler_that_was_never_started_is_not_a_failure():
    from app import monitoring

    snapshot = await monitoring.health_snapshot()
    assert snapshot["checks"]["scheduler"]["ok"] is True
