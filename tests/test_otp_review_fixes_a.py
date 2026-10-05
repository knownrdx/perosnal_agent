"""Review fixes for the OTP-bot automation (batch A).

1. A start cut short by an error must not drop the files it never reached,
   nor wipe files uploaded while it was running.
2. A resume that fails must not leave copies of the active files queued.
3. Resuming a country whose limit passed while it was paused begins a fresh
   run, instead of finishing and re-pausing it on the next pass.
4. Pausing/resuming from chat or the API goes through set_paused.
5. The one-time v2 migration also turns off per-country delete_when_done.

The fake target bot follows tests/test_otp_owner_scenarios.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.automation import otp_bot, otp_schedule
from app.integrations.telegram_user import UserbotError, set_userbot

DUBAI = otp_schedule.DUBAI_TZ

BD_FLAG = "\U0001F1E7\U0001F1E9"
STOCK_HEADER = "\U0001F30D Country Stock (yours):"
FAST_ADD = "⚡ Fast Add Complete!\n\nBangladesh: 3501 added (16499 dup)"

NUMBERS = {
    "Bangladesh": "+8801711111111\n+8801722222222\n+8801733333333\n",
    "Nigeria": "+2348031111111\n+2348032222222\n",
    "Kenya": "+254711111111\n+254722222222\n",
}


# --------------------------------------------------------------------------- #
# Clock
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
    """Answers like the real bot and records every send, in order."""

    def __init__(self, stock: dict[str, int] | None = None, add_reply: str = FAST_ADD) -> None:
        self.stock: dict[str, int] = stock if stock is not None else {"Bangladesh": 0}
        self.add_reply = add_reply
        self.log: list[tuple[str, str, int | None]] = []
        self._next_message_id = 1000
        self._reply_id = 0
        self._current: dict[str, Any] | None = None

    def stock_text(self) -> str:
        lines = ["\U0001F4CA Bot Statistics", "", STOCK_HEADER]
        for country, count in self.stock.items():
            flag = BD_FLAG if country == "Bangladesh" else "\U0001F3F3️"
            lines.append(f"  {flag} {country}: {count} (+5 taken)")
        return "\n".join(lines)

    def _answer(self, text: str, reply_to: int | None) -> str:
        head = text.strip().split()[0] if text.strip() else ""
        if reply_to is not None or head == "/fan":
            return self.add_reply
        if head == "/st":
            return self.stock_text()
        return "✅ OK"

    async def send_message(self, target: str, text: str, reply_to: int | None = None) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.log.append(("msg", text, reply_to))
        self._next_message_id += 1
        self._reply_id += 1
        self._current = {"id": self._reply_id, "text": self._answer(text, reply_to), "out": False}
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
        return {"deleted": len(message_ids)}

    def files(self) -> list[str]:
        return [Path(p).name for kind, p, _ in self.log if kind == "file"]


class DropsOnFile(PBDxBot):
    """The Telegram connection dies while sending one particular file."""

    def __init__(self, fail_on: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.fail_on = fail_on

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        if Path(path).name == self.fail_on:
            raise UserbotError("connection lost")
        return await super().send_file(target, path, caption, reply_to)


class UploadDuringStart(PBDxBot):
    """The owner drops another file while the start is busy adding."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.uploaded = False

    async def send_file(self, target: str, path: str, caption: str = "", reply_to: int | None = None) -> dict[str, Any]:
        if not self.uploaded:
            self.uploaded = True
            await _upload("Nigeria", "late.txt")
        return await super().send_file(target, path, caption, reply_to)


class Unreachable(PBDxBot):
    """The account is logged out: every call fails."""

    async def send_message(self, *a: Any, **kw: Any) -> dict[str, Any]:
        raise UserbotError("not authorised")

    async def send_file(self, *a: Any, **kw: Any) -> dict[str, Any]:
        raise UserbotError("not authorised")

    async def read_messages(self, *a: Any, **kw: Any) -> list[dict[str, Any]]:
        raise UserbotError("not authorised")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _upload(country: str, name: str, caption: str = "whatsapp") -> dict[str, Any]:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(NUMBERS[country], encoding="utf-8")
    return await otp_bot.enqueue_file(rel_path(path), name, caption)


def _countries(entries: list[dict[str, Any]]) -> list[str]:
    return [e.get("country") or e.get("name") for e in entries]


# --------------------------------------------------------------------------- #
# 1. start: abort mid-way / uploads during the start
# --------------------------------------------------------------------------- #
async def test_a_start_cut_short_keeps_the_files_it_never_reached_queued(clock):
    bot = DropsOnFile("ng.txt", stock={"Bangladesh": 0, "Nigeria": 0, "Kenya": 0})
    set_userbot(bot)
    await _upload("Bangladesh", "bd.txt")
    await _upload("Nigeria", "ng.txt")
    await _upload("Kenya", "ke.txt")

    result = await otp_bot.start_automation()

    assert bot.files() == ["bd.txt"], "precondition: only Bangladesh went through"
    # The one that broke and the one never reached both wait for a retry.
    assert sorted(_countries(await otp_bot.get_queue())) == ["Kenya", "Nigeria"]
    # Bangladesh is on the bot, so it is monitored and the start says so.
    assert result["ok"] is True, result
    assert result.get("partial") is True
    assert "connection lost" in result["error"]
    assert _countries(await otp_bot.get_active_files()) == ["Bangladesh"]
    assert await otp_schedule.get_run_state("Bangladesh")
    assert (await otp_bot.get_config())["enabled"] is True
    assert sorted(_countries(result["failed"])) == ["Kenya", "Nigeria"]


async def test_a_file_uploaded_while_a_start_is_running_stays_queued(clock):
    bot = UploadDuringStart(stock={"Bangladesh": 0})
    set_userbot(bot)
    await _upload("Bangladesh", "bd.txt")

    result = await otp_bot.start_automation()

    assert result["ok"] is True, result
    assert bot.uploaded, "precondition: the upload landed mid-start"
    assert _countries(await otp_bot.get_active_files()) == ["Bangladesh"]
    queue = await otp_bot.get_queue()
    assert [e["name"] for e in queue] == ["late.txt"]


async def test_a_clean_start_still_empties_the_queue(clock):
    bot = PBDxBot(stock={"Bangladesh": 0, "Nigeria": 0})
    set_userbot(bot)
    await _upload("Bangladesh", "bd.txt")
    await _upload("Nigeria", "ng.txt")

    result = await otp_bot.start_automation()

    assert result["ok"] is True, result
    assert not result.get("partial")
    assert await otp_bot.get_queue() == []
    assert sorted(_countries(await otp_bot.get_active_files())) == ["Bangladesh", "Nigeria"]


# --------------------------------------------------------------------------- #
# 2. resume that fails leaves no duplicates behind
# --------------------------------------------------------------------------- #
async def test_a_failed_resume_takes_its_copies_back_out_of_the_queue(clock):
    set_userbot(PBDxBot(stock={"Bangladesh": 0}))
    await _upload("Bangladesh", "bd.txt")
    assert (await otp_bot.start_automation())["ok"]
    # Something the owner queued since, which a failed resume must not touch.
    waiting = (await _upload("Nigeria", "ng.txt"))["entries"][0]

    set_userbot(Unreachable())
    reply = await otp_bot.handle_resume_trigger()

    assert "Couldn't resume" in reply
    assert [e["id"] for e in await otp_bot.get_queue()] == [waiting["id"]]
    assert _countries(await otp_bot.get_active_files()) == ["Bangladesh"]


async def test_a_failed_resume_with_nothing_else_queued_leaves_it_empty(clock):
    set_userbot(PBDxBot(stock={"Bangladesh": 0}))
    await _upload("Bangladesh", "bd.txt")
    assert (await otp_bot.start_automation())["ok"]

    set_userbot(Unreachable())
    reply = await otp_bot.handle_resume_trigger()

    assert "Couldn't resume" in reply
    assert await otp_bot.get_queue() == []


# --------------------------------------------------------------------------- #
# 3. set_paused: limit passed while paused
# --------------------------------------------------------------------------- #
async def test_resuming_after_the_limit_passed_while_paused_starts_a_fresh_run(clock):
    # Live "Senegal": paused mid-run, its 12h went by, never marked finished.
    await otp_schedule.set_country_settings("Senegal", {"run_minutes": 720})
    await otp_schedule.begin_run("Senegal")
    await otp_schedule.set_paused("Senegal", True)
    clock.advance(721)
    base = await otp_bot.get_config()
    assert await otp_schedule.finished_reason("Senegal", base), "precondition"
    assert not (await otp_schedule.get_run_state("Senegal")).get("finished_at")

    await otp_schedule.set_paused("Senegal", False)

    state = await otp_schedule.get_run_state("Senegal")
    assert state["started_at"] == clock.now.isoformat()
    assert int(state["refills"]) == 0
    assert await otp_schedule.finished_reason("Senegal", base) is None
    assert await otp_schedule.is_paused("Senegal") is False


async def test_resuming_mid_run_keeps_the_run_clock(clock):
    await otp_schedule.set_country_settings("Senegal", {"run_minutes": 720})
    await otp_schedule.begin_run("Senegal")
    began = (await otp_schedule.get_run_state("Senegal"))["started_at"]
    await otp_schedule.set_paused("Senegal", True)
    clock.advance(60)

    await otp_schedule.set_paused("Senegal", False)

    assert (await otp_schedule.get_run_state("Senegal"))["started_at"] == began


# --------------------------------------------------------------------------- #
# 4. chat / API pause changes go through set_paused
# --------------------------------------------------------------------------- #
async def _finished_senegal(clock: Clock) -> None:
    await otp_schedule.set_country_settings("Senegal", {"run_minutes": 60})
    await otp_schedule.begin_run("Senegal")
    clock.advance(61)
    await otp_schedule.mark_finished("Senegal", "1h run time reached")


async def test_resuming_from_chat_starts_a_fresh_run(clock):
    await _finished_senegal(clock)

    await otp_bot.set_country_setting_from_chat("Senegal", "paused", "off")

    state = await otp_schedule.get_run_state("Senegal")
    assert not state.get("finished_at")
    assert state["started_at"] == clock.now.isoformat()
    assert await otp_schedule.is_paused("Senegal") is False
    assert await otp_schedule.finished_reason("Senegal", await otp_bot.get_config()) is None


async def test_resuming_from_the_api_settings_endpoint_starts_a_fresh_run(clock):
    from app.api import create_app

    await _finished_senegal(clock)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/otpbot/country/Senegal/settings",
            json={"paused": False},
            headers={"X-API-Token": "test-api-token"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["settings"]["paused"] is False

    state = await otp_schedule.get_run_state("Senegal")
    assert not state.get("finished_at")
    assert state["started_at"] == clock.now.isoformat()


# --------------------------------------------------------------------------- #
# 5. v2 migration also covers per-country delete_when_done
# --------------------------------------------------------------------------- #
async def _store_v1_config() -> None:
    from app.db import repo
    from app.db.base import session_scope

    async with session_scope() as session:
        await repo.set_setting(
            session, otp_bot.SETTING_KEY, {"enabled": True, "delete_when_done": True}
        )


async def test_the_v2_migration_turns_off_per_country_delete_when_done(environment):
    await _store_v1_config()
    await otp_schedule.apply_preset("One-shot burst", "Senegal")
    await otp_schedule.set_country_settings("Kenya", {"interval_minutes": 5})
    assert (await otp_schedule.get_country_settings("Senegal"))["delete_when_done"] is True

    cfg = await otp_bot.get_config()

    assert cfg["delete_when_done"] is False
    senegal = await otp_schedule.get_country_settings("Senegal")
    assert senegal["delete_when_done"] is False
    assert senegal["display_name"] == "Senegal"
    assert senegal["max_refills"] == 5, "the rest of the preset is untouched"
    assert (await otp_schedule.effective_config("Senegal", cfg))["delete_when_done"] is False
    assert "delete_when_done" not in await otp_schedule.get_country_settings("Kenya")


async def test_the_v2_migration_runs_only_once(environment):
    await _store_v1_config()
    await otp_bot.get_config()

    # Deliberately turned back on afterwards: it must stay on.
    await otp_schedule.set_country_settings("Senegal", {"delete_when_done": True})
    await otp_bot.get_config()

    assert (await otp_schedule.get_country_settings("Senegal"))["delete_when_done"] is True
