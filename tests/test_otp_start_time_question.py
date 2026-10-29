"""The "when should this be added?" question.

Every new upload is asked when it should go in - now, or a time (Dubai) -
unless the caption already said. A replacement file for a running country is
asked too: a start time is an instruction about one file, not a standing
property of the country. The rate per OTP is the target bot's own business
(config v5 dropped the agent's per-country rates), so nothing about it is
asked or sent.

Driven through the real entry points with a fake @PBDxbot that records every
command, so what is asserted is what reaches the bot.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.db import repo
from app.db.base import session_scope
from app.integrations.telegram_user import set_userbot

pytestmark = pytest.mark.asyncio

DUBAI = otp_schedule.DUBAI_TZ
STOCK_HEADER = "\U0001F30D Country Stock (yours):"
FAST_ADD = "⚡ Fast Add Complete!\n\nBangladesh: 3 added (0 dup)\n\U0001F4B5 Rate: 0.0005"


def dubai(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=DUBAI).astimezone(timezone.utc)


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def set(self, when: datetime) -> None:
        self.now = when


@pytest.fixture
def clock(monkeypatch) -> Clock:
    """10:00 Dubai on 5 October - a daytime upload."""
    fake = Clock(dubai(5, 10, 0))
    monkeypatch.setattr(otp_schedule, "_now", fake)
    monkeypatch.setattr(otp_bot, "_now_iso", lambda: fake.now.isoformat(), raising=False)
    return fake


class PBDxBot:
    """Answers like the real bot and records every send, in order."""

    def __init__(self, stock: dict[str, int] | None = None) -> None:
        self.stock = dict(stock or {})
        self.log: list[tuple[str, str, int | None]] = []
        self._next_message_id = 1000
        self._reply_id = 0
        self._current: dict[str, Any] | None = None

    def _stock_text(self) -> str:
        lines = ["\U0001F4CA Bot Statistics", "", STOCK_HEADER]
        for country, count in self.stock.items():
            lines.append(f"  \U0001F3F3 {country}: {count} (+5 taken)")
        return "\n".join(lines)

    def _answer(self, text: str, reply_to: int | None) -> str:
        head = text.strip().split()[0] if text.strip() else ""
        if reply_to is not None or head == "/fan":
            return FAST_ADD
        if head == "/st":
            return self._stock_text()
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

    def add_commands(self) -> list[str]:
        return [t for kind, t, reply_to in self.log if kind == "msg" and reply_to is not None]

    def files(self) -> list[str]:
        return [p.replace("\\", "/").rsplit("/", 1)[-1] for kind, p, _ in self.log if kind == "file"]


@pytest.fixture(autouse=True)
def _no_sleep_and_reset_userbot(monkeypatch):
    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)
    yield
    set_userbot(None)


async def _upload(name: str = "numbers.txt", caption: str = "", prefix: str = "+880") -> dict[str, Any]:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(f"{prefix}1711111111\n{prefix}1722222222\n{prefix}1733333333\n", encoding="utf-8")
    return await otp_bot.enqueue_file(rel_path(path), name, caption)


async def _only_the_start_question() -> None:
    """Tag known, run length not asked: the start question is the only one
    left."""
    await otp_bot.save_config({"default_tag": "WhatsApp", "ask_run_time": False})


# --------------------------------------------------------------------------- #
# The start-time question
# --------------------------------------------------------------------------- #
async def test_a_new_upload_is_asked_when_it_should_be_added(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _only_the_start_question()
    await _upload()

    assert await otp_bot.countries_needing_start(await otp_bot.get_queue()) == ["Bangladesh"]
    assert await otp_bot.maybe_auto_start() is None
    asked = await otp_bot.handle_start_trigger()
    assert "When should Bangladesh be added" in asked
    assert "04:00" in asked                        # the default, stated
    assert (await otp_bot.get_awaiting_start())["countries"] == ["Bangladesh"]


async def test_answering_now_adds_it_straight_away(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _only_the_start_question()
    await _upload()
    await otp_bot.handle_start_trigger()

    answered = await otp_bot.handle_start_answer("now")

    assert "Started" in answered
    assert bot.files() == ["numbers.txt"]
    assert await otp_schedule.has_started("Bangladesh", await otp_bot.get_config())
    assert await otp_bot.get_awaiting_start() in (None, {})


async def test_answering_a_time_holds_the_file_until_then(clock):
    bot = PBDxBot({"Bangladesh": 0})
    set_userbot(bot)
    await _only_the_start_question()
    await _upload()
    await otp_bot.handle_start_trigger()

    answered = await otp_bot.handle_start_answer("21:00")

    assert "Scheduled" in answered and "21:00" in answered
    assert bot.files() == []
    cfg = await otp_bot.get_config()
    assert not await otp_schedule.has_started("Bangladesh", cfg)

    clock.set(dubai(5, 20, 59))
    await otp_bot.run_cycle()
    assert bot.files() == []

    clock.set(dubai(5, 21, 0))
    await otp_bot.run_cycle()
    assert bot.files() == ["numbers.txt"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]


async def test_answering_default_follows_the_global_start_time(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _only_the_start_question()
    await _upload()
    await otp_bot.handle_start_trigger()

    answered = await otp_bot.handle_start_answer("default")

    assert "04:00" in answered
    assert "start_at" not in await otp_schedule.get_country_settings("Bangladesh")
    assert bot.files() == []                        # waits for 04:00
    begins = await otp_schedule.pending_start_at("Bangladesh", await otp_bot.get_config())
    assert begins == dubai(6, 4, 0)


async def test_a_caption_start_time_is_not_asked_again(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _only_the_start_question()
    await _upload(caption="ekhoni shuru")

    assert await otp_bot.countries_needing_start(await otp_bot.get_queue()) == []
    started = await otp_bot.maybe_auto_start()
    assert started is not None and started["ok"]
    assert bot.files() == ["numbers.txt"]


async def test_a_replacement_file_for_a_running_country_waits_for_its_time(clock):
    """The new file replaces the old one at once; the country then waits for
    the chosen time before the new file is first added, and nothing is
    refilled from the old file in between (it is gone)."""
    bot = PBDxBot({"Bangladesh": 0})
    set_userbot(bot)
    await _only_the_start_question()
    await _upload(caption="ekhoni shuru")
    assert (await otp_bot.maybe_auto_start())["ok"]
    assert bot.files() == ["numbers.txt"]

    await _upload("numbers2.txt")
    assert await otp_bot.countries_needing_start(await otp_bot.get_queue()) == ["Bangladesh"]
    asked = await otp_bot.handle_start_trigger()
    assert "When should Bangladesh be added" in asked
    await otp_bot.handle_start_answer("9pm")

    active = [e for e in await otp_bot.get_active_files() if e["country"] == "Bangladesh"]
    assert len(active) == 1 and active[0]["name"] == "numbers2.txt"
    assert active[0]["waiting_start"] == "21:00"

    clock.set(dubai(5, 15, 0))
    await otp_bot.run_cycle()
    assert bot.files() == ["numbers.txt"]                # nothing re-added from the old file

    clock.set(dubai(5, 21, 0))
    await otp_bot.run_cycle()
    assert bot.files() == ["numbers.txt", "numbers2.txt"]


async def test_a_replacement_file_answered_now_does_not_regate_a_running_country(clock):
    bot = PBDxBot({"Bangladesh": 0})
    set_userbot(bot)
    await _only_the_start_question()
    await _upload(caption="ekhoni shuru")
    assert (await otp_bot.maybe_auto_start())["ok"]

    await _upload("numbers2.txt")
    await otp_bot.handle_start_trigger()
    await otp_bot.handle_start_answer("ekhoni")

    active = [e for e in await otp_bot.get_active_files() if e["country"] == "Bangladesh"]
    assert len(active) == 1 and active[0]["name"] == "numbers2.txt"
    assert not active[0].get("waiting_start")
    assert await otp_schedule.has_started("Bangladesh", await otp_bot.get_config())
    assert bot.files() == ["numbers.txt", "numbers2.txt"]


async def test_a_bad_start_answer_keeps_the_question_open(clock):
    set_userbot(PBDxBot())
    await _only_the_start_question()
    await _upload()
    await otp_bot.handle_start_trigger()

    reply = await otp_bot.handle_start_answer("whenever")

    assert reply.startswith("❌")
    assert (await otp_bot.get_awaiting_start())["countries"] == ["Bangladesh"]


async def test_turning_the_start_question_off_applies_the_default_silently(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _only_the_start_question()
    await otp_bot.save_config({"ask_start_time": False})
    await _upload()

    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert started["files"][0]["waiting_start"] == "04:00"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("now", otp_bot.START_NOW), ("ekhoni", otp_bot.START_NOW), ("ekhuni", otp_bot.START_NOW),
        ("21:00", "21:00"), ("9pm", "21:00"), ("rat 9ta", "21:00"), ("2100", "21:00"),
        ("21.00", "21:00"), ("default", None),
    ],
)
def test_parse_start_time_reads_what_the_owner_types(text, expected):
    assert otp_bot.parse_start_time(text) == expected


@pytest.mark.parametrize("text", ["", "whenever", "25:00"])
def test_parse_start_time_rejects_nonsense(text):
    with pytest.raises(ValueError):
        otp_bot.parse_start_time(text)


# --------------------------------------------------------------------------- #
# The whole chain: tag -> start time -> run time -> start
# --------------------------------------------------------------------------- #
async def test_questions_come_in_order_tag_start_runtime(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _upload("mystery.txt")                    # no tag to be inferred

    assert await otp_bot.maybe_auto_start() is None
    asked = await otp_bot.handle_start_trigger()
    assert "Which service" in asked

    asked = await otp_bot.handle_tag_answer("WhatsApp")
    assert "When should Bangladesh be added" in asked
    assert await otp_bot.get_awaiting_start()

    asked = await otp_bot.handle_start_answer("now")
    assert "How long should Bangladesh run" in asked
    assert await otp_bot.get_awaiting_start() in (None, {})
    assert await otp_bot.get_awaiting_runtime()

    done = await otp_bot.handle_runtime_answer("8h")
    assert "Started" in done
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]
    own = await otp_schedule.get_country_settings("Bangladesh")
    assert own["run_minutes"] == 480
    assert (await otp_bot.get_config())["enabled"] is True


async def test_no_rate_is_ever_asked_or_sent(clock):
    """The target bot handles rates itself: the agent neither asks for one
    nor puts "-r" on the add command, and a "rate" in the caption is not a
    caption field any more."""
    bot = PBDxBot()
    set_userbot(bot)
    await _upload(caption="whatsapp ekhoni shuru 20h")

    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]
    assert not hasattr(otp_bot, "get_awaiting_rate")
    assert not hasattr(otp_bot, "handle_rate_answer")
    assert "rate" not in otp_schedule.OVERRIDABLE
    assert "default_rate" not in otp_bot.DEFAULT_CONFIG
    assert "ask_rate" not in otp_bot.DEFAULT_CONFIG


async def test_a_full_caption_is_asked_nothing_at_all(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _upload(caption="whatsapp ekhoni shuru 20h")

    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]


async def test_clearing_the_queue_closes_the_open_questions():
    set_userbot(PBDxBot())
    await _upload()
    await otp_bot.set_awaiting_start(["Bangladesh"])
    await otp_bot.set_awaiting_runtime(["Bangladesh"])

    await otp_bot.clear_queue()

    assert await otp_bot.get_awaiting_start() in (None, {})
    assert await otp_bot.get_awaiting_runtime() in (None, {})


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #
V4_TEMPLATE = "/fan -t {tag} -l {limit} -c {count}{rate_flag}"


async def test_the_default_add_command_has_no_rate_flag():
    assert otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE == "/fan -t {tag} -l {limit} -c {count}"
    assert "rate" not in otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert otp_bot.CONFIG_VERSION == 5


async def test_the_v5_migration_takes_the_rate_flag_off_the_v4_template():
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 4,
            "add_command_template": V4_TEMPLATE,
            "default_rate": "0.0005", "ask_rate": True,
        })

    cfg = await otp_bot.get_config()

    assert cfg["add_command_template"] == otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert cfg["config_version"] == 5
    assert "default_rate" not in cfg and "ask_rate" not in cfg
    async with session_scope() as session:
        stored = await repo.get_setting(session, otp_bot.SETTING_KEY)
    assert stored["add_command_template"] == otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert stored["config_version"] == 5
    assert "default_rate" not in stored and "ask_rate" not in stored


async def test_the_v5_migration_drops_remembered_per_country_rates():
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 4, "add_command_template": V4_TEMPLATE,
        })
        await repo.set_setting(session, otp_schedule.COUNTRY_KEY, {
            "bangladesh": {"display_name": "Bangladesh", "rate": "0.0005", "run_minutes": 480},
            "nigeria": {"display_name": "Nigeria", "rate": ""},
        })

    await otp_bot.get_config()

    own = await otp_schedule.get_all_country_settings()
    assert "rate" not in own["bangladesh"] and own["bangladesh"]["run_minutes"] == 480
    assert "rate" not in own["nigeria"]


async def test_the_v5_migration_leaves_a_customised_template_alone():
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 4,
            "add_command_template": "/fan -t {tag} -c {country}",
        })

    cfg = await otp_bot.get_config()

    assert cfg["add_command_template"] == "/fan -t {tag} -c {country}"
    assert cfg["config_version"] == 5


async def test_a_v3_config_migrates_straight_to_v5():
    """Every earlier block still runs, and the plain template stays plain."""
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 3,
            "add_command_template": "/fan -t {tag} -l {limit} -c {count}",
        })

    cfg = await otp_bot.get_config()

    assert cfg["add_command_template"] == otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert cfg["config_version"] == 5


async def test_a_customised_template_still_sends_exactly_what_it_says():
    bot = PBDxBot()
    set_userbot(bot)
    await otp_bot.save_config({
        "default_tag": "WhatsApp", "ask_start_time": False, "ask_run_time": False,
        "start_at": otp_bot.START_NOW,
        "add_command_template": "/fan -t {tag} -c {country}",
    })
    await _upload()

    assert (await otp_bot.start_automation())["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -c Bangladesh"]
