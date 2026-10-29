"""The rate per OTP and the "when should this be added?" question.

The target bot pays a revenue per OTP ("-r" on /fan) and does NOT remember it
per country: leave the flag off and it silently uses its global default. Some
countries need a different rate, so the agent remembers one per country, asks
the first time a country is seen, and puts "-r <rate>" on every add command
from then on. Separately, every new upload is asked when it should go in -
now, or a time - unless the caption already said.

Driven through the real entry points with a fake @PBDxbot that records every
command, so what is asserted is what reaches the bot.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.automation import otp_bot, otp_caption, otp_schedule
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


async def _no_questions_but_rate() -> None:
    """Only the rate question stays on, so a test about rates is not also a
    test about start times and run lengths."""
    await otp_bot.save_config({
        "default_tag": "WhatsApp", "ask_start_time": False, "ask_run_time": False,
        "start_at": otp_bot.START_NOW,
    })


# --------------------------------------------------------------------------- #
# The add command carries the rate
# --------------------------------------------------------------------------- #
async def test_the_default_add_command_carries_the_rate_flag():
    assert otp_bot.DEFAULT_CONFIG["add_command_template"] == "/fan -t {tag} -l {limit} -c {count}{rate_flag}"
    assert "rate" in otp_schedule.OVERRIDABLE


async def test_a_zero_rate_is_sent_as_minus_r_zero():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_schedule.set_country_settings("Bangladesh", {"rate": "0"})
    await _upload()

    result = await otp_bot.start_automation()

    assert result["ok"], result
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0"]
    assert result["files"][0]["rate"] == "0"


async def test_a_bot_default_rate_sends_no_minus_r_at_all():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_schedule.set_country_settings("Bangladesh", {"rate": ""})
    await _upload()

    result = await otp_bot.start_automation()

    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]
    assert result["files"][0]["rate"] == ""
    assert "Rate: bot default" in otp_bot.format_start_result(result)


async def test_the_global_default_rate_applies_to_a_country_without_its_own():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_bot.save_config({"ask_rate": False, "default_rate": "0.001"})
    await _upload()

    result = await otp_bot.start_automation()

    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.001"]
    assert "Rate: 0.001" in otp_bot.format_start_result(result)


async def test_refills_carry_the_same_rate(clock):
    bot = PBDxBot({"Bangladesh": 0})
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_bot.save_config({"start_at": otp_bot.START_NOW})
    await _upload(caption="rate 0.0005")
    assert (await otp_bot.start_automation())["ok"]

    clock.set(clock.now + timedelta(minutes=5))
    await otp_bot.run_cycle()

    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"] * 2


# --------------------------------------------------------------------------- #
# Asking, remembering, and the caption
# --------------------------------------------------------------------------- #
async def test_a_new_country_is_asked_its_rate_before_anything_is_sent():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await _upload()

    assert await otp_bot.countries_needing_rate(await otp_bot.get_queue()) == ["Bangladesh"]
    assert await otp_bot.maybe_auto_start() is None
    assert bot.add_commands() == []

    asked = await otp_bot.handle_start_trigger()
    assert "What rate for Bangladesh" in asked
    assert "0.0005" in asked and "default" in asked
    assert (await otp_bot.get_awaiting_rate())["countries"] == ["Bangladesh"]

    answered = await otp_bot.handle_rate_answer("0.0005")

    assert "Started" in answered
    assert await otp_bot.get_awaiting_rate() in (None, {})
    assert (await otp_schedule.get_country_settings("Bangladesh"))["rate"] == "0.0005"
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"]


async def test_a_remembered_rate_is_reused_without_asking():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_bot.handle_set_rate("Bangladesh", "0.0005")
    await _upload()

    assert await otp_bot.countries_needing_rate(await otp_bot.get_queue()) == []
    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"]


async def test_a_caption_rate_answers_the_question_in_advance():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    analysis = await _upload(caption="rate 0.0005")

    assert analysis["caption"]["rate"] == "0.0005"
    assert "rate 0.0005" in analysis["caption_note"]
    assert await otp_bot.countries_needing_rate(await otp_bot.get_queue()) == []
    started = await otp_bot.maybe_auto_start()
    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"]


async def test_answering_default_remembers_the_bots_own_rate():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await _upload()
    await otp_bot.handle_start_trigger()

    answered = await otp_bot.handle_rate_answer("default")

    assert "Started" in answered
    own = await otp_schedule.get_country_settings("Bangladesh")
    assert own["rate"] == ""                       # remembered as "bot default"
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]
    # ... and not asked again next time.
    await _upload("numbers2.txt")
    assert await otp_bot.countries_needing_rate(await otp_bot.get_queue()) == []


async def test_a_bad_rate_answer_keeps_the_question_open():
    set_userbot(PBDxBot())
    await _no_questions_but_rate()
    await _upload()
    await otp_bot.handle_start_trigger()

    reply = await otp_bot.handle_rate_answer("cheap")

    assert reply.startswith("❌")
    assert (await otp_bot.get_awaiting_rate())["countries"] == ["Bangladesh"]
    assert "rate" not in await otp_schedule.get_country_settings("Bangladesh")


async def test_turning_the_rate_question_off_uses_the_default_silently():
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_bot.save_config({"ask_rate": False})
    await _upload()

    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4"]


async def test_auto_start_waits_while_a_rate_question_is_open():
    set_userbot(PBDxBot())
    await _no_questions_but_rate()
    await otp_bot.save_config({"ask_rate": False})
    await _upload()
    await otp_bot.set_awaiting_rate(["Bangladesh"])

    assert await otp_bot.maybe_auto_start() is None


# --------------------------------------------------------------------------- #
# Rate tools for the owner
# --------------------------------------------------------------------------- #
async def test_rates_can_be_listed_and_set_from_a_command():
    await otp_bot.handle_set_rate("Bangladesh", "0.0005")
    await otp_bot.handle_set_rate("Nigeria", "default")
    await otp_bot.save_config({"default_rate": "0.001"})

    listing = await otp_bot.handle_rate_list()

    assert "Bangladesh: 0.0005" in listing
    assert "Nigeria: bot default" in listing
    assert "Default for new countries: 0.001" in listing
    with pytest.raises(ValueError):
        await otp_bot.handle_set_rate("Bangladesh", "-1")


async def test_the_rate_is_a_per_country_setting_from_chat():
    reply = await otp_bot.set_country_setting_from_chat("Bangladesh", "rate", "0.0005")
    assert "rate = 0.0005" in reply
    assert (await otp_schedule.get_country_settings("Bangladesh"))["rate"] == "0.0005"

    reply = await otp_bot.set_setting_from_chat("default_rate", "default")
    assert "bot default" in reply
    assert (await otp_bot.get_config())["default_rate"] == ""


async def test_the_schedule_overview_shows_each_countrys_rate():
    await otp_schedule.set_country_settings("Bangladesh", {"rate": "0.0005"})
    cfg = await otp_bot.get_config()

    rows = {r["country"]: r for r in await otp_schedule.schedule_overview(["Bangladesh", "Nigeria"], cfg)}

    assert rows["Bangladesh"]["rate"] == "0.0005" and rows["Bangladesh"]["rate_remembered"] is True
    assert rows["Nigeria"]["rate"] == "" and rows["Nigeria"]["rate_remembered"] is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", "0"), ("0.0005", "0.0005"), (".0005", "0.0005"), ("0.00050", "0.0005"),
        ("default", ""), ("bot default", ""), ("skip", ""), ("-r 0.001", "0.001"),
        ("rate: 0", "0"), ("0,0005", "0.0005"),
    ],
)
def test_parse_rate_reads_what_the_owner_types(text, expected):
    assert otp_bot.parse_rate(text) == expected


@pytest.mark.parametrize("text", ["", "cheap", "-1", "0.0.5", "1e400"])
def test_parse_rate_rejects_nonsense_with_a_message(text):
    with pytest.raises(ValueError) as excinfo:
        otp_bot.parse_rate(text)
    assert str(excinfo.value)


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("bangladesh whatsapp rate 0.0005", "0.0005"),
        ("rate: 0", "0"),
        ("-r 0.0005 20h", "0.0005"),
        ("whatsapp r 0", "0"),
        ("rate default", ""),
    ],
)
def test_the_caption_states_the_rate(caption, expected):
    parsed = otp_caption.parse_caption(caption)
    assert parsed.get("rate") == expected


def test_a_rate_in_the_caption_is_not_misread_as_a_time():
    parsed = otp_caption.parse_caption("whatsapp rate 0.05 20h")
    assert parsed["rate"] == "0.05"
    assert "stop_at" not in parsed
    assert parsed["run_minutes"] == 1200
    assert "rate" not in otp_caption.parse_caption("rat 9 ta porjonto")
    assert "rate" not in otp_caption.parse_caption("for 20h")


# --------------------------------------------------------------------------- #
# The start-time question
# --------------------------------------------------------------------------- #
async def _rate_known() -> None:
    """Rate answered already, run length not asked: the start question is
    the only one left."""
    await otp_bot.save_config({"default_tag": "WhatsApp", "ask_run_time": False})
    await otp_schedule.set_country_settings("Bangladesh", {"rate": "0.0005"})


async def test_a_new_upload_is_asked_when_it_should_be_added(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _rate_known()
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
    await _rate_known()
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
    await _rate_known()
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
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"]


async def test_answering_default_follows_the_global_start_time(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _rate_known()
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
    await _rate_known()
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
    await _rate_known()
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
    await _rate_known()
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
    await _rate_known()
    await _upload()
    await otp_bot.handle_start_trigger()

    reply = await otp_bot.handle_start_answer("whenever")

    assert reply.startswith("❌")
    assert (await otp_bot.get_awaiting_start())["countries"] == ["Bangladesh"]


async def test_turning_the_start_question_off_applies_the_default_silently(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _rate_known()
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
# The whole chain: tag -> rate -> start time -> run time -> start
# --------------------------------------------------------------------------- #
async def test_questions_come_in_order_tag_rate_start_runtime(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _upload("mystery.txt")                    # no tag to be inferred

    assert await otp_bot.maybe_auto_start() is None
    asked = await otp_bot.handle_start_trigger()
    assert "Which service" in asked

    asked = await otp_bot.handle_tag_answer("WhatsApp")
    assert "What rate for Bangladesh" in asked
    assert await otp_bot.get_awaiting_rate()

    asked = await otp_bot.handle_rate_answer("0.0005")
    assert "When should Bangladesh be added" in asked
    assert await otp_bot.get_awaiting_rate() in (None, {})
    assert await otp_bot.get_awaiting_start()

    asked = await otp_bot.handle_start_answer("now")
    assert "How long should Bangladesh run" in asked
    assert await otp_bot.get_awaiting_start() in (None, {})
    assert await otp_bot.get_awaiting_runtime()

    done = await otp_bot.handle_runtime_answer("8h")
    assert "Started" in done
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0.0005"]
    own = await otp_schedule.get_country_settings("Bangladesh")
    assert own["rate"] == "0.0005" and own["run_minutes"] == 480
    assert (await otp_bot.get_config())["enabled"] is True


async def test_a_full_caption_is_asked_nothing_at_all(clock):
    bot = PBDxBot()
    set_userbot(bot)
    await _upload(caption="whatsapp rate 0 ekhoni shuru 20h")

    started = await otp_bot.maybe_auto_start()

    assert started is not None and started["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -l 4 -c 4 -r 0"]


async def test_clearing_the_queue_closes_the_open_questions():
    set_userbot(PBDxBot())
    await _upload()
    await otp_bot.set_awaiting_rate(["Bangladesh"])
    await otp_bot.set_awaiting_start(["Bangladesh"])

    await otp_bot.clear_queue()

    assert await otp_bot.get_awaiting_rate() in (None, {})
    assert await otp_bot.get_awaiting_start() in (None, {})


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #
async def test_the_v4_migration_puts_the_rate_flag_on_the_untouched_template():
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 3,
            "add_command_template": "/fan -t {tag} -l {limit} -c {count}",
        })

    cfg = await otp_bot.get_config()

    assert cfg["add_command_template"] == otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert cfg["config_version"] == 4
    async with session_scope() as session:
        stored = await repo.get_setting(session, otp_bot.SETTING_KEY)
    assert stored["add_command_template"] == otp_bot.DEFAULT_ADD_COMMAND_TEMPLATE
    assert stored["config_version"] == 4


async def test_the_v4_migration_leaves_a_customised_template_alone():
    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY, {
            "enabled": False, "config_version": 3,
            "add_command_template": "/fan -t {tag} -c {country}",
        })

    cfg = await otp_bot.get_config()

    assert cfg["add_command_template"] == "/fan -t {tag} -c {country}"
    assert cfg["config_version"] == 4


async def test_an_old_template_without_the_placeholder_still_works():
    """A customised template that never heard of {rate_flag} keeps sending
    exactly what it always did."""
    bot = PBDxBot()
    set_userbot(bot)
    await _no_questions_but_rate()
    await otp_bot.save_config({"add_command_template": "/fan -t {tag} -c {country}"})
    await otp_schedule.set_country_settings("Bangladesh", {"rate": "0.0005"})
    await _upload()

    assert (await otp_bot.start_automation())["ok"]
    assert bot.add_commands() == ["/fan -t WhatsApp -c Bangladesh"]
