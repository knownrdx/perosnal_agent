"""The Telegram callback handler - what actually happens when a button is
tapped.

The panel tests cover keyboard CONSTRUCTION; these cover DELIVERY, which is
where the real failure was: allowed_updates omitted "callback_query", so
Telegram never delivered a button press and the symptom was a button that
did nothing and logged nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.integrations.telegram_user import set_userbot

pytestmark = pytest.mark.asyncio


class _Bot:
    """Enough of the userbot for start_automation to succeed."""

    def __init__(self) -> None:
        self._id = 500

    async def send_message(self, *a: Any, **k: Any) -> dict:
        self._id += 1
        return {"message_id": self._id}

    async def send_file(self, *a: Any, **k: Any) -> dict:
        self._id += 1
        return {"message_id": self._id}

    async def read_messages(self, *a: Any, **k: Any) -> list[dict]:
        self._id += 1
        return [{"id": self._id, "text": "Added.", "out": False}]


class _Message:
    """Captures what the handler sends back."""

    def __init__(self) -> None:
        self.answers: list[dict] = []
        self.edits: list[dict] = []

    async def answer(self, text: str, reply_markup: Any = None, **k: Any) -> None:
        self.answers.append({"text": text, "markup": reply_markup})

    async def edit_text(self, text: str, reply_markup: Any = None, **k: Any) -> None:
        self.edits.append({"text": text, "markup": reply_markup})


class _User:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _Query:
    def __init__(self, data: str, user_id: int) -> None:
        self.data = data
        self.from_user = _User(user_id)
        self.message = _Message()
        self.acks: list[dict] = []

    async def answer(self, text: str = "", show_alert: bool = False, **k: Any) -> None:
        self.acks.append({"text": text, "alert": show_alert})


def _handler(runner: Any):
    """Pull the registered callback handler out of the dispatcher."""
    for observer in runner.dp.observers.values():
        for handler in getattr(observer, "handlers", []):
            if handler.callback.__name__ == "_otp_callback":
                return handler.callback
    raise AssertionError("the OTP callback handler is not registered")


@pytest.fixture
def runner(environment):
    from app.telegram.bot import AgentBot

    return AgentBot()


async def _queue_one(name: str = "bd.txt", prefix: str = "+880") -> dict:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text("\n".join(f"{prefix}1711{i:06d}" for i in range(4)), encoding="utf-8")
    analysis = await otp_bot.enqueue_file(rel_path(path), name)
    return analysis["entries"][0]


# --------------------------------------------------------------------------- #
# Delivery: the bug that made every button silently do nothing
# --------------------------------------------------------------------------- #
async def test_polling_asks_telegram_for_button_presses(environment, monkeypatch):
    """Without callback_query in allowed_updates, Telegram drops the update
    server-side: no handler runs, nothing is logged, the button just dies.
    """
    from app.telegram.bot import AgentBot

    captured: dict[str, Any] = {}

    class _FakeDp:
        def start_polling(self, bot: Any, **kwargs: Any):
            captured.update(kwargs)

            async def _noop() -> None:
                return None

            return _noop()

    class _FakeBot:
        async def delete_webhook(self, **k: Any) -> None:
            return None

    runner = AgentBot()
    monkeypatch.setattr(runner, "dp", _FakeDp())
    monkeypatch.setattr(runner, "bot", _FakeBot())
    # telegram_enabled is a read-only computed property, so patch the guard
    # the runner actually consults rather than the Settings field.
    monkeypatch.setattr(
        type(runner.settings), "telegram_enabled", property(lambda self: True)
    )

    await runner.start()

    assert "callback_query" in captured["allowed_updates"]
    assert "message" in captured["allowed_updates"]


async def test_the_callback_handler_is_registered(runner):
    assert _handler(runner) is not None


# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #
async def test_a_stranger_cannot_drive_the_panel(runner):
    query = _Query("otp:stop:", user_id=999999)
    await _handler(runner)(query)

    assert query.acks[-1]["alert"] is True
    assert "authorized" in query.acks[-1]["text"].lower()
    # And nothing was changed.
    assert query.message.edits == []


# --------------------------------------------------------------------------- #
# The buttons themselves
# --------------------------------------------------------------------------- #
async def _owner_id(runner) -> int:
    return next(iter(runner.settings.allowed_user_ids))


async def test_interval_button_changes_the_interval(runner):
    query = _Query("otp:int:30", await _owner_id(runner))
    await _handler(runner)(query)

    assert (await otp_bot.get_config())["interval_minutes"] == 30
    assert query.message.edits, "the panel should be refreshed in place"


async def test_cleanup_button_switches_mode(runner):
    await _handler(runner)(_Query("otp:clean:force", await _owner_id(runner)))
    assert (await otp_bot.get_config())["force_delete_before_add"] is True

    await _handler(runner)(_Query("otp:clean:used", await _owner_id(runner)))
    assert (await otp_bot.get_config())["force_delete_before_add"] is False


async def test_service_button_tags_the_file_and_starts(runner):
    entry = await _queue_one()
    set_userbot(_Bot())

    query = _Query(f"otp:svc:{entry['id']}:WhatsApp", await _owner_id(runner))
    await _handler(runner)(query)

    active = await otp_bot.get_active_files()
    assert [e["tag"] for e in active] == ["WhatsApp"]


async def test_skip_button_drops_that_country(runner):
    entry = await _queue_one()

    query = _Query(f"otp:skip:{entry['id']}", await _owner_id(runner))
    await _handler(runner)(query)

    assert await otp_bot.get_queue() == []


async def test_per_country_interval_button_only_touches_that_country(runner):
    entry = await _queue_one()
    await otp_bot.set_queue_tag(entry["id"], "WhatsApp")
    await _queue_one("ng.txt", prefix="+234")

    # Index 0 is the first country in the queue ordering.
    query = _Query("otp:cint:0:5", await _owner_id(runner))
    await _handler(runner)(query)

    cfg = await otp_bot.get_config()
    assert (await otp_schedule.effective_config("Bangladesh", cfg))["interval_minutes"] == 5
    assert (await otp_schedule.effective_config("Nigeria", cfg))["interval_minutes"] == cfg[
        "interval_minutes"
    ]


async def test_preset_button_applies_to_the_chosen_country(runner):
    await _queue_one()

    query = _Query("otp:usepre:0:0", await _owner_id(runner))  # first country, first preset
    await _handler(runner)(query)

    settings = await otp_schedule.get_country_settings("Bangladesh")
    assert settings, "the preset should have written per-country settings"


async def test_a_button_for_a_country_that_is_gone_says_so(runner):
    """Buttons live in chat history; the owner can tap one long after the
    country was removed, and that must not raise.
    """
    query = _Query("otp:cint:7:5", await _owner_id(runner))
    await _handler(runner)(query)

    assert query.acks[-1]["alert"] is True
    assert "gone" in query.acks[-1]["text"].lower()


async def test_clear_queue_button_empties_the_queue(runner):
    await _queue_one()
    assert await otp_bot.get_queue()

    await _handler(runner)(_Query("otp:clearq:", await _owner_id(runner)))
    assert await otp_bot.get_queue() == []


async def test_refresh_countries_button_learns_from_the_bot(runner):
    class _StockBot(_Bot):
        async def read_messages(self, *a: Any, **k: Any) -> list[dict]:
            self._id += 1
            return [{
                "id": self._id,
                "text": "\U0001F30D Country Stock (yours):\n  Laos: 0\n  Nigeria: 5\n",
                "out": False,
            }]

    set_userbot(_StockBot())
    await _handler(runner)(_Query("otp:refreshc:", await _owner_id(runner)))

    known = (await otp_bot.get_known_countries())["names"]
    assert "laos" in known
    assert "nigeria" in known


async def test_malformed_callback_data_is_ignored_quietly(runner):
    """Old or corrupted payloads must not raise inside the handler."""
    query = _Query("otp", await _owner_id(runner))
    await _handler(runner)(query)
    assert query.acks  # acknowledged, not crashed


async def test_an_unknown_action_is_acknowledged(runner):
    query = _Query("otp:no_such_action:x", await _owner_id(runner))
    await _handler(runner)(query)
    assert query.acks


class _Command:
    """Stands in for aiogram's CommandObject."""

    def __init__(self, args: str = "") -> None:
        self.args = args


class _PlainMessage(_Message):
    def __init__(self, user_id: int) -> None:
        super().__init__()
        self.from_user = _User(user_id)
        self.chat = type("_Chat", (), {"id": user_id})()


def _command_handler(runner: Any, name: str):
    for observer in runner.dp.observers.values():
        for handler in getattr(observer, "handlers", []):
            if handler.callback.__name__ == name:
                return handler.callback
    raise AssertionError(f"{name} is not registered")


# --------------------------------------------------------------------------- #
# /otpset and /otppreset - settings from Telegram
# --------------------------------------------------------------------------- #
async def test_otpset_with_no_args_lists_everything(runner):
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otpset")(message, _Command(""))

    text = message.answers[-1]["text"]
    assert "interval_minutes" in text
    assert "/otpset" in text


async def test_otpset_changes_a_global_setting(runner):
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otpset")(message, _Command("interval_minutes 20"))

    assert (await otp_bot.get_config())["interval_minutes"] == 20


async def test_otpset_handles_a_multi_word_country(runner):
    """"Central African Republic interval_minutes 5" - the country name has
    spaces, so the key has to be located rather than assumed to be word two.
    """
    await otp_bot.learn_countries({"Central African Republic": 1})
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otpset")(
        message, _Command("Central African Republic interval_minutes 5")
    )

    settings = await otp_schedule.get_country_settings("Central African Republic")
    assert settings["interval_minutes"] == 5


async def test_otpset_reports_a_bad_value_instead_of_storing_it(runner):
    before = (await otp_bot.get_config())["interval_minutes"]
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otpset")(message, _Command("interval_minutes banana"))

    assert "\u274c" in message.answers[-1]["text"].lower() or "number" in message.answers[-1]["text"]
    assert (await otp_bot.get_config())["interval_minutes"] == before


async def test_otppreset_saves_a_custom_threshold(runner):
    """The thing actually asked for: a preset that refills at a chosen
    stock level rather than waiting for zero.
    """
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otppreset")(
        message, _Command("save My limit quota_threshold=500 interval_minutes=20 limit=6")
    )

    presets = await otp_schedule.get_presets()
    assert presets["My limit"]["quota_threshold"] == 500
    assert presets["My limit"]["interval_minutes"] == 20
    assert presets["My limit"]["limit"] == 6


async def test_a_preset_threshold_of_zero_survives(runner):
    """0 means "wait until empty" - a truthiness filter would drop it and
    silently fall back to the global value.
    """
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otppreset")(
        message, _Command("save Drain quota_threshold=0")
    )

    presets = await otp_schedule.get_presets()
    assert presets["Drain"]["quota_threshold"] == 0


async def test_otppreset_applies_to_a_multi_word_country(runner):
    await otp_bot.learn_countries({"Central African Republic": 1})
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otppreset")(
        message, _Command("use Fast burn for Central African Republic")
    )

    settings = await otp_schedule.get_country_settings("Central African Republic")
    assert settings["interval_minutes"] == 5


async def test_otppreset_rejects_a_field_that_is_not_per_country(runner):
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otppreset")(
        message, _Command("save Bad target_bot=@nope")
    )

    assert "\u274c" in message.answers[-1]["text"]
    assert "Bad" not in await otp_schedule.get_presets()


async def test_otppreset_lists_presets_with_their_thresholds(runner):
    message = _PlainMessage(await _owner_id(runner))
    await _command_handler(runner, "_otppreset")(message, _Command(""))

    text = message.answers[-1]["text"]
    assert "Low-stock refill" in text
    assert "refill at 200" in text

