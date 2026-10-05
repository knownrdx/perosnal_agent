"""How the owner drives the OTP automation from Telegram - the failure modes
that only show up against the real Bot API.

Each test here pins one bug that was found by reading the handlers against
Telegram's actual limits and against the order things really happen in:

* buttons that carried a country by its POSITION in a list that reorders
  (every auto-start moves the restarted country to the end), so a stale
  button acted on a different country - including "Bad dao";
* replies over Telegram's 4096-character cap, which the API rejects
  outright, so a long start report or status simply never arrived;
* exception paths that left a button spinning or an upload unanswered;
* control words ("stop", "status") swallowed as the answer to an open
  question - "stop" during the service question tagged the file "stop" and
  then STARTED it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.automation import otp_bot, otp_schedule
from app.integrations.telegram_user import set_userbot
from app.telegram import otp_panel

pytestmark = pytest.mark.asyncio

TELEGRAM_MAX = 4096
LONG_NAMES = (
    "Saint Vincent And The Grenadines",
    "Bonaire, Sint Eustatius and Saba",
    "Sint Maarten (Dutch part)",
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Userbot:
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


class _TooLong(Exception):
    pass


class _Message:
    """Captures replies, and refuses an over-long one the way Telegram does."""

    def __init__(self) -> None:
        self.answers: list[dict] = []
        self.edits: list[dict] = []

    async def answer(self, text: str, reply_markup: Any = None, **k: Any) -> None:
        if len(text) > TELEGRAM_MAX or not text.strip():
            raise _TooLong("Telegram server says - Bad Request: message is too long")
        self.answers.append({"text": text, "markup": reply_markup})

    async def edit_text(self, text: str, reply_markup: Any = None, **k: Any) -> None:
        if len(text) > TELEGRAM_MAX or not text.strip():
            raise _TooLong("Telegram server says - Bad Request: message is too long")
        self.edits.append({"text": text, "markup": reply_markup})

    def everything(self) -> str:
        return "\n".join(a["text"] for a in self.answers + self.edits)


class _User:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _Chat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _Query:
    def __init__(self, data: str, user_id: int) -> None:
        self.data = data
        self.from_user = _User(user_id)
        self.message = _Message()
        self.acks: list[dict] = []

    async def answer(self, text: str = "", show_alert: bool = False, **k: Any) -> None:
        self.acks.append({"text": text, "alert": show_alert})


class _PlainMessage(_Message):
    def __init__(self, user_id: int) -> None:
        super().__init__()
        self.from_user = _User(user_id)
        self.chat = _Chat(user_id)


class _Command:
    def __init__(self, args: str = "") -> None:
        self.args = args


class _Document:
    def __init__(self, name: str, size: int) -> None:
        self.file_name = name
        self.file_size = size
        self.file_unique_id = "uniq1"
        self.file_id = "file1"


class _TelegramFiles:
    """message.bot as the document handler uses it: fetch + download."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def get_file(self, file_id: str) -> Any:
        return type("_File", (), {"file_path": "documents/file_1"})()

    async def download_file(self, file_path: str, destination: str) -> None:
        Path(destination).write_bytes(self.data)


class _DocumentMessage(_PlainMessage):
    def __init__(self, user_id: int, name: str, data: bytes, caption: str = "") -> None:
        super().__init__(user_id)
        self.document = _Document(name, len(data))
        self.caption = caption
        self.bot = _TelegramFiles(data)


async def _no_sleep(seconds: float) -> None:
    return None


@pytest.fixture
def runner(environment, monkeypatch):
    from app.telegram.bot import AgentBot

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)
    yield AgentBot()
    set_userbot(None)


def _callback(runner: Any, name: str):
    for observer in runner.dp.observers.values():
        for handler in getattr(observer, "handlers", []):
            if handler.callback.__name__ == name:
                return handler.callback
    raise AssertionError(f"{name} is not registered")


def _owner(runner: Any) -> int:
    return next(iter(runner.settings.allowed_user_ids))


async def _tap(runner: Any, data: str) -> _Query:
    query = _Query(data, _owner(runner))
    await _callback(runner, "_otp_callback")(query)
    return query


def _button(markup: Any, needle: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if needle in button.text:
                return button.callback_data
    raise AssertionError(f"no button containing {needle!r}")


async def _queue(name: str, prefix: str, caption: str = "whatsapp 20h") -> dict:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text("\n".join(f"{prefix}1711{i:06d}" for i in range(4)), encoding="utf-8")
    analysis = await otp_bot.enqueue_file(rel_path(path), name, caption)
    return analysis["entries"][0]


async def _many_active(count: int = 60) -> list[str]:
    """A big run: enough countries that the status no longer fits one message."""
    countries = [f"{LONG_NAMES[i % 3]} {i}" for i in range(count)]
    entries = [
        {
            "id": f"{i:08x}",
            "batch_id": "b0",
            "path": "uploads/none.txt",
            "name": f"numbers_{i}.txt [{country}]",
            "source_name": f"numbers_{i}.txt",
            "country": country,
            "count": 1000,
            "tag": "WhatsApp",
            "uploaded_at": "2026-10-05T00:00:00+00:00",
        }
        for i, country in enumerate(countries)
    ]
    await otp_bot._save_files(otp_bot.ACTIVE_KEY, entries)
    await otp_bot.save_config({"enabled": True})
    return countries


def _long_start_result(countries: list[str]) -> dict:
    return {
        "ok": True,
        "target_bot": "@PBDxbot",
        "files": [
            {
                "name": f"{c}.txt",
                "country": c,
                "count": 1000,
                "tag": "WhatsApp",
                "added": True,
                "reply": "Added 1000 numbers to " + c + ". " + "x" * 120,
            }
            for c in countries
        ],
        "kept_running": [],
        "limits": {},
    }


# --------------------------------------------------------------------------- #
# Country buttons must name the country, not its position
# --------------------------------------------------------------------------- #
async def test_every_country_keyboard_fits_the_64_byte_limit_with_long_names():
    entries = [
        {"id": f"{i:08x}", "country": name, "count": 3}
        for i, name in enumerate(LONG_NAMES * 40)
    ] + [{"id": f"{i:08x}", "country": f"{LONG_NAMES[0]} {i}"} for i in range(300)]
    key = otp_panel.country_key(LONG_NAMES[0])
    markups = [
        otp_panel.country_keyboard(entries, "fields"),
        otp_panel.country_keyboard(entries, "prec"),
        otp_panel.country_toggle_keyboard(entries, set()),
        otp_panel.removal_keyboard(entries),
        otp_panel.file_delete_keyboard(entries, entries),
        otp_panel.country_field_keyboard(key, LONG_NAMES[0]),
        otp_panel.country_interval_keyboard(key, 10),
        otp_panel.country_threshold_keyboard(key, 0),
        otp_panel.stop_time_keyboard(key, "06:00"),
        otp_panel.start_time_keyboard(key, "06:00"),
        otp_panel.country_runtime_keyboard(key, 0),
        otp_panel.preset_keyboard(["A very long preset name " * 4] * 30, key),
        otp_panel.runtime_keyboard(),
        otp_panel.threshold_keyboard(0),
        otp_panel.stop_time_keyboard(-1, ""),
        otp_panel.start_time_keyboard(-1, ""),
        otp_panel.country_runtime_keyboard(-1, 0),
    ]
    for markup in markups:
        for row in markup.inline_keyboard:
            for button in row:
                assert len(button.callback_data.encode("utf-8")) <= 64, button.callback_data


async def test_bad_dao_on_a_stale_country_panel_never_removes_another_country(runner):
    """The field panel for Bangladesh carried "index 0". Once Bangladesh was
    gone, index 0 was Nigeria - and "Remove" (then "Bad dao") removed Nigeria.
    """
    await _queue("bd.txt", "+880")
    await _queue("ng.txt", "+234")

    picker = await _tap(runner, "otp:ask_country:")
    fields = await _tap(runner, _button(picker.message.answers[-1]["markup"], "Bangladesh"))
    bad_dao = _button(fields.message.answers[-1]["markup"], "Remove")

    await otp_bot.remove_by_country("Bangladesh")  # e.g. its run finished
    await _tap(runner, bad_dao)

    assert [e["country"] for e in await otp_bot.get_queue()] == ["Nigeria"]


async def test_a_country_button_still_means_that_country_after_a_restart_reorders(runner):
    """Every start puts the restarted country at the END of the active list,
    so a button rendered before it pointed at a different country after.
    """
    set_userbot(_Userbot())
    await _queue("bd.txt", "+880")
    await _queue("ng.txt", "+234")
    assert (await otp_bot.start_automation())["ok"]

    picker = await _tap(runner, "otp:ask_country:")
    fields = await _tap(runner, _button(picker.message.answers[-1]["markup"], "Bangladesh"))
    off = _button(fields.message.answers[-1]["markup"], "Pause")

    # A fresh Bangladesh file restarts Bangladesh - and reorders the list.
    await _queue("bd2.txt", "+880")
    assert (await otp_bot.start_automation())["ok"]
    assert otp_panel.country_names(await otp_bot.get_active_files())[0] == "Nigeria"

    await _tap(runner, off)

    assert await otp_schedule.is_paused("Bangladesh") is True
    assert await otp_schedule.is_paused("Nigeria") is False


async def test_a_negative_index_does_not_wrap_round_to_the_last_country(runner):
    await _queue("bd.txt", "+880")
    await _queue("ng.txt", "+234")

    query = await _tap(runner, "otp:cpause:-1")

    assert await otp_schedule.is_paused("Nigeria") is False
    assert query.acks and query.acks[-1]["alert"] is True


async def test_custom_global_stop_time_is_global_not_the_last_country(runner):
    """"Off at" -> "Other time" is the ALL-countries picker (-1). It used to
    take names[-1]: the typed time was saved for whichever country happened
    to be last, and with nothing queued the button crashed outright.
    """
    await _queue("bd.txt", "+880")
    await _queue("ng.txt", "+234")

    ask = await _tap(runner, "otp:ask_stopall:")
    custom = _button(ask.message.answers[-1]["markup"], "Other time")
    query = await _tap(runner, custom)

    pending = await otp_bot.get_pending_input()
    assert pending["field"] == "stop_at"
    assert pending.get("country") is None
    assert query.message.answers, "the owner must be asked for the time"


async def test_custom_global_stop_time_works_with_nothing_queued(runner):
    ask = await _tap(runner, "otp:ask_stopall:")
    custom = _button(ask.message.answers[-1]["markup"], "Other time")
    query = await _tap(runner, custom)

    assert query.acks, "the button must be acknowledged, not left spinning"
    assert (await otp_bot.get_pending_input())["field"] == "stop_at"


async def test_the_wipe_mode_warning_no_longer_promises_useddelete(runner):
    query = await _tap(runner, "otp:clean:force")
    assert query.message.answers
    assert "/useddelete" not in query.message.answers[-1]["text"]


# --------------------------------------------------------------------------- #
# Telegram's 4096-character cap
# --------------------------------------------------------------------------- #
async def test_the_panel_refreshes_even_when_the_status_is_huge(runner):
    await _many_active()

    query = await _tap(runner, "otp:status:")

    assert query.message.edits, "an over-long edit was rejected and swallowed"
    assert all(len(e["text"]) <= TELEGRAM_MAX for e in query.message.edits)


async def test_otpbot_status_arrives_when_it_is_longer_than_one_message(runner):
    await _many_active(150)
    message = _PlainMessage(_owner(runner))

    await _callback(runner, "_otpbot")(message, _Command(""))

    assert len(message.answers) == 1
    assert len(message.answers[0]["text"]) <= TELEGRAM_MAX
    assert "more" in message.everything()  # the rows that did not fit are counted
    assert message.answers[-1]["markup"] is not None  # the panel is still attached


async def test_a_long_start_report_from_the_start_button_reaches_the_owner(runner, monkeypatch):
    countries = [f"{LONG_NAMES[i % 3]} {i}" for i in range(40)]

    async def _fake_start(**k: Any) -> dict:
        return _long_start_result(countries)

    monkeypatch.setattr(otp_bot, "start_automation", _fake_start)
    query = await _tap(runner, "otp:start:")

    assert query.acks[0]["text"], "acknowledged before the long start"
    assert countries[-1] in query.message.everything()
    texts = [m["text"] for m in query.message.answers + query.message.edits]
    assert all(len(t) <= TELEGRAM_MAX for t in texts)


# --------------------------------------------------------------------------- #
# A button that fails must still answer
# --------------------------------------------------------------------------- #
async def test_a_corrupted_number_in_a_button_does_not_leave_it_spinning(runner):
    query = await _tap(runner, "otp:int:abc")
    assert query.acks


async def test_a_crashing_button_tells_the_owner(runner, monkeypatch):
    async def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(otp_bot, "set_cleanup_mode", _boom)
    query = await _tap(runner, "otp:clean:used")

    assert query.acks
    assert "❌" in query.message.everything()
    assert "database is locked" in query.message.everything()


# --------------------------------------------------------------------------- #
# Document uploads in the OTP thread
# --------------------------------------------------------------------------- #
async def _upload(runner: Any, name: str, data: bytes, caption: str = "") -> _DocumentMessage:
    message = _DocumentMessage(_owner(runner), name, data, caption)
    await _callback(runner, "_document")(message)
    return message


async def test_an_upload_is_queued_with_its_caption_and_auto_started(runner, monkeypatch):
    await otp_bot.ensure_thread(_owner(runner))
    calls: list[str] = []

    async def _fake_auto_start() -> None:
        calls.append("auto")
        return None

    monkeypatch.setattr(otp_bot, "maybe_auto_start", _fake_auto_start)
    numbers = "\n".join(f"+8801711{i:06d}" for i in range(3)).encode()
    message = await _upload(runner, "bd.txt", numbers, "whatsapp 20h")

    queue = await otp_bot.get_queue()
    assert [e["tag"] for e in queue] == ["WhatsApp"]
    assert queue[0]["caption"] == "whatsapp 20h"
    assert calls == ["auto"]
    assert "Bangladesh" in message.everything()


async def test_an_upload_in_the_otp_thread_is_not_attached_to_the_next_task(runner):
    """The file is already queued for the bot. Leaving it as pending_upload
    attached it to the owner's next unrelated job in any thread.
    """
    from app.db import repo
    from app.db.base import session_scope

    await otp_bot.ensure_thread(_owner(runner))
    numbers = "\n".join(f"+8801711{i:06d}" for i in range(3)).encode()
    await _upload(runner, "bd.txt", numbers, "whatsapp 20h")

    async with session_scope() as session:
        row = await repo.ensure_session(session, _owner(runner))
        assert "pending_upload" not in (row.context or {})


async def test_a_spreadsheet_is_refused_not_queued_as_numbers(runner):
    await otp_bot.ensure_thread(_owner(runner))
    xlsx = b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00[Content_Types].xml\n\x00\x01\x02 1234567"

    message = await _upload(runner, "numbers.xlsx", xlsx, "whatsapp")

    assert await otp_bot.get_queue() == []
    assert ".txt" in message.everything()


async def test_an_empty_file_is_reported_and_starts_nothing(runner, monkeypatch):
    await otp_bot.ensure_thread(_owner(runner))
    calls: list[str] = []

    async def _fake_auto_start() -> None:
        calls.append("auto")
        return None

    monkeypatch.setattr(otp_bot, "maybe_auto_start", _fake_auto_start)
    message = await _upload(runner, "empty.txt", b"\n\n  \n")

    assert await otp_bot.get_queue() == []
    assert calls == []
    assert "number" in message.everything().lower()


async def test_a_failing_upload_is_reported_not_swallowed(runner, monkeypatch):
    await otp_bot.ensure_thread(_owner(runner))

    async def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("disk full")

    monkeypatch.setattr(otp_bot, "enqueue_file", _boom)
    message = await _upload(runner, "bd.txt", b"+8801711000001\n")

    assert "❌" in message.everything()
    assert "disk full" in message.everything()


async def test_a_long_auto_start_report_on_upload_still_arrives(runner, monkeypatch):
    await otp_bot.ensure_thread(_owner(runner))
    countries = [f"{LONG_NAMES[i % 3]} {i}" for i in range(40)]

    async def _fake_auto_start() -> dict:
        result = _long_start_result(countries)
        result["auto"] = True
        return result

    monkeypatch.setattr(otp_bot, "maybe_auto_start", _fake_auto_start)
    message = await _upload(runner, "bd.txt", b"+8801711000001\n", "whatsapp 20h")

    assert message.answers
    assert all(len(a["text"]) <= TELEGRAM_MAX for a in message.answers)
    assert countries[-1] in message.everything()


# --------------------------------------------------------------------------- #
# Control words win over an open question
# --------------------------------------------------------------------------- #
CHAT_ID = 42


async def _ask_the_service_question() -> dict:
    from app.agent.conversation import handle_message

    await otp_bot.ensure_thread(CHAT_ID)
    entry = await _queue("plain.txt", "+880", caption="20h")  # no service stated
    assert entry["tag"] is None
    reply = await handle_message(CHAT_ID, CHAT_ID, "start")
    assert await otp_bot.get_awaiting_tag_entry() is not None, reply.text
    return entry


async def test_stop_during_the_service_question_does_not_start_a_run_tagged_stop(environment):
    from app.agent.conversation import handle_message

    entry = await _ask_the_service_question()
    await handle_message(CHAT_ID, CHAT_ID, "stop")

    queue = {e["id"]: e for e in await otp_bot.get_queue()}
    assert entry["id"] in queue, "the file must still be waiting, not started"
    assert queue[entry["id"]]["tag"] is None
    assert (await otp_bot.get_config())["enabled"] is False


async def test_status_during_the_service_question_answers_status(environment):
    from app.agent.conversation import handle_message

    entry = await _ask_the_service_question()
    reply = await handle_message(CHAT_ID, CHAT_ID, "status")

    assert "OTP-bot automation" in reply.text
    queue = {e["id"]: e for e in await otp_bot.get_queue()}
    assert queue[entry["id"]]["tag"] is None
    assert await otp_bot.get_awaiting_tag_entry() is not None  # still asked


async def test_stop_during_the_run_length_question_stops_the_run(environment):
    from app.agent.conversation import handle_message

    await otp_bot.ensure_thread(CHAT_ID)
    await otp_bot.save_config({"enabled": True})
    await otp_bot.set_awaiting_runtime(["Bangladesh"])

    await handle_message(CHAT_ID, CHAT_ID, "stop")

    assert (await otp_bot.get_config())["enabled"] is False


async def test_bad_dao_with_nothing_open_is_answered_without_the_llm(environment):
    """With no question open, a bare "bad dao" fell through to the router,
    which filed it as a background JOB ("On it ... ID: ...") - an agent task
    told only "remove", with no idea what.
    """
    from app.agent.conversation import handle_message
    from app.agent.router import Intent
    from app.llm import LLMError

    class _DownLLM:
        async def chat(self, *a: Any, **k: Any) -> Any:
            raise LLMError("every provider is down")

    await otp_bot.ensure_thread(CHAT_ID)
    reply = await handle_message(CHAT_ID, CHAT_ID, "bad dao", llm=_DownLLM())

    assert reply.intent is Intent.CONTROL
    assert not reply.created_task
    assert "bad dao" in reply.text.lower()  # tells the owner what to type
