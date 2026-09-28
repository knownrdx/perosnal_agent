"""Caption parsing, wall-clock start times, and full-coverage country detection.

Each test here pins a behaviour the owner asked for after living with the
previous one: a caption that already says "bangladesh whatsapp 20h" must not
be followed by three questions asking exactly that, a run scheduled for 21:00
must not touch the target bot at 14:00, and a perfectly ordinary Lithuanian
number must not come back as "+370".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.automation import otp_bot, otp_caption, otp_schedule, phone_countries

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Country detection
# --------------------------------------------------------------------------- #
def test_countries_outside_the_old_hand_written_table_now_resolve():
    """The old table held ~100 hand-picked entries; everything else became a
    "+370" bucket the target bot has never heard of."""
    assert phone_countries.country_of("+37061234567") == "Lithuania"
    assert phone_countries.country_of("+50612345678") == "Costa Rica"
    assert phone_countries.country_of("+996555123456") == "Kyrgyzstan"
    assert phone_countries.country_of("+35799123456") == "Cyprus"


def test_shared_calling_codes_split_by_area_code():
    """+1 is 25 different countries to the target bot, not one."""
    assert phone_countries.country_of("+18091234567") == "Dominican Republic"
    assert phone_countries.country_of("+14165551234") == "Canada"
    assert phone_countries.country_of("+12125551234") == "United States"
    # +7 is Russia AND Kazakhstan; a flat prefix table merges them.
    assert phone_countries.country_of("+79161234567") == "Russia"
    assert phone_countries.country_of("+77012345678") == "Kazakhstan"


def test_the_countries_the_owner_actually_uploads_still_resolve():
    """Regression guard: the new table must not lose the old one's cases."""
    for number, expected in (
        ("+8801712345678", "Bangladesh"),
        ("+2367712345", "Central African Republic"),
        ("+919876543210", "India"),
        ("+2348012345678", "Nigeria"),
        ("+6281234567890", "Indonesia"),
        ("+254712345678", "Kenya"),
    ):
        assert phone_countries.country_of(number) == expected


def test_digits_only_and_junk_lines_are_handled():
    assert phone_countries.country_of("880 1712-345678") == "Bangladesh"
    assert phone_countries.country_of("") == "Unknown"
    assert phone_countries.country_of("not a number") == "Unknown"


def test_an_unknown_calling_code_is_surfaced_not_silently_bucketed():
    """A missing entry must be visible and fixable, not quietly wrong."""
    assert phone_countries.country_of("+99912345678").startswith("+")


# --------------------------------------------------------------------------- #
# Caption parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "caption,expected",
    [
        ("bangladesh whatsapp 20h",
         {"country": "Bangladesh", "tag": "WhatsApp", "run_minutes": 1200}),
        ("shuru 21:00 theke bondho 06:00",
         {"start_at": "21:00", "stop_at": "06:00"}),
        ("nigeria telegram 23:30 e off",
         {"country": "Nigeria", "tag": "Telegram", "stop_at": "23:30"}),
        ("9pm porjonto", {"stop_at": "21:00"}),
        ("india 2 din cholbe", {"country": "India", "run_minutes": 2880}),
        ("tag fb bangladesh", {"tag": "Facebook", "country": "Bangladesh"}),
    ],
)
def test_a_caption_states_what_would_otherwise_be_asked(caption, expected):
    parsed = otp_caption.parse_caption(caption)
    for key, value in expected.items():
        assert parsed.get(key) == value, f"{caption!r} -> {key}"


def test_no_stop_is_expressible_in_a_caption():
    parsed = otp_caption.parse_caption("facebook, kono stop nai")
    assert parsed["no_stop"] is True
    assert parsed["run_minutes"] == 0


def test_a_caption_that_says_nothing_parses_to_nothing():
    """Silence must leave the normal questions exactly as they were."""
    assert otp_caption.parse_caption("") == {}
    assert otp_caption.parse_caption("just some numbers") == {}
    assert otp_caption.parse_caption("bd file") == {}


def test_short_aliases_need_an_explicit_introduction():
    """"wa" and "x" are ordinary words; guessing them wrong files numbers
    under the wrong service, which is worse than one extra question."""
    assert "tag" not in otp_caption.parse_caption("wa")
    assert otp_caption.parse_caption("service wa")["tag"] == "WhatsApp"


def test_a_delay_is_not_read_as_a_run_length():
    assert "run_minutes" not in otp_caption.parse_caption("2 ghonta por start")


# --------------------------------------------------------------------------- #
# Caption -> queue
# --------------------------------------------------------------------------- #
async def _write(environment, name: str, prefix: str = "+880") -> str:
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(
        "\n".join(f"{prefix}1711{i:06d}" for i in range(4)), encoding="utf-8"
    )
    return rel_path(path)


async def test_a_caption_answers_the_tag_question(environment):
    rel = await _write(environment, "bd.txt")
    analysis = await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp 20h")

    assert analysis["entries"][0]["tag"] == "WhatsApp"
    # No tag question is left open, so nothing is waiting on the owner.
    assert await otp_bot.next_untagged_entry() is None


async def test_a_caption_country_replaces_what_the_numbers_say(environment):
    """The owner knows which stock a file is for; a prefix table does not."""
    rel = await _write(environment, "mystery.txt", "+880")
    analysis = await otp_bot.enqueue_file(rel, "mystery.txt", "nigeria whatsapp 20h")

    assert list(analysis["countries"]) == ["Nigeria"]
    assert analysis["entries"][0]["country"] == "Nigeria"


async def test_a_mixed_file_keeps_its_per_number_split(environment):
    """Overriding a genuinely mixed file would merge distinct stock."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / "mixed.txt"
    path.write_text("+8801711000001\n+2348011000002\n", encoding="utf-8")

    analysis = await otp_bot.enqueue_file(
        rel_path(path), "mixed.txt", "india whatsapp"
    )
    assert set(analysis["countries"]) == {"Bangladesh", "Nigeria"}


async def test_caption_timing_lands_on_the_country_settings(environment):
    rel = await _write(environment, "bd.txt")
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp shuru 21:00 bondho 06:00")

    settings = await otp_schedule.get_country_settings("Bangladesh")
    assert settings["start_at"] == "21:00"
    assert settings["stop_at"] == "06:00"


# --------------------------------------------------------------------------- #
# Wall-clock start times
# --------------------------------------------------------------------------- #
def test_a_start_time_is_read_in_dubai_time():
    begins = otp_schedule.next_occurrence("21:00")
    assert begins is not None
    assert begins.astimezone(otp_schedule.DUBAI_TZ).hour == 21


def test_a_start_time_already_past_today_means_tomorrow():
    """A file queued at 22:00 asking for 21:00 wants 21:00 TOMORROW.

    Anchoring on "today at 00:00" instead would make the run begin instantly,
    which is the opposite of what a start time is for.
    """
    armed = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)   # 22:00 Dubai
    begins = otp_schedule.next_occurrence("21:00", after=armed)
    assert begins is not None
    assert begins > armed
    assert (begins - armed) > timedelta(hours=20)
    assert otp_schedule._start_time_reached("21:00", armed.isoformat()) is False


def test_no_start_time_means_start_now():
    assert otp_schedule._start_time_reached("", None) is True


class _FakeUserbot:
    """Minimal stand-in: every send succeeds, every read returns a fresh id.

    A fresh id per read matters - the automation waits for a message NEWER
    than the one it saw before sending, so a static id would hang.
    """

    def __init__(self) -> None:
        self._reply_id = 0

    async def send_message(self, target, text, reply_to=None):
        self._reply_id += 1
        return {"sent": True, "message_id": self._reply_id}

    async def send_file(self, target, path, caption="", reply_to=None):
        self._reply_id += 1
        return {"sent": True, "message_id": self._reply_id}

    async def read_messages(self, target, limit=20):
        self._reply_id += 1
        return [{"text": "Added.", "id": self._reply_id, "out": False}]

    async def delete_messages(self, target, message_ids):
        return {"deleted": len(message_ids)}


async def test_a_scheduled_country_is_not_added_until_its_time(environment):
    """The whole point of "start at 21:00" is that nothing is sent before it."""
    from app.integrations.telegram_user import set_userbot

    rel = await _write(environment, "bd.txt")
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp shuru 21:00 20h")
    # Make sure the start time is genuinely in the future for this run.
    now_dubai = datetime.now(otp_schedule.DUBAI_TZ)
    future = (now_dubai + timedelta(hours=3)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": future})

    set_userbot(_FakeUserbot())
    try:
        result = await otp_bot.start_automation()
    finally:
        set_userbot(None)

    assert result["ok"] is True
    assert result["files"][0]["added"] is False
    assert result["files"][0]["waiting_start"] == future
    # It IS being tracked - it simply has not run yet.
    assert len(await otp_bot.get_active_files()) == 1
    assert await otp_schedule.has_started("Bangladesh", await otp_bot.get_config()) is False


async def test_a_waiting_country_is_never_declared_finished(environment):
    """A "20h run starting at 21:00" uploaded at 09:00 must not be called
    finished at 17:00, before it has sent anything at all."""
    cfg = await otp_bot.get_config()
    now_dubai = datetime.now(otp_schedule.DUBAI_TZ)
    future = (now_dubai + timedelta(hours=2)).strftime("%H:%M")

    await otp_schedule.set_country_settings(
        "Bangladesh", {"start_at": future, "run_minutes": 30}
    )
    await otp_schedule.begin_run("Bangladesh")

    # 30 minutes of "run time" have notionally passed, but the run has not
    # begun, so there is nothing to finish.
    assert await otp_schedule.finished_reason("Bangladesh", cfg) is None


async def test_the_run_clock_starts_when_the_start_time_arrives(environment):
    """"20h, starting 21:00" means twenty hours FROM 21:00."""
    cfg = await otp_bot.get_config()
    # Armed an hour ago, scheduled to start in an hour, running for 90 min.
    armed = datetime.now(timezone.utc) - timedelta(hours=1)
    soon = (datetime.now(otp_schedule.DUBAI_TZ) + timedelta(hours=1)).strftime("%H:%M")
    await otp_schedule.set_country_settings(
        "Bangladesh", {"start_at": soon, "run_minutes": 90}
    )
    await otp_schedule.begin_run("Bangladesh", armed_at=armed.isoformat())

    state = await otp_schedule.get_run_state("Bangladesh")
    effective = otp_schedule._effective_start(
        await otp_schedule.effective_config("Bangladesh", cfg), state
    )
    began = datetime.fromisoformat(effective)
    # The clock begins in the future (at the start time), not an hour ago.
    assert began > datetime.now(timezone.utc)


async def test_a_scheduled_country_is_skipped_by_the_due_check(environment):
    cfg = await otp_bot.get_config()
    future = (datetime.now(otp_schedule.DUBAI_TZ) + timedelta(hours=4)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": future})
    await otp_schedule.begin_run("Bangladesh")
    # Due immediately, were it not waiting to start.
    await otp_schedule.arm_country("Bangladesh", 1)

    due = await otp_schedule.due_countries(["Bangladesh"], cfg)
    assert due == []


async def test_the_overview_says_a_country_is_waiting(environment):
    cfg = await otp_bot.get_config()
    future = (datetime.now(otp_schedule.DUBAI_TZ) + timedelta(hours=4)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": future})
    await otp_schedule.begin_run("Bangladesh")

    row = (await otp_schedule.schedule_overview(["Bangladesh"], cfg))[0]
    assert row["waiting_to_start"] is True
    assert row["start_at"] == future
    assert row["starts_in_seconds"] > 0


async def test_the_numbers_are_sent_when_the_start_time_arrives(environment, monkeypatch):
    """The cycle after the start time is where the first add actually happens.

    Without this the country would sit in the active set forever, being
    "checked" against stock it never contributed to.
    """
    from app.integrations.telegram_user import set_userbot

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)

    rel = await _write(environment, "bd.txt")
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp 20h")
    future = (datetime.now(otp_schedule.DUBAI_TZ) + timedelta(hours=3)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": future})

    set_userbot(_FakeUserbot())
    try:
        await otp_bot.start_automation()
        active = await otp_bot.get_active_files()
        assert active[0].get("waiting_start") == future

        # The start time arrives: clear it and let a forced cycle run.
        await otp_schedule.set_country_settings("Bangladesh", {"start_at": ""})
        result = await otp_bot.run_cycle(force=True)
    finally:
        set_userbot(None)

    assert [item["country"] for item in result.started_now] == ["Bangladesh"]
    assert result.started_now[0]["added"] is True
    # The marker is gone, so the next cycle treats it as an ordinary country.
    assert (await otp_bot.get_active_files())[0].get("waiting_start") is None


# --------------------------------------------------------------------------- #
# Run-length parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("20h", 1200), ("20 ghonta", 1200), ("90 min", 90), ("2 din", 2880),
        ("1200", 1200), ("limit nai", 0), ("sara rat", 0), ("0", 0),
    ],
)
def test_run_length_is_read_the_way_the_owner_writes_it(text, expected):
    assert otp_bot.parse_run_minutes(text) == expected


def test_an_unreadable_run_length_says_so_rather_than_guessing():
    with pytest.raises(ValueError):
        otp_bot.parse_run_minutes("whenever")


@pytest.mark.parametrize(
    "minutes,expected",
    [(1200, "20h"), (480, "8h"), (2880, "2 din"), (90, "1h 30m"), (0, "kono limit nai")],
)
def test_run_length_is_displayed_readably(minutes, expected):
    assert otp_bot.format_run_minutes(minutes) == expected


async def test_a_country_with_timing_already_set_is_not_asked_again(environment):
    rel = await _write(environment, "bd.txt")
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp 20h")

    queue = await otp_bot.get_queue()
    assert await otp_bot.countries_needing_runtime(queue) == []


async def test_a_country_with_no_timing_is_asked(environment):
    rel = await _write(environment, "bd.txt")
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp")

    queue = await otp_bot.get_queue()
    assert await otp_bot.countries_needing_runtime(queue) == ["Bangladesh"]
