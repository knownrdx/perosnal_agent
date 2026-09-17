"""Per-country scheduling and presets.

The behaviour under test is the reason this module exists: countries drain
at wildly different rates, so they must not share one clock, and a restart
must not reset every country's timer to "now".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.automation import otp_schedule

asyncio_test = pytest.mark.asyncio

BASE = {
    "interval_minutes": 10,
    "quota_threshold": 0,
    "limit": 4,
    "count": 4,
    "tag": "General",
    "force_delete_before_add": False,
}


async def _backdate(country: str) -> None:
    due = await otp_schedule._get_due_map()
    due[otp_schedule._key(country)] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    await otp_schedule._save_due_map(due)


# --------------------------------------------------------------------------- #
# Per-country overrides
# --------------------------------------------------------------------------- #
@asyncio_test
async def test_a_country_without_overrides_uses_the_global_config(environment):
    cfg = await otp_schedule.effective_config("Bangladesh", BASE)
    assert cfg["interval_minutes"] == 10
    assert cfg["limit"] == 4


@asyncio_test
async def test_overrides_layer_on_top_of_the_global_config(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5, "limit": 20})
    cfg = await otp_schedule.effective_config("Bangladesh", BASE)

    assert cfg["interval_minutes"] == 5
    assert cfg["limit"] == 20
    # Untouched fields still come from the global config.
    assert cfg["count"] == 4


@asyncio_test
async def test_countries_do_not_inherit_each_others_settings(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    nigeria = await otp_schedule.effective_config("Nigeria", BASE)
    assert nigeria["interval_minutes"] == 10


@asyncio_test
async def test_country_names_are_matched_loosely(environment):
    """A name can arrive from a filename guess, a button, or typing - they
    must all address the same country rather than creating parallel entries
    with divergent settings.
    """
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    for spelling in ("bangladesh", "  BANGLADESH  ", "Bangladesh"):
        cfg = await otp_schedule.effective_config(spelling, BASE)
        assert cfg["interval_minutes"] == 5


@asyncio_test
async def test_setting_a_field_to_none_hands_it_back_to_the_global_default(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": None})

    cfg = await otp_schedule.effective_config("Bangladesh", BASE)
    assert cfg["interval_minutes"] == 10


# --------------------------------------------------------------------------- #
# Independent timers
# --------------------------------------------------------------------------- #
@asyncio_test
async def test_a_new_country_is_armed_not_fired(environment):
    """Otherwise every restart triggers an immediate refill of everything."""
    ready = await otp_schedule.due_countries(["Bangladesh"], BASE)
    assert ready == []
    assert await otp_schedule.get_due_at("Bangladesh") is not None


@asyncio_test
async def test_only_the_country_whose_timer_expired_comes_back(environment):
    await otp_schedule.due_countries(["Bangladesh", "Nigeria"], BASE)
    await _backdate("Bangladesh")

    ready = await otp_schedule.due_countries(["Bangladesh", "Nigeria"], BASE)
    assert ready == ["Bangladesh"]


@asyncio_test
async def test_firing_rearms_on_that_countrys_own_interval(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    await otp_schedule.due_countries(["Bangladesh"], BASE)
    await _backdate("Bangladesh")

    assert await otp_schedule.due_countries(["Bangladesh"], BASE) == ["Bangladesh"]
    # Fires once, then waits its own 5 minutes - not the global 10.
    assert await otp_schedule.due_countries(["Bangladesh"], BASE) == []

    when = await otp_schedule.get_due_at("Bangladesh")
    delta = (when - datetime.now(timezone.utc)).total_seconds()
    assert 4 * 60 < delta <= 5 * 60 + 5


@asyncio_test
async def test_due_times_survive_a_restart(environment):
    """Timers are persisted, so a redeploy does not reset every country to
    "due now" and stampede the target bot.
    """
    await otp_schedule.due_countries(["Bangladesh"], BASE)
    before = await otp_schedule.get_due_at("Bangladesh")

    # Simulate a restart: nothing in memory, read it back from storage.
    after = await otp_schedule.get_due_at("Bangladesh")
    assert before == after
    assert await otp_schedule.due_countries(["Bangladesh"], BASE) == []


@asyncio_test
async def test_scheduler_polls_at_the_shortest_country_interval(environment):
    """A 5-minute country must not be capped by a 30-minute global setting."""
    base = dict(BASE, interval_minutes=30)
    assert await otp_schedule.shortest_interval_minutes(base) == 30

    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    assert await otp_schedule.shortest_interval_minutes(base) == 5


@asyncio_test
async def test_forgetting_a_country_clears_its_timer_and_settings(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    await otp_schedule.due_countries(["Bangladesh"], BASE)

    await otp_schedule.forget_country("Bangladesh")

    assert await otp_schedule.get_due_at("Bangladesh") is None
    assert await otp_schedule.get_country_settings("Bangladesh") == {}


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
@asyncio_test
async def test_low_stock_preset_refills_before_reaching_zero(environment):
    """Waiting for zero means the country is briefly dead; this preset tops
    up while numbers are still left.
    """
    presets = await otp_schedule.get_presets()
    assert "Low-stock refill" in presets
    assert presets["Low-stock refill"]["quota_threshold"] == 200

    await otp_schedule.apply_preset("Low-stock refill", "Bangladesh")
    cfg = await otp_schedule.effective_config("Bangladesh", BASE)
    assert cfg["quota_threshold"] == 200
    assert cfg["force_delete_before_add"] is True


# --------------------------------------------------------------------------- #
# Per-country on/off and a wall-clock stop time
# --------------------------------------------------------------------------- #
@asyncio_test
async def test_pausing_a_country_skips_it_without_losing_anything(environment):
    """Pausing is not removing: the file and every setting stay put."""
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    await otp_schedule.due_countries(["Bangladesh", "Nigeria"], BASE)
    await _backdate("Bangladesh")
    await _backdate("Nigeria")

    await otp_schedule.set_paused("Bangladesh", True)

    ready = await otp_schedule.due_countries(["Bangladesh", "Nigeria"], BASE)
    assert ready == ["Nigeria"]
    # Settings survive the pause.
    assert (await otp_schedule.get_country_settings("Bangladesh"))["interval_minutes"] == 5


@asyncio_test
async def test_resuming_starts_the_clock_from_now(environment):
    """A country paused for hours must not fire the instant it comes back
    just because its old due time went by.
    """
    await otp_schedule.due_countries(["Bangladesh"], BASE)
    await otp_schedule.set_paused("Bangladesh", True)
    await _backdate("Bangladesh")

    await otp_schedule.set_paused("Bangladesh", False)

    assert await otp_schedule.due_countries(["Bangladesh"], BASE) == []
    assert await otp_schedule.is_paused("Bangladesh") is False


@asyncio_test
async def test_pause_is_per_country(environment):
    await otp_schedule.set_paused("Bangladesh", True)
    assert await otp_schedule.is_paused("Bangladesh") is True
    assert await otp_schedule.is_paused("Nigeria") is False


def test_stop_times_are_parsed_forgivingly():
    assert otp_schedule.parse_stop_time("23:30") == "23:30"
    assert otp_schedule.parse_stop_time("9:05") == "09:05"
    assert otp_schedule.parse_stop_time("2330") == "23:30"    # no colon
    assert otp_schedule.parse_stop_time("23.30") == "23:30"   # dot
    assert otp_schedule.parse_stop_time("") == ""             # clears it


def test_an_impossible_stop_time_is_rejected():
    """Stored unvalidated, "25:00" would simply never fire and the owner
    would think the feature was broken.
    """
    for bad in ("25:00", "12:75", "abc", "9pm"):
        with pytest.raises(ValueError):
            otp_schedule.parse_stop_time(bad)


def test_the_clock_reports_both_zones():
    clock = otp_schedule.clock_now()
    assert ":" in clock["utc"] and ":" in clock["dubai"]
    assert "Dubai" in clock["dubai_full"]
    assert "UTC" in clock["utc_full"]

    # Dubai is UTC+4 with no DST, so the gap is always exactly four hours.
    utc_h = int(clock["utc"].split(":")[0])
    dubai_h = int(clock["dubai"].split(":")[0])
    assert (dubai_h - utc_h) % 24 == 4


def test_a_stop_time_later_today_fires_only_after_it_passes():
    now = datetime.now(otp_schedule.DUBAI_TZ)
    started = (now - timedelta(minutes=30)).isoformat()

    an_hour_ahead = (now + timedelta(hours=1)).strftime("%H:%M")
    a_minute_ago = (now - timedelta(minutes=1)).strftime("%H:%M")

    assert otp_schedule._stop_time_passed(an_hour_ahead, started) is False
    assert otp_schedule._stop_time_passed(a_minute_ago, started) is True


def test_an_overnight_stop_time_means_tomorrow():
    """Started 22:00, stop at 01:00 - that is 01:00 the NEXT day, which is
    the whole point of an overnight run. Comparing against "today" would
    finish the run immediately.
    """
    now = datetime.now(otp_schedule.DUBAI_TZ)
    started_two_hours_ago = (now - timedelta(hours=2)).isoformat()
    # A time one hour BEFORE the start: already past on the start day.
    earlier_than_start = (now - timedelta(hours=3)).strftime("%H:%M")

    assert otp_schedule._stop_time_passed(earlier_than_start, started_two_hours_ago) is False


@asyncio_test
async def test_a_country_finishes_at_its_stop_time(environment):
    """A run started earlier today, with a stop time that has since passed."""
    now = datetime.now(otp_schedule.DUBAI_TZ)
    await otp_schedule.begin_run("Bangladesh")

    # Backdate the run two hours, then stop at one hour ago: the stop time
    # falls between the start and now, so it has genuinely passed.
    state = await otp_schedule._get_run_state()
    state["bangladesh"]["started_at"] = (now - timedelta(hours=2)).isoformat()
    await otp_schedule._save_run_state(state)

    passed = (now - timedelta(hours=1)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"stop_at": passed})

    reason = await otp_schedule.finished_reason("Bangladesh", BASE)
    assert reason is not None
    assert passed in reason


@asyncio_test
async def test_a_stop_time_not_yet_reached_keeps_it_running(environment):
    now = datetime.now(otp_schedule.DUBAI_TZ)
    await otp_schedule.begin_run("Bangladesh")
    later = (now + timedelta(hours=2)).strftime("%H:%M")
    await otp_schedule.set_country_settings("Bangladesh", {"stop_at": later})

    assert await otp_schedule.finished_reason("Bangladesh", BASE) is None


@asyncio_test
async def test_no_stop_time_means_it_keeps_going(environment):
    await otp_schedule.begin_run("Bangladesh")
    assert await otp_schedule.finished_reason("Bangladesh", BASE) is None


@asyncio_test
async def test_the_overview_reports_pause_and_stop_time(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"stop_at": "23:30"})
    await otp_schedule.set_paused("Bangladesh", True)

    rows = {r["country"]: r for r in await otp_schedule.schedule_overview(["Bangladesh"], BASE)}
    assert rows["Bangladesh"]["paused"] is True
    assert rows["Bangladesh"]["stop_at"] == "23:30"


@asyncio_test
async def test_builtin_presets_are_available_out_of_the_box(environment):
    presets = await otp_schedule.get_presets()
    assert "Fast burn" in presets
    assert "Replace stock" in presets
    assert presets["Fast burn"]["interval_minutes"] == 5


@asyncio_test
async def test_applying_a_preset_configures_the_country(environment):
    applied = await otp_schedule.apply_preset("Fast burn", "Bangladesh")
    assert applied is not None

    cfg = await otp_schedule.effective_config("Bangladesh", BASE)
    assert cfg["interval_minutes"] == 5
    # And it takes effect now rather than after the old interval expires.
    assert await otp_schedule.get_due_at("Bangladesh") is not None


@asyncio_test
async def test_applying_an_unknown_preset_reports_it(environment):
    assert await otp_schedule.apply_preset("Does not exist", "Bangladesh") is None


@asyncio_test
async def test_owner_presets_can_shadow_a_builtin(environment):
    """So a wrong default can be corrected rather than worked around."""
    await otp_schedule.save_preset("Fast burn", {"interval_minutes": 2, "limit": 9})
    presets = await otp_schedule.get_presets()
    assert presets["Fast burn"]["interval_minutes"] == 2


@asyncio_test
async def test_saving_and_deleting_an_own_preset(environment):
    await otp_schedule.save_preset("My profile", {"interval_minutes": 7, "count": 3})
    assert "My profile" in await otp_schedule.get_presets()

    assert await otp_schedule.delete_preset("My profile") is True
    assert "My profile" not in await otp_schedule.get_presets()


@asyncio_test
async def test_a_builtin_cannot_be_deleted_only_shadowed(environment):
    assert await otp_schedule.delete_preset("Fast burn") is False
    assert "Fast burn" in await otp_schedule.get_presets()


@asyncio_test
async def test_a_preset_needs_a_name(environment):
    with pytest.raises(ValueError):
        await otp_schedule.save_preset("   ", {"interval_minutes": 5})


@asyncio_test
async def test_overview_reports_what_applies_and_when(environment):
    await otp_schedule.set_country_settings("Bangladesh", {"interval_minutes": 5})
    await otp_schedule.due_countries(["Bangladesh", "Nigeria"], BASE)

    rows = {r["country"]: r for r in await otp_schedule.schedule_overview(
        ["Bangladesh", "Nigeria"], BASE
    )}

    assert rows["Bangladesh"]["interval_minutes"] == 5
    assert rows["Bangladesh"]["customised"] == ["interval_minutes"]
    assert rows["Nigeria"]["interval_minutes"] == 10
    assert rows["Nigeria"]["customised"] == []
    assert rows["Bangladesh"]["due_in_seconds"] is not None
