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
