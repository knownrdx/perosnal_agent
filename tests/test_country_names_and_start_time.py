"""Two live bugs the owner caught, and the defaults they forced.

1. The bot spells countries its own way. Ours said "Congo (DRC)", the
   reply said "DR Congo", the letters-only prefix test found no match, an
   absent match read as ZERO STOCK, and the automation cleaned up and
   re-added the entire file on every cycle - 16,499 duplicates skipped
   every five minutes, all night.

2. A start time set for one upload stayed on the country. The next file
   for that country silently inherited it and sat waiting for a time the
   owner had set days earlier.
"""

import pytest

from app.automation import country_names, otp_bot, otp_schedule

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------- #
# the same country, spelled differently
# --------------------------------------------------------------------- #

def test_the_two_spellings_that_cost_16499_duplicates():
    """The exact pair from the owner's screenshot."""
    assert country_names.same_country("Congo (DRC)", "DR Congo")


def test_stock_is_found_however_the_bot_spelled_it():
    stock = {"DR Congo": 3501, "Nigeria": 120}
    assert otp_bot._stock_for(stock, "Congo (DRC)") == 3501


def test_zero_stock_and_missing_stock_stay_different():
    """An explicit zero must not be confused with "country not listed"."""
    assert otp_bot._stock_for({"DR Congo": 0}, "Congo (DRC)") == 0
    assert otp_bot._stock_for({"Nigeria": 5}, "Congo (DRC)") is None


@pytest.mark.parametrize("left,right", [
    ("Congo (DRC)", "Congo Kinshasa"),
    ("Central African Rep.", "Central African Republic"),
    ("Cote d'Ivoire", "Ivory Coast"),
    ("Myanmar", "Myanmar (Burma)"),
    ("Viet Nam", "Vietnam"),
    ("United States", "United States of America"),
    ("UAE", "United Arab Emirates"),
    ("South Korea", "Korea"),
    ("Guinea", "Republic of Guinea"),
])
def test_names_that_mean_the_same_country(left, right):
    assert country_names.same_country(left, right)


@pytest.mark.parametrize("left,right", [
    # Each of these pairs is two countries with two separate stocks.
    # Merging them files numbers under the wrong name, which is worse than
    # the missed match this whole module exists to fix.
    ("Niger", "Nigeria"),
    ("Sudan", "South Sudan"),
    ("South Korea", "North Korea"),
    ("Guinea", "Guinea-Bissau"),
    ("Guinea", "Equatorial Guinea"),
    ("Guinea", "Papua New Guinea"),
    ("Congo (DRC)", "Republic of the Congo"),
    ("Dominica", "Dominican Republic"),
    ("Samoa", "American Samoa"),
    ("US Virgin Islands", "British Virgin Islands"),
    ("India", "Indonesia"),
    ("Mali", "Malawi"),
    ("Chad", "Chile"),
])
def test_names_that_are_different_countries(left, right):
    assert not country_names.same_country(left, right)


def test_an_unknown_spelling_in_a_colliding_family_matches_nothing():
    """Better to ask again than to guess which Congo."""
    assert not country_names.same_country("Congo Something", "DR Congo")


# --------------------------------------------------------------------- #
# start time is per upload, not per country
# --------------------------------------------------------------------- #

async def _upload(name: str = "bd.txt", prefix: str = "+880") -> str:
    """A real file in the real workspace, like the other suites do."""
    from app.config import get_settings
    from app.security import rel_path

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / name
    path.write_text(
        "\n".join(f"{prefix}1711{i:06d}" for i in range(4)), encoding="utf-8"
    )
    return rel_path(path)


async def test_a_new_file_clears_the_last_upload_s_start_time(environment):
    """Last night's "start at 21:00" must not hold up today's file."""
    await otp_schedule.set_country_settings("Bangladesh", {"start_at": "21:00"})
    assert (await otp_schedule.get_country_settings("Bangladesh"))["start_at"] == "21:00"

    rel = await _upload()
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp 20h")

    settings = await otp_schedule.get_country_settings("Bangladesh")
    assert not settings.get("start_at"), "a stale start time survived the upload"


async def test_a_caption_start_time_still_applies(environment):
    """Clearing the stale value must not clear the one just stated."""
    rel = await _upload()
    await otp_bot.enqueue_file(rel, "bd.txt", "whatsapp shuru 21:00")

    settings = await otp_schedule.get_country_settings("Bangladesh")
    assert settings.get("start_at") == "21:00"


async def test_the_default_start_time_is_four_in_the_morning(environment):
    """The owner's choice: the target bot is quietest then."""
    assert otp_bot.DEFAULT_CONFIG["start_at"] == "04:00"


# --------------------------------------------------------------------- #
# changing the default must not pause work already running
# --------------------------------------------------------------------- #

async def test_a_running_country_is_not_paused_by_a_new_default(environment):
    """The trap in shipping a 04:00 default.

    A country armed at 01:00 with no start time is running. Introducing a
    global 04:00 default must not make it "waiting" until 04:00 - it would
    stop refilling for three hours with nothing to explain why.
    """
    await otp_schedule.begin_run("Bangladesh", start_at="")

    paused = await otp_schedule.has_started("Bangladesh", {"start_at": "04:00"})
    assert paused is True


async def test_a_country_armed_with_a_start_time_does_wait(environment):
    """The gate still works for the run that actually asked for one."""
    await otp_schedule.begin_run("Nigeria", start_at="23:59")

    assert await otp_schedule.has_started("Nigeria", {"start_at": ""}) is False


async def test_the_gate_is_released_once_the_numbers_go_in(environment):
    """_add_after_wait re-bases the run with no gate - the wait is over."""
    await otp_schedule.begin_run("Uganda", start_at="23:59")
    assert await otp_schedule.has_started("Uganda", {}) is False

    await otp_schedule.begin_run("Uganda")          # what _add_after_wait does
    assert await otp_schedule.has_started("Uganda", {}) is True


async def test_a_run_from_before_this_field_is_treated_as_started(environment):
    """Old rows have no "gate" key; they are running, not waiting."""
    state = await otp_schedule._get_run_state()
    state[otp_schedule._key("Kenya")] = {
        "started_at": otp_schedule._now().isoformat(),
        "armed_at": otp_schedule._now().isoformat(),
        "refills": 0,
        "display_name": "Kenya",
    }
    await otp_schedule._save_run_state(state)

    assert await otp_schedule.has_started("Kenya", {"start_at": "04:00"}) is True


async def test_an_old_empty_start_time_adopts_the_new_default(environment):
    """The live box already holds start_at="" from before this setting.

    That empty string was written on every save, so it means "nobody chose",
    not "start immediately" - and left alone it would mask the 04:00 default
    forever on exactly the install the default was added for.
    """
    from app.db import repo
    from app.db.base import session_scope

    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY,
                               {"enabled": True, "start_at": ""})

    cfg = await otp_bot.get_config()
    assert cfg["start_at"] == "04:00"


async def test_choosing_start_now_is_respected(environment):
    """Deliberately asking to start at once must survive the migration."""
    from app.db import repo
    from app.db.base import session_scope

    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY,
                               {"enabled": True, "start_at": otp_bot.START_NOW})

    cfg = await otp_bot.get_config()
    assert cfg["start_at"] == otp_bot.START_NOW
    # ...and it means "no wait" everywhere a start time is read.
    assert otp_schedule._is_now(cfg["start_at"]) is True


async def test_a_chosen_clock_time_is_untouched(environment):
    from app.db import repo
    from app.db.base import session_scope

    async with session_scope() as session:
        await repo.set_setting(session, otp_bot.SETTING_KEY,
                               {"enabled": True, "start_at": "21:00"})

    assert (await otp_bot.get_config())["start_at"] == "21:00"


def test_every_country_is_still_detected_from_its_numbers():
    """The owner asked whether all countries were added. They are - the
    failure he saw was name matching, not detection. This walks every
    region the library knows so a regression cannot hide in the long tail.
    """
    import phonenumbers
    from phonenumbers import PhoneMetadata

    from app.automation import phone_countries

    unresolved = []
    for region in sorted(phonenumbers.SUPPORTED_REGIONS):
        meta = PhoneMetadata.metadata_for_region(region)
        example = meta.mobile.example_number if meta and meta.mobile else None
        if not example:
            continue
        code = phonenumbers.country_code_for_region(region)
        name = phone_countries.country_of(f"{code}{example}")
        if name == phone_countries.UNKNOWN or name.startswith("+"):
            unresolved.append((region, code, name))

    assert not unresolved, f"{len(unresolved)} regions unresolved: {unresolved[:10]}"
