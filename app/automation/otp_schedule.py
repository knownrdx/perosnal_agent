"""Per-country settings, the shared check timer, and reusable presets.

Every country is checked on ONE shared timer (the global interval_minutes):
a single /st reports stock for all of them at once, so per-country timers
only staggered the checks - one country looked at now, the next an hour
later - for no benefit. The next-check time is persisted, so a restart does
not reset the clock to "now".

What a country can still have of its own (all optional - anything not set
falls back to the global config, so a country the owner never touches keeps
behaving exactly as before):
  - threshold, limit, count, service and wipe-before-add mode,
  - its lifecycle: pause, start/stop time, run length, re-add limit.

Presets are the same settings under a name. The target bot has genuinely
per-country knobs (/setlimit <country> <tag> <N>, /setcooldown, and
/setcountrylimit), so "how do I add this country" is a real, reusable
decision worth templating rather than retyping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger

log = get_logger(__name__)

COUNTRY_KEY = "otp_bot_country_settings"
DUE_KEY = "otp_bot_country_due"
PRESET_KEY = "otp_bot_presets"

# The fields a country (or a preset) may override. Everything else stays
# global - target bot, command templates and so on are properties of the
# bot being driven, not of one country. The check interval is deliberately
# NOT here: it is one shared timer for every country.
OVERRIDABLE = (
    "quota_threshold",
    "limit",
    "count",
    "tag",
    "force_delete_before_add",
    # Lifecycle: when should this country stop on its own? Without these a
    # started country runs until the owner remembers to stop it.
    "max_refills",       # stop after N re-adds (0 = no limit)
    "run_minutes",       # stop N minutes after starting (0 = no limit)
    "delete_when_done",  # clear the country's numbers off the bot at the end
    "paused",            # temporarily off without losing its settings/place
    "stop_at",           # wall-clock time to stop, e.g. "23:30" (Dubai time)
    # Wall-clock time to BEGIN, e.g. "21:00" (Dubai). Until it arrives the
    # country sits in the active set doing nothing - the owner uploads during
    # the day for a run that should only touch the target bot at night.
    "start_at",
)

# Dubai is UTC+4 all year - no daylight saving - so a fixed offset is exact
# and needs no tzdata in the container. Times the owner types are read in
# this zone, because that is the clock they are actually looking at.
DUBAI_TZ = timezone(timedelta(hours=4), name="Dubai")


def clock_now() -> dict[str, str]:
    """Current time in both zones, for any UI that offers a stop time.

    Shown side by side deliberately: the server thinks in UTC, the owner
    thinks in Dubai time, and a stop time picked against the wrong one is
    four hours out.
    """
    now = _now()
    local = now.astimezone(DUBAI_TZ)
    return {
        "utc": now.strftime("%H:%M"),
        "dubai": local.strftime("%H:%M"),
        "utc_full": now.strftime("%Y-%m-%d %H:%M UTC"),
        "dubai_full": local.strftime("%Y-%m-%d %H:%M Dubai (UTC+4)"),
    }


# "begin immediately", distinct from "" so an empty string left behind by an
# older version can be told apart from a deliberate choice. Treated as no
# start time wherever one is read.
_START_NOW_WORDS = frozenset({"now", "ekhoni", "ekhuni"})


def _is_now(value: str) -> bool:
    text = str(value or "").strip()
    return not text or text.casefold() in _START_NOW_WORDS


def parse_stop_time(text: str) -> str:
    """Validate a "HH:MM" stop time given in Dubai time.

    Kept as a wall-clock string rather than a timestamp so "stop at 23:30"
    keeps meaning 23:30 on whatever day the run is still going, instead of
    silently expiring after the first night.
    """
    raw = text.strip().replace(".", ":").replace(" ", "")
    if not raw:
        return ""
    if raw.casefold() in _START_NOW_WORDS:
        # "now" is a valid start time, meaning no wait. Stored as a word
        # rather than "" so it is distinguishable from never having chosen.
        return raw.casefold()
    if ":" not in raw and raw.isdigit() and len(raw) in (3, 4):
        raw = f"{raw[:-2]}:{raw[-2:]}"          # "2330" -> "23:30"
    try:
        hour_s, minute_s = raw.split(":", 1)
        hour, minute = int(hour_s), int(minute_s)
    except ValueError:
        raise ValueError("Please give the time as HH:MM (e.g. 23:30).") from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("The time must be between 00:00 and 23:59.")
    return f"{hour:02d}:{minute:02d}"


def _stop_time_passed(stop_at: str, started_at: str | None) -> bool:
    """Has the chosen wall-clock time arrived since the run started?

    Compared against the run's start rather than "today", so a run begun at
    22:00 with a 01:00 stop ends at 01:00 the NEXT day - the obvious reading
    of "stop at 1am", and the case an overnight run actually hits.
    """
    try:
        hour, minute = (int(part) for part in stop_at.split(":", 1))
    except (ValueError, AttributeError):
        return False

    now_local = _now().astimezone(DUBAI_TZ)
    if started_at:
        try:
            began = datetime.fromisoformat(started_at).astimezone(DUBAI_TZ)
        except ValueError:
            began = now_local
    else:
        began = now_local

    target = began.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= began:
        # The time already passed on the start day, so it means tomorrow.
        target += timedelta(days=1)
    return now_local >= target


def next_occurrence(clock: str, *, after: datetime | None = None) -> datetime | None:
    """The next moment a "HH:MM" Dubai wall-clock time comes around.

    Returns an aware UTC datetime, or None if the string is unusable. Used
    for start times: "21:00" means 21:00 today if that is still ahead,
    otherwise 21:00 tomorrow.
    """
    try:
        hour, minute = (int(part) for part in str(clock).split(":", 1))
    except (ValueError, AttributeError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    reference = (after or _now()).astimezone(DUBAI_TZ)
    target = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= reference:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc)


def _start_time_reached(start_at: str, armed_at: str | None) -> bool:
    """Has a country's start time arrived yet?

    Measured from when the country was armed (upload/start), not from "today
    at 00:00": a file queued at 22:00 with a 21:00 start is asking for 21:00
    TOMORROW, not for a start time that technically already passed an hour
    ago. Without the anchor such a run would begin instantly, which is the
    opposite of what a start time is for.
    """
    clock = str(start_at or "").strip()
    if not clock or clock.casefold() in _START_NOW_WORDS:
        return True

    anchor: datetime | None = None
    if armed_at:
        try:
            anchor = datetime.fromisoformat(armed_at)
        except ValueError:
            anchor = None
    begins = next_occurrence(clock, after=anchor)
    if begins is None:
        return True
    return _now() >= begins


def starts_at(start_at: str, armed_at: str | None) -> datetime | None:
    """When a country with this start time will actually begin, or None."""
    clock = str(start_at or "").strip()
    if not clock:
        return None
    anchor: datetime | None = None
    if armed_at:
        try:
            anchor = datetime.fromisoformat(armed_at)
        except ValueError:
            anchor = None
    return next_occurrence(clock, after=anchor)


# Starting points, not a closed list - the owner edits these or adds their
# own. Chosen to span the range actually seen in practice: a country that
# drains fast, a large batch, and a "replace the stock wholesale" profile
# that needs /frcd. None sets an interval: every country shares one check.
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "Fast burn": {
        "quota_threshold": 0,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": False,
        "_note": "High-demand country: top up the moment it empties.",
    },
    "Steady": {
        "quota_threshold": 0,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": False,
        "_note": "Default-ish settings for a country with normal turnover.",
    },
    "Slow / large stock": {
        "quota_threshold": 0,
        "limit": 10,
        "count": 10,
        "force_delete_before_add": False,
        "_note": "Big batch that lasts - adds 10 at a time.",
    },
    "Low-stock refill": {
        "quota_threshold": 200,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": True,
        "_note": (
            "Refills at 200 left instead of waiting for zero, so the country "
            "never actually runs dry. Wipes with /frcd first - needs your uid."
        ),
    },
    "One-shot burst": {
        "quota_threshold": 200,
        "limit": 4,
        "count": 4,
        "max_refills": 5,
        "delete_when_done": True,
        "force_delete_before_add": False,
        "_note": (
            "Short campaign: refills at 200 left, stops after 5 re-adds, then "
            "clears the country off the bot. Needs your uid for the delete."
        ),
    },
    "Replace stock": {
        "quota_threshold": 0,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": True,
        "_note": "Wipes the country with /frcd before adding. Needs your uid set.",
    },
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(country: str) -> str:
    """Countries are matched case/space-insensitively.

    The name can arrive from a filename guess, a button, or something the
    owner typed, and "bangladesh" must not become a second country beside
    "Bangladesh" with its own divergent settings.
    """
    return " ".join(country.strip().casefold().split())


# --------------------------------------------------------------------------- #
# Per-country settings
# --------------------------------------------------------------------------- #
async def get_all_country_settings() -> dict[str, dict[str, Any]]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, COUNTRY_KEY)
    return dict(stored or {})


async def get_country_settings(country: str) -> dict[str, Any]:
    return (await get_all_country_settings()).get(_key(country), {})


async def set_country_settings(country: str, values: dict[str, Any]) -> dict[str, Any]:
    """Merge overrides for one country. Passing None for a field clears it,
    so a country can be handed back to the global default without having to
    delete and rebuild the whole entry.
    """
    all_settings = await get_all_country_settings()
    key = _key(country)
    current = dict(all_settings.get(key, {}))
    current["display_name"] = country.strip()
    # Left over from per-country intervals; the interval is global now.
    current.pop("interval_minutes", None)

    for field in OVERRIDABLE:
        if field not in values:
            continue
        value = values[field]
        if value is None:
            current.pop(field, None)
        else:
            current[field] = value

    all_settings[key] = current
    async with session_scope() as session:
        await repo.set_setting(session, COUNTRY_KEY, all_settings)

    if "start_at" in values:
        await _regate_if_waiting(country, values["start_at"])
    return current


async def _regate_if_waiting(country: str, start_at: str | None) -> bool:
    """Point a run that is still WAITING at its new start time.

    The gate a run waits on is recorded when it is armed (see begin_run), so
    that changing a default later cannot pause work already going. The flip
    side was that pressing "On at -> Ekhoni" on a country auto-armed for
    04:00 changed only the setting: the run kept waiting on its recorded
    04:00 and was neither checked nor refilled until the next morning - the
    owner saw "it added once at the timer and then never again".

    A run that has already started is left alone: a new start time is about
    the next run, never a reason to pause the current one. ``None`` means
    the country's own value was cleared, so it now follows the global
    default.
    """
    state = await _get_run_state()
    key = _key(country)
    entry = state.get(key)
    if not entry:
        return False
    gate = str(entry.get("gate") or "").strip()
    if _is_now(gate) or _start_time_reached(gate, entry.get("armed_at") or entry.get("started_at")):
        return False

    if start_at is None:
        from app.automation import otp_bot

        start_at = str((await otp_bot.get_config()).get("start_at") or "")
    new_gate = str(start_at or "").strip()
    now_iso = _now().isoformat()
    entry["gate"] = "" if _is_now(new_gate) else new_gate
    # Measured from now: "21:00" chosen at 22:00 means tomorrow's 21:00.
    entry["armed_at"] = now_iso
    state[key] = entry
    await _save_run_state(state)

    if _is_now(new_gate):
        # The shared check is due on the very next pass, not one interval
        # from whenever it was last armed.
        await arm_shared_check(at=_now())
    return True


async def regate_waiting_runs(start_at: str) -> list[str]:
    """The global default start time changed: every run still waiting on the
    default follows it. A country with its own start time keeps it.
    """
    overrides = await get_all_country_settings()
    moved: list[str] = []
    for key, entry in (await _get_run_state()).items():
        own = overrides.get(key, {})
        if own.get("start_at") is not None:
            continue
        name = entry.get("display_name") or key
        if await _regate_if_waiting(name, start_at):
            moved.append(name)
    return moved


async def clear_country_settings(country: str) -> bool:
    all_settings = await get_all_country_settings()
    if _key(country) not in all_settings:
        return False
    all_settings.pop(_key(country))
    async with session_scope() as session:
        await repo.set_setting(session, COUNTRY_KEY, all_settings)
    return True


async def effective_config(country: str, base: dict[str, Any]) -> dict[str, Any]:
    """The config to actually use for one country: global, with that
    country's overrides layered on top.
    """
    merged = dict(base)
    overrides = await get_country_settings(country)
    for field in OVERRIDABLE:
        if field in overrides:
            merged[field] = overrides[field]
    return merged


# --------------------------------------------------------------------------- #
# The shared check timer
# --------------------------------------------------------------------------- #
# The one next-check time, stored in the due map under a key no real country
# name produces. Older versions kept one entry per country in this same map;
# migrate_to_shared_timer() removes those.
SHARED_DUE = "__all__"


def interval_of(base: dict[str, Any]) -> int:
    """The one check interval, in minutes, from the GLOBAL config."""
    try:
        return max(1, int(base.get("interval_minutes") or 1))
    except (TypeError, ValueError):
        return 1


async def _global_interval() -> int:
    from app.automation import otp_bot

    return interval_of(await otp_bot.get_config())


async def _get_due_map() -> dict[str, str]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, DUE_KEY)
    return dict(stored or {})


async def _save_due_map(due: dict[str, str]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, DUE_KEY, due)


def _parse_due(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


async def get_next_check_at() -> datetime | None:
    """When the shared check (every country at once) is next due."""
    return _parse_due((await _get_due_map()).get(SHARED_DUE))


async def arm_shared_check(
    minutes: int | None = None, *, at: datetime | None = None
) -> datetime:
    """Set the shared check to ``at``, or ``minutes`` from now (default: the
    global interval). Unconditional - use arm_country() to arm without ever
    pushing an earlier check later."""
    if at is None:
        if minutes is None:
            minutes = await _global_interval()
        at = _now() + timedelta(minutes=max(1, int(minutes)))
    due = await _get_due_map()
    due[SHARED_DUE] = at.isoformat()
    await _save_due_map(due)
    return at


async def arm_country(country: str, minutes: int | None = None) -> datetime:
    """Compatibility shim: make sure the SHARED check is armed.

    There are no per-country timers any more. Arms the shared check for
    ``minutes`` from now (default: the global interval) unless it is already
    due sooner - starting or resuming one country must never postpone the
    check every other country is waiting on. Returns the shared due time.
    ``country`` is accepted for compatibility and not used.
    """
    del country
    if minutes is None:
        minutes = await _global_interval()
    wanted = _now() + timedelta(minutes=max(1, int(minutes)))
    current = await get_next_check_at()
    if current is not None and current <= wanted:
        return current
    return await arm_shared_check(at=wanted)


async def get_due_at(country: str) -> datetime | None:
    """Compatibility shim: the shared next-check time, the same for every
    country. ``country`` is accepted and not used."""
    del country
    return await get_next_check_at()


async def forget_country(country: str) -> None:
    """Drop a country's own settings (and any leftover per-country due time
    from before the shared timer). The shared timer itself is untouched -
    the other countries still run on it."""
    due = await _get_due_map()
    key = _key(country)
    if key != SHARED_DUE and due.pop(key, None) is not None:
        await _save_due_map(due)
    await clear_country_settings(country)


async def migrate_to_shared_timer() -> int:
    """One-off (config v3): from per-country timers to the shared one.

    Removes every country's own interval_minutes override and every
    per-country due time, and makes the shared check due NOW, so countries
    that were sitting on an old 60-minute timer are all checked on the very
    next pass. Returns how many interval overrides were removed.
    """
    settings = await get_all_country_settings()
    removed = 0
    for entry in settings.values():
        if isinstance(entry, dict) and "interval_minutes" in entry:
            entry.pop("interval_minutes")
            removed += 1
    if removed:
        async with session_scope() as session:
            await repo.set_setting(session, COUNTRY_KEY, settings)
    await _save_due_map({SHARED_DUE: _now().isoformat()})
    return removed


async def due_countries(countries: list[str], base: dict[str, Any]) -> list[str]:
    """Which of these countries the shared check covers right now.

    One timer for all: when it is due, EVERY started, unpaused country is
    returned (a single /st answers for all of them) and the timer is
    re-armed for one global interval from now. Otherwise nothing is.

    With no recorded due time the timer is armed rather than fired: a fresh
    install must not trigger an immediate refill for everything at once.
    When it is due but no country is eligible (all paused or still waiting
    for a start time) it is left due, so a start time arriving is picked up
    on the next look instead of one interval later.
    """
    interval = interval_of(base)
    now = _now()
    when = await get_next_check_at()
    if when is None:
        await arm_shared_check(at=now + timedelta(minutes=interval))
        return []
    if now < when:
        return []

    ready: list[str] = []
    for country in dict.fromkeys(countries):
        # A paused country keeps its settings, its file and its place - it
        # is simply skipped until resumed.
        if (await effective_config(country, base)).get("paused"):
            continue
        # Not started yet: the owner asked for this country to begin at a
        # wall-clock time. It sits in the active set, visible and countable,
        # but nothing is sent to the target bot until that time arrives.
        if not await has_started(country, base):
            continue
        ready.append(country)

    if ready:
        await arm_shared_check(at=now + timedelta(minutes=interval))
    return ready


RUN_STATE_KEY = "otp_bot_country_runstate"


async def _get_run_state() -> dict[str, dict[str, Any]]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, RUN_STATE_KEY)
    return dict(stored or {})


async def _save_run_state(state: dict[str, dict[str, Any]]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, RUN_STATE_KEY, state)


async def begin_run(country: str, *, armed_at: str | None = None,
                    start_at: str | None = None) -> None:
    """Mark a country as freshly started: zero refills, clock from now.

    ``armed_at`` anchors a wall-clock start time. It defaults to now, which
    is what an ordinary start means; callers pass it explicitly only to keep
    an existing anchor across a re-arm.

    ``start_at`` records the gate THIS run is waiting on, decided once when
    the run was armed. has_started() reads it from here rather than from the
    live config, because otherwise changing the global default start time
    retroactively pauses every country already running: a country that began
    at 01:00 would suddenly be "waiting" for the new 04:00 default and stop
    refilling until then, with nothing in the log to say why.
    """
    state = await _get_run_state()
    now_iso = _now().isoformat()
    state[_key(country)] = {
        "started_at": now_iso,
        "armed_at": armed_at or now_iso,
        "refills": 0,
        "display_name": country.strip(),
        "gate": str(start_at or "").strip(),
    }
    await _save_run_state(state)


async def has_started(country: str, base: dict[str, Any]) -> bool:
    """False while a country is still waiting for its wall-clock start time.

    A country with no start_at (the common case) is always started, so this
    is a no-op for everything that never asked to be scheduled.
    """
    state = await get_run_state(country)
    if state:
        # The gate this run was armed with. Recorded at arm time so a later
        # config change cannot pause work that is already going.
        gate = str(state.get("gate") or "").strip()
        if _is_now(gate):
            # Either no start time applied, or this run predates the field.
            # An existing run with no recorded gate has already started by
            # definition - it would not have a run state otherwise.
            return True
        if _start_time_reached(gate, state.get("armed_at") or state.get("started_at")):
            return True
        # Still gated - unless the owner has since told this country to
        # start right away. That instruction can only have come AFTER the
        # run was armed: an immediate start_at present at arm time would
        # have produced no gate at all. Runs armed before the setting change
        # re-gated itself (see _regate_if_waiting) were left waiting on a
        # morning start they had already been told to skip.
        own = await get_country_settings(country)
        return "start_at" in own and _is_now(own["start_at"])

    cfg = await effective_config(country, base)
    start_at = str(cfg.get("start_at") or "").strip()
    if _is_now(start_at):
        return True
    return _start_time_reached(start_at, None)


async def pending_start_at(country: str, base: dict[str, Any]) -> datetime | None:
    """When a not-yet-started country will begin, or None if it already has.

    Decided exactly as has_started() decides it - from the gate the run was
    armed with. Reading the live start time instead showed "WAIT 04:00" for
    countries that had started hours earlier, simply because the default
    start time is 04:00.
    """
    if await has_started(country, base):
        return None
    state = await get_run_state(country)
    if state:
        gate = str(state.get("gate") or "").strip()
        anchor = state.get("armed_at") or state.get("started_at")
    else:
        gate = str((await effective_config(country, base)).get("start_at") or "").strip()
        anchor = None
    if _is_now(gate):
        return None
    return starts_at(gate, anchor)


async def record_refill(country: str) -> int:
    """Count one re-add for this country. Returns the new total."""
    state = await _get_run_state()
    entry = state.get(_key(country)) or {
        "started_at": _now().isoformat(),
        "refills": 0,
        "display_name": country.strip(),
    }
    entry["refills"] = int(entry.get("refills", 0)) + 1
    entry["last_refill_at"] = _now().isoformat()
    state[_key(country)] = entry
    await _save_run_state(state)
    return entry["refills"]


async def get_run_state(country: str) -> dict[str, Any]:
    return (await _get_run_state()).get(_key(country), {})


async def clear_run_state(country: str) -> None:
    state = await _get_run_state()
    if state.pop(_key(country), None) is not None:
        await _save_run_state(state)


async def finished_reason(country: str, base: dict[str, Any]) -> str | None:
    """Why this country should stop now, or None to keep going.

    Checked BEFORE a refill, not after: stopping after the limit-th re-add
    would do one more than the owner asked for.
    """
    cfg = await effective_config(country, base)
    state = await get_run_state(country)
    if not state:
        return None

    # A country waiting for its start time has not run for a single minute,
    # so no finish condition can have been met. Without this a "start 21:00,
    # 8h run" upload made at 09:00 would be declared finished at 17:00 -
    # before it had ever sent anything.
    if not await has_started(country, base):
        return None

    max_refills = int(cfg.get("max_refills") or 0)
    if max_refills and int(state.get("refills", 0)) >= max_refills:
        return f"{max_refills:,} re-add{'' if max_refills == 1 else 's'} done"

    stop_at = str(cfg.get("stop_at") or "").strip()
    if stop_at and _stop_time_passed(stop_at, _effective_start(cfg, state)):
        return f"{stop_at} (Dubai) stop time reached"

    run_minutes = int(cfg.get("run_minutes") or 0)
    if run_minutes:
        began_iso = _effective_start(cfg, state)
        try:
            started = datetime.fromisoformat(began_iso) if began_iso else None
        except (TypeError, ValueError):
            return None
        if started is None:
            return None
        if _now() >= started + timedelta(minutes=run_minutes):
            return f"{_human_minutes(run_minutes)} run time reached"
    return None


def _human_minutes(minutes: int) -> str:
    """'30 min', '12h', '1h 30m' - the same shape the scheduler's owner
    notifications use, so a finish reason reads identically everywhere."""
    minutes = max(0, int(minutes))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h {rest}m" if rest else f"{hours}h"


def _effective_start(cfg: dict[str, Any], state: dict[str, Any]) -> str | None:
    """When this run's clock actually began.

    For a country with a wall-clock start time that is the moment the start
    time arrived, NOT when the file was uploaded - "20h run, starts 21:00"
    means twenty hours of running, not twenty hours minus however long it
    waited. Everything else keeps using started_at unchanged.

    The start time comes from the run's own recorded gate, for the same
    reason has_started() reads it there: a country that began immediately
    must not have its clock re-based onto a default start time introduced
    afterwards, which would push its finish hours into the future.
    """
    start_at = str(state.get("gate") or "").strip()
    fallback = state.get("started_at")
    if _is_now(start_at):
        return fallback

    anchor = state.get("armed_at") or fallback
    began = starts_at(start_at, anchor)
    if began is None:
        return fallback
    return began.isoformat()


async def set_paused(country: str, paused: bool) -> dict[str, Any]:
    """Turn one country off/on without losing its settings or its file.

    Distinct from removing it: removal throws the file away, pausing keeps
    everything and simply skips the country until it is resumed.
    """
    applied = await set_country_settings(country, {"paused": bool(paused)})
    if not paused:
        from app.automation import otp_bot

        base = await otp_bot.get_config()
        # Back on the shared timer: it joins the next check every country
        # gets (arming it only if nothing is due sooner - resuming one
        # country never postpones the others).
        await arm_country(country, interval_of(base))
        state = await get_run_state(country)
        # A run that went past its limit while paused was never marked
        # finished (the due pass skips paused countries), yet resuming it
        # as-is would have the next pass finish and re-pause it at once.
        if state.get("finished_at") or await finished_reason(country, base) is not None:
            # Resuming a run that ended on its own is a fresh run: a new
            # clock and refill count, or it would finish again instantly on
            # the limit it already reached.
            await begin_run(country)
            await otp_bot.clear_finished(country)
    return applied


async def mark_finished(country: str, reason: str) -> None:
    """A country's run reached its limit: pause it and remember why.

    Paused rather than forgotten, so its file and settings are held and
    set_paused(country, False) picks it up again as a fresh run.
    """
    await set_country_settings(country, {"paused": True})
    state = await _get_run_state()
    entry = state.get(_key(country)) or {"display_name": country.strip()}
    entry["finished_at"] = _now().isoformat()
    entry["finished_reason"] = reason
    state[_key(country)] = entry
    await _save_run_state(state)


async def is_paused(country: str) -> bool:
    return bool((await get_country_settings(country)).get("paused"))


async def shortest_interval_minutes(base: dict[str, Any]) -> int:
    """The check interval. Kept for compatibility: there is one interval for
    every country now, so this is simply the global value."""
    return interval_of(base)


async def schedule_overview(
    countries: list[str], base: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-country view for the UIs: what settings apply and when it fires.

    ``interval_minutes`` and ``next_check_at``/``due_in_seconds`` are the
    SHARED values, identical on every row - every country is checked
    together."""
    now = _now()
    interval = interval_of(base)
    when = await get_next_check_at()
    rows: list[dict[str, Any]] = []
    for country in sorted(set(countries)):
        cfg = await effective_config(country, base)
        overrides = await get_country_settings(country)
        begins = await pending_start_at(country, base)
        rows.append({
            "country": country,
            "interval_minutes": interval,
            "quota_threshold": cfg.get("quota_threshold"),
            "limit": cfg.get("limit"),
            "count": cfg.get("count"),
            "tag": cfg.get("tag"),
            "force_delete_before_add": bool(cfg.get("force_delete_before_add")),
            "paused": bool(cfg.get("paused")),
            "stop_at": cfg.get("stop_at") or "",
            "start_at": cfg.get("start_at") or "",
            # Null once running, so a UI can show "waiting until X" without
            # having to recompute the wall-clock arithmetic itself.
            "starts_at": begins.isoformat() if begins else None,
            "starts_in_seconds": int((begins - now).total_seconds()) if begins else None,
            "waiting_to_start": begins is not None,
            "max_refills": cfg.get("max_refills") or 0,
            "run_minutes": cfg.get("run_minutes") or 0,
            "customised": sorted(f for f in OVERRIDABLE if f in overrides),
            "next_check_at": when.isoformat() if when else None,
            "due_in_seconds": int((when - now).total_seconds()) if when else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
async def get_presets() -> dict[str, dict[str, Any]]:
    """Built-ins plus the owner's own. A saved preset with a built-in's name
    shadows it, so the defaults can be corrected rather than worked around.
    """
    async with session_scope() as session:
        stored = await repo.get_setting(session, PRESET_KEY)
    presets = dict(BUILTIN_PRESETS)
    for name, entry in dict(stored or {}).items():
        # A preset saved before the interval became global may still carry
        # one; it would be ignored on apply, so do not show it either.
        presets[name] = (
            {k: v for k, v in entry.items() if k != "interval_minutes"}
            if isinstance(entry, dict) else entry
        )
    return presets


async def save_preset(name: str, values: dict[str, Any]) -> dict[str, Any]:
    name = name.strip()[:60]
    if not name:
        raise ValueError("preset needs a name")

    async with session_scope() as session:
        stored = dict(await repo.get_setting(session, PRESET_KEY) or {})
        # Membership, not truthiness: quota_threshold=0 ("wait until empty")
        # and force_delete_before_add=False are both meaningful values that a
        # truthiness filter would silently drop from the preset.
        entry = {f: values[f] for f in OVERRIDABLE if values.get(f) is not None}
        if values.get("_note"):
            entry["_note"] = str(values["_note"])[:200]
        stored[name] = entry
        await repo.set_setting(session, PRESET_KEY, stored)
    return entry


async def delete_preset(name: str) -> bool:
    async with session_scope() as session:
        stored = dict(await repo.get_setting(session, PRESET_KEY) or {})
        if name not in stored:
            # Built-ins are not stored, so they cannot be deleted - only
            # shadowed. Saying so beats a silent no-op.
            return False
        stored.pop(name)
        await repo.set_setting(session, PRESET_KEY, stored)
    return True


async def apply_preset(name: str, country: str) -> dict[str, Any] | None:
    presets = await get_presets()
    preset = presets.get(name)
    if preset is None:
        return None
    # OVERRIDABLE has no interval: an old saved preset's interval_minutes is
    # dropped here rather than giving one country its own timer.
    values = {f: preset[f] for f in OVERRIDABLE if f in preset}
    return await set_country_settings(country, values)
