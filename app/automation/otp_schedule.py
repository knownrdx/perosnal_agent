"""Per-country settings, independent timers, and reusable presets.

Why this exists as its own module: the base automation in otp_bot.py treats
the active set as one batch on one timer, which is wrong as soon as the owner
runs several countries at once. Bangladesh might burn through its stock every
10 minutes while a Central African Republic batch lasts hours; refilling both
on the same clock either hammers the bot for nothing or starves the fast one.

So each country gets:
  - its own interval, threshold, limit, count, service and cleanup mode,
  - its own next-due timestamp, persisted so a restart does not reset every
    country's clock to "now" and cause a thundering-herd refill,
  - all of it optional: anything not set falls back to the global config, so
    a country the owner never touches keeps behaving exactly as before.

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
# bot being driven, not of one country.
OVERRIDABLE = (
    "interval_minutes",
    "quota_threshold",
    "limit",
    "count",
    "tag",
    "force_delete_before_add",
)

# Starting points, not a closed list - the owner edits these or adds their
# own. Chosen to span the range actually seen in practice: a country that
# drains in minutes, a slow one worth checking rarely, and a "replace the
# stock wholesale" profile that needs /frcd.
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "Fast burn": {
        "interval_minutes": 5,
        "quota_threshold": 0,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": False,
        "_note": "High-demand country: check often, top up the moment it empties.",
    },
    "Steady": {
        "interval_minutes": 15,
        "quota_threshold": 0,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": False,
        "_note": "Default-ish pace for a country with normal turnover.",
    },
    "Slow / large stock": {
        "interval_minutes": 60,
        "quota_threshold": 0,
        "limit": 10,
        "count": 10,
        "force_delete_before_add": False,
        "_note": "Big batch that lasts - checking every few minutes is wasted work.",
    },
    "Low-stock refill": {
        "interval_minutes": 10,
        "quota_threshold": 200,
        "limit": 4,
        "count": 4,
        "force_delete_before_add": True,
        "_note": (
            "Refills at 200 left instead of waiting for zero, so the country "
            "never actually runs dry. Wipes with /frcd first - needs your uid."
        ),
    },
    "Replace stock": {
        "interval_minutes": 30,
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
    return current


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
# Independent timers
# --------------------------------------------------------------------------- #
async def _get_due_map() -> dict[str, str]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, DUE_KEY)
    return dict(stored or {})


async def _save_due_map(due: dict[str, str]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, DUE_KEY, due)


async def arm_country(country: str, minutes: int) -> datetime:
    """Set (or reset) when this country is next due."""
    when = _now() + timedelta(minutes=max(1, int(minutes)))
    due = await _get_due_map()
    due[_key(country)] = when.isoformat()
    await _save_due_map(due)
    return when


async def get_due_at(country: str) -> datetime | None:
    raw = (await _get_due_map()).get(_key(country))
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


async def forget_country(country: str) -> None:
    due = await _get_due_map()
    if due.pop(_key(country), None) is not None:
        await _save_due_map(due)
    await clear_country_settings(country)


async def due_countries(countries: list[str], base: dict[str, Any]) -> list[str]:
    """Which of these countries are due for a check right now.

    A country with no recorded due time is armed rather than fired: a fresh
    start (or a restart) must not trigger an immediate refill for everything
    at once.
    """
    due_map = await _get_due_map()
    now = _now()
    ready: list[str] = []
    dirty = False

    for country in countries:
        key = _key(country)
        cfg = await effective_config(country, base)
        interval = max(1, int(cfg.get("interval_minutes", 10)))

        raw = due_map.get(key)
        if not raw:
            due_map[key] = (now + timedelta(minutes=interval)).isoformat()
            dirty = True
            continue
        try:
            when = datetime.fromisoformat(raw)
        except ValueError:
            due_map[key] = (now + timedelta(minutes=interval)).isoformat()
            dirty = True
            continue

        if now >= when:
            ready.append(country)
            due_map[key] = (now + timedelta(minutes=interval)).isoformat()
            dirty = True

    if dirty:
        await _save_due_map(due_map)
    return ready


async def shortest_interval_minutes(base: dict[str, Any]) -> int:
    """The tightest interval any country asks for.

    The scheduler paces itself by this: polling on the GLOBAL interval would
    silently cap a 5-minute country at whatever the global value happens to
    be, which is exactly the bug per-country timers exist to avoid.
    """
    intervals = [int(base.get("interval_minutes", 10) or 10)]
    for settings in (await get_all_country_settings()).values():
        if "interval_minutes" in settings:
            with_override = int(settings["interval_minutes"] or 0)
            if with_override > 0:
                intervals.append(with_override)
    return max(1, min(intervals))


async def schedule_overview(
    countries: list[str], base: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-country view for the UIs: what settings apply and when it fires."""
    now = _now()
    rows: list[dict[str, Any]] = []
    for country in sorted(set(countries)):
        cfg = await effective_config(country, base)
        overrides = await get_country_settings(country)
        when = await get_due_at(country)
        rows.append({
            "country": country,
            "interval_minutes": cfg.get("interval_minutes"),
            "quota_threshold": cfg.get("quota_threshold"),
            "limit": cfg.get("limit"),
            "count": cfg.get("count"),
            "tag": cfg.get("tag"),
            "force_delete_before_add": bool(cfg.get("force_delete_before_add")),
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
    presets.update(dict(stored or {}))
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
    values = {f: preset[f] for f in OVERRIDABLE if f in preset}
    applied = await set_country_settings(country, values)
    # Re-arm immediately: a preset that changes the interval should take
    # effect from now, not whenever the old interval happened to expire.
    if "interval_minutes" in values:
        await arm_country(country, int(values["interval_minutes"]))
    return applied
