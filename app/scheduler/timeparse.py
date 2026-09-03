"""Natural-language-ish time parsing for the scheduler.

Deliberately small and deterministic: the LLM passes structured hints
("in 2 hours", "tomorrow 10:00", "0 9 * * *") and this module turns them into a
concrete UTC ``next_run_at``.  No extra dependency beyond croniter.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

try:  # croniter is optional at import time so tests can run without it
    from croniter import croniter
except Exception:  # pragma: no cover
    croniter = None  # type: ignore[assignment]


class TimeParseError(ValueError):
    pass


_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}

_DELAY_RE = re.compile(r"(?:in\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)", re.IGNORECASE)
_AT_RE = re.compile(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_duration(text: str) -> int:
    """'2h', 'in 30 minutes', '90s' -> seconds."""
    match = _DELAY_RE.search(text.strip())
    if not match:
        raise TimeParseError(f"could not parse duration: {text!r}")
    value, unit = float(match.group(1)), match.group(2).lower()
    if unit not in _UNITS:
        raise TimeParseError(f"unknown time unit: {unit!r}")
    return int(value * _UNITS[unit])


def parse_when(text: str, *, base: datetime | None = None) -> datetime:
    """Parse a one-off time expression into an aware UTC datetime."""
    base = base or now_utc()
    raw = text.strip().lower()
    if not raw:
        raise TimeParseError("empty time expression")

    # ISO timestamp
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    if raw in {"now", "immediately"}:
        return base

    if raw.startswith("in ") or _DELAY_RE.fullmatch(raw):
        return base + timedelta(seconds=parse_duration(raw))

    day_offset = None
    if raw.startswith("tomorrow"):
        day_offset, raw = 1, raw.replace("tomorrow", "", 1).strip()
    elif raw.startswith("today"):
        day_offset, raw = 0, raw.replace("today", "", 1).strip()

    if day_offset is not None or raw.startswith("at "):
        match = _AT_RE.search(raw)
        if not match:
            raise TimeParseError(f"could not parse time of day: {text!r}")
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = (match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise TimeParseError(f"invalid time of day: {text!r}")
        target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        target += timedelta(days=day_offset if day_offset is not None else 0)
        if target <= base:
            target += timedelta(days=1)
        return target

    raise TimeParseError(
        f"unsupported time expression: {text!r} "
        "(use 'in 2 hours', 'tomorrow 09:30', an ISO timestamp, or a cron expression)"
    )


def next_cron(expr: str, *, base: datetime | None = None) -> datetime:
    if croniter is None:  # pragma: no cover
        raise TimeParseError("cron scheduling requires the 'croniter' package")
    base = base or now_utc()
    if not croniter.is_valid(expr):
        raise TimeParseError(f"invalid cron expression: {expr!r}")
    return croniter(expr, base).get_next(datetime).astimezone(timezone.utc)
