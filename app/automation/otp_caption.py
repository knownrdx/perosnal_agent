"""Read the settings out of an upload's caption.

The owner uploads a numbers file and, more often than not, already says what
they want in the caption: "bangladesh whatsapp 20h", "start 21:00 stop 06:00",
"facebook, kono stop nai". Asking those same questions back one button at a
time is the bot making the owner repeat themselves.

So: anything the caption states is taken, and ONLY what is missing is asked.
Three things can be stated -

    service/tag   "whatsapp", "tag: fb", "service telegram"
    country       any country name this module's table knows
    timing        a start time, a stop time, or a run length

Deliberately conservative. A caption that says nothing parses to nothing and
the normal questions happen exactly as before; a word that merely looks like a
service is not one. Getting this wrong silently files numbers under the wrong
service, which is worse than one extra question.
"""

from __future__ import annotations

import re
from typing import Any

# Services the target bot actually sells. Matched case-insensitively with a
# few spellings/abbreviations the owner uses in practice.
SERVICE_ALIASES: dict[str, str] = {
    "whatsapp": "WhatsApp", "whats app": "WhatsApp", "wa": "WhatsApp",
    "wp": "WhatsApp", "wtsp": "WhatsApp", "hoyatsapp": "WhatsApp",
    "telegram": "Telegram", "tg": "Telegram", "tele": "Telegram",
    "facebook": "Facebook", "fb": "Facebook", "face book": "Facebook",
    "google": "Google", "gmail": "Google", "gm": "Google",
    "instagram": "Instagram", "insta": "Instagram", "ig": "Instagram",
    "signal": "Signal",
    "viber": "Viber", "imo": "IMO", "tiktok": "TikTok", "tik tok": "TikTok",
    "twitter": "Twitter", "x": "Twitter", "discord": "Discord",
    "snapchat": "Snapchat", "snap": "Snapchat", "line": "Line",
    "wechat": "WeChat", "uber": "Uber", "amazon": "Amazon",
    "general": "General", "gen": "General",
}

# Two-letter aliases are real words in Banglish ("wa", "x", "gm"), so they
# only count when explicitly introduced as a service/tag.
_SHORT_ALIASES = {alias for alias in SERVICE_ALIASES if len(alias) <= 2}

_TAG_INTRO_RE = re.compile(
    r"\b(?:service|tag|sarvice|servis)\s*[:=\-]?\s*([A-Za-z][A-Za-z ]{0,18})",
    re.IGNORECASE,
)

# "23:30", "23.30", "2330" (4 digits only - "230" is far more likely a count).
_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3])\s*[:.\u0964]\s*([0-5]\d)\b")
_CLOCK_COMPACT_RE = re.compile(r"\b([01]\d|2[0-3])([0-5]\d)\b")
# "9pm", "9 pm", "11:30pm"
_AMPM_RE = re.compile(r"\b(1[0-2]|0?\d)(?:\s*[:.]\s*([0-5]\d))?\s*(am|pm)\b", re.IGNORECASE)

# "rat 9ta", "shokal 6 ta", "bikal 5টা" - how a time of day is normally said
# in Bengali, with the part of the day carrying the am/pm. Without this the
# hour was read as a bare number and dropped, so "rat 9 ta porjonto" set no
# stop time at all and the owner was asked for one he had already given.
_BN_TOD_RE = re.compile(
    r"\b(rat|raat|sondha|shondha|sokal|shokal|bikal|bikel|dupur)\s*"
    r"(1[0-2]|0?\d)\s*(?:ta|tay|tar)?\b",
    re.IGNORECASE,
)
# Which half of the clock each part of the day means. "rat 9" is 21:00 but
# "rat 3" is 03:00, so night wraps rather than adding a flat twelve hours.
_BN_PM_WORDS = {"rat", "raat", "sondha", "shondha", "bikal", "bikel", "dupur"}

# "20h", "20 hours", "20 ghonta", "2 din", "90 min"
_DURATION_RE = re.compile(
    r"\b(\d{1,4})\s*"
    r"(h|hr|hrs|hour|hours|ghonta|ghanta|ghonda|"
    r"m|min|mins|minute|minutes|minit|"
    r"d|day|days|din)\b",
    re.IGNORECASE,
)

_HOUR_UNITS = {"h", "hr", "hrs", "hour", "hours", "ghonta", "ghanta", "ghonda"}
_MINUTE_UNITS = {"m", "min", "mins", "minute", "minutes", "minit"}
_DAY_UNITS = {"d", "day", "days", "din"}

# Words placing a time as a START rather than a stop.
_START_WORDS = (
    "start", "shuru", "suru", "begin", "chalu", "theke", "thekei", "from",
)
# Words placing a time as a STOP.
_STOP_WORDS = (
    "stop", "off", "bondho", "bondo", "end", "shesh", "ses", "porjonto",
    "until", "till", "porjonto",
)
# Bengali postpositions: they FOLLOW the time they label ("9 ta theke",
# "11 ta porjonto"), unlike English labels which precede it. _nearest()
# reverses its tie-break bias for these.
_POSTPOSITIONS = frozenset({"theke", "thekei", "porjonto", "porjonto"})
# "no stop time at all"
_NEVER_PHRASES = (
    "kono stop nai", "kono stop time nai", "stop nai", "kono time nai",
    "no stop", "never stop", "no limit", "limit nai", "unlimited",
    "sara din", "sara rat", "sararat", "always", "24/7", "24x7",
    "kokhono na", "bondho korbe na", "cholte thakbe",
)

# "start straight away" - the way out of the 04:00 default without opening
# the panel. Sets start_at to "" explicitly, which is different from not
# mentioning it at all (that inherits the default).
# Mirrors otp_bot.START_NOW. Defined here rather than imported because
# otp_bot imports this module, and "" would be read by get_config() as an
# old empty value and replaced with the default start time.
START_NOW = "now"

_NOW_PHRASES = (
    "ekhoni shuru", "ekhuni shuru", "ekhoni start", "ekhuni start",
    "shuru ekhoni", "start ekhoni", "ekhoni chalu", "chalu ekhoni",
    "start now", "now start", "right now", "immediately", "ekhoni",
    "ekhuni", "sathe sathe", "shonge shonge",
)


def _nearest(text: str, words: tuple[str, ...], start: int, end: int) -> float | None:
    """Distance from the match to the closest of ``words``, or None.

    Distance, not position: "start 21:00 stop 06:00" names both, and each
    time belongs to whichever keyword sits nearer to IT. Ranking by position
    in the string instead would hand both times to the same keyword.

    A keyword BEFORE the time counts as very slightly nearer than one the
    same distance after it, because "start 21:00 stop 06:00" is written that
    way round - the label precedes the value it labels.
    """
    lowered = text.casefold()
    best: float | None = None
    for word in words:
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lowered):
            if match.end() <= start:
                distance = float(start - match.end())
                # English labels lead their value ("start 21:00"), so a word
                # in front is marginally the better candidate - EXCEPT for
                # the Bengali postpositions, which only ever FOLLOW the time
                # they label ("6 ta theke", "11 ta porjonto"). Giving those a
                # lead bonus handed "sokal 6 ta theke rat 11 ta porjonto" to
                # the wrong keyword: "theke" sat one character before the
                # second time and stole it from the "porjonto" right after.
                if word not in _POSTPOSITIONS:
                    distance -= 0.5
            elif match.start() >= end:
                distance = float(match.start() - end)
                if word in _POSTPOSITIONS:
                    distance -= 0.5
            else:
                distance = 0.0
            if distance > 24:  # too far away to be talking about this time
                continue
            if best is None or distance < best:
                best = distance
    return best


def _classify(text: str, start: int, end: int) -> str:
    """Is the time at [start:end] a start time, a stop time, or unqualified?"""
    start_at = _nearest(text, _START_WORDS, start, end)
    stop_at = _nearest(text, _STOP_WORDS, start, end)
    if start_at is not None and stop_at is not None:
        return "start" if start_at < stop_at else "stop"
    if start_at is not None:
        return "start"
    if stop_at is not None:
        return "stop"
    return ""


def _clock_matches(text: str) -> list[tuple[str, int, int]]:
    """Every wall-clock time in the caption as ("HH:MM", start, end)."""
    found: list[tuple[str, int, int]] = []
    taken: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in taken)

    for match in _AMPM_RE.finditer(text):
        hour = int(match.group(1)) % 12
        minute = int(match.group(2) or 0)
        if match.group(3).lower() == "pm":
            hour += 12
        found.append((f"{hour:02d}:{minute:02d}", match.start(), match.end()))
        taken.append(match.span())

    # "rat 9 ta" / "shokal 6 ta": the part of the day supplies the am/pm.
    # Runs after _AMPM_RE so an explicit "9pm" always wins, and the span
    # covers the day-word too, so the hour cannot also be read as a bare
    # number by the patterns below.
    for match in _BN_TOD_RE.finditer(text):
        if overlaps(match.span()):
            continue
        hour = int(match.group(2)) % 12
        if match.group(1).lower() in _BN_PM_WORDS:
            # Night wraps instead of adding twelve: "rat 9" is 21:00, but
            # "rat 2" is 02:00 - nobody means 14:00 by it.
            hour = hour + 12 if hour >= 4 else hour
        found.append((f"{hour:02d}:00", match.start(), match.end()))
        taken.append(match.span())

    for match in _CLOCK_RE.finditer(text):
        if overlaps(match.span()):
            continue
        found.append(
            (f"{int(match.group(1)):02d}:{match.group(2)}", match.start(), match.end())
        )
        taken.append(match.span())

    for match in _CLOCK_COMPACT_RE.finditer(text):
        if overlaps(match.span()):
            continue
        found.append(
            (f"{match.group(1)}:{match.group(2)}", match.start(), match.end())
        )
        taken.append(match.span())

    return found


def _duration_minutes(value: int, unit: str) -> int:
    unit = unit.casefold()
    if unit in _HOUR_UNITS:
        return value * 60
    if unit in _DAY_UNITS:
        return value * 1440
    if unit in _MINUTE_UNITS:
        return value
    return 0


def _find_service(text: str) -> str | None:
    """Service named in the caption, or None.

    An explicit "service: X" wins and may name anything (the bot's service
    list is the owner's to extend). Otherwise only a known alias counts, and
    the very short ones need that explicit introduction - "wa" and "x" are
    ordinary words.
    """
    intro = _TAG_INTRO_RE.search(text)
    if intro:
        raw = intro.group(1).strip().rstrip(".,;")
        # Greedy capture: "tag fb bangladesh" must give "fb", not the country
        # that followed it. Longest known alias first, so "whats app" beats
        # "whatsapp" never mattering and "tik tok" stays one token.
        lowered_raw = raw.casefold()
        for alias in sorted(SERVICE_ALIASES, key=len, reverse=True):
            if lowered_raw == alias or lowered_raw.startswith(alias + " "):
                return SERVICE_ALIASES[alias]
        if raw:
            # Unknown service: take the first word only. The bot's service
            # list is the owner's to extend, but a whole trailing sentence is
            # never the tag.
            return raw.split()[0][:60]

    lowered = f" {' '.join(text.casefold().split())} "
    best: tuple[int, str] | None = None
    for alias, canonical in SERVICE_ALIASES.items():
        if alias in _SHORT_ALIASES:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
            if best is None or len(alias) > best[0]:
                best = (len(alias), canonical)
    return best[1] if best else None


def parse_caption(text: str) -> dict[str, Any]:
    """Everything the caption states, as a dict of only the keys it stated.

    Possible keys:
        tag         str  - service to add the numbers under
        country     str  - overrides what the numbers themselves say
        start_at    str  - "HH:MM" Dubai, when the run should begin
        stop_at     str  - "HH:MM" Dubai, when it should end
        run_minutes int  - run this long from the start ("20h")
        no_stop     bool - explicitly asked for no stop time at all
    """
    from app.automation import phone_countries

    out: dict[str, Any] = {}
    if not text or not text.strip():
        return out

    # The patterns below are Latin text and the times/durations are ASCII
    # digits, so a caption typed in Bengali ("বাংলাদেশ হোয়াটসঅ্যাপ ২০ ঘন্টা")
    # matched nothing and every field was asked for again. Transliterating
    # first lets one set of patterns serve both scripts.
    from app.agent import language

    text = language.normalise(text).strip()
    lowered = " ".join(text.casefold().split())

    service = _find_service(text)
    if service:
        out["tag"] = service

    country = phone_countries.find_country_name(text)
    if country:
        out["country"] = country

    if any(phrase in lowered for phrase in _NEVER_PHRASES):
        out["no_stop"] = True
        out["stop_at"] = ""
        out["run_minutes"] = 0

    # "ekhoni shuru" - begin immediately, overriding the default start time.
    # Recorded as START_NOW so enqueue_file can tell it apart from a
    # caption that simply never mentioned starting (which inherits the
    # default) and from an empty string left by an older version.
    if any(phrase in lowered for phrase in _NOW_PHRASES):
        out["start_at"] = START_NOW

    for value, start, end in _clock_matches(text):
        kind = _classify(text, start, end)
        if kind == "start":
            # A stated clock time beats "ekhoni" if the caption somehow says
            # both; the specific instruction wins over the general one.
            if not out.get("start_at"):
                out["start_at"] = value
        elif kind == "stop":
            out.pop("no_stop", None)
            out["stop_at"] = value
        else:
            # An unqualified clock time in a caption about a run starting now
            # reads as "run until then".
            out.setdefault("stop_at", value)
            out.pop("no_stop", None)

    for match in _DURATION_RE.finditer(text):
        minutes = _duration_minutes(int(match.group(1)), match.group(2))
        if minutes <= 0:
            continue
        # A duration is a run LENGTH unless it is explicitly a delay
        # ("2 ghonta por start"). Classifying it by nearby start/stop words
        # instead swallowed "21:00 e start, 8h cholbe" entirely.
        after = text[match.end() : match.end() + 12].casefold()
        if re.match(r"\s*(por|pore|later|after)\b", after):
            continue
        out.pop("no_stop", None)
        out["run_minutes"] = minutes
        break

    return out


def describe(parsed: dict[str, Any]) -> str:
    """One line naming what was read out of the caption, for the reply."""
    parts: list[str] = []
    if parsed.get("country"):
        parts.append(f"country {parsed['country']}")
    if parsed.get("tag"):
        parts.append(f"service {parsed['tag']}")
    if parsed.get("start_at"):
        parts.append(f"start {parsed['start_at']}")
    if parsed.get("stop_at"):
        parts.append(f"stop {parsed['stop_at']}")
    if parsed.get("run_minutes"):
        minutes = int(parsed["run_minutes"])
        if minutes % 60 == 0:
            parts.append(f"runs {minutes // 60}h")
        else:
            parts.append(f"runs {minutes} min")
    if parsed.get("no_stop"):
        parts.append("no stop time")
    if not parts:
        return ""
    return "\U0001F4DD From your caption: " + ", ".join(parts)
