"""Deterministic, LLM-free automation for OTP-number distribution bots.

Why this exists (do not route this through the normal LLM task engine): the
target bot (e.g. @PBDxbot) has a strict, undocumented-by-us FSM: a file must
be sent, then the actual command must be sent as a REPLY to that file
message, or the bot silently rejects it ("No valid phone numbers found").
Getting an LLM to reliably reproduce that exact two-step reply sequence,
every single cycle, forever, is a bad bet - and if the LLM provider is slow
or down (it has been, repeatedly), a normal Task-based job would simply never
run. Everything here calls the owner's userbot directly; no LLM in the loop.

Workflow (matches how the owner actually uses it):
    1. Upload one or more files (Telegram document / web dashboard paperclip).
       Each lands in the QUEUE, untagged.
    2. Say "start" (a command, a plain trigger phrase, or the web dashboard's
       Start button). Every file in the queue needs its own tag - different
       files often mean different countries/campaigns - so start() refuses to
       run until every queued file has one; the caller is expected to collect
       missing tags (one prompt per untagged file) before calling start().
    3. start() sends the cleanup command once, then for every queued file:
       sends the file, replies to that file's message with the add-numbers
       command carrying its own tag. Successful files move from the queue
       into ACTIVE_FILES and the periodic monitor turns on.
    4. The scheduler (see app/workers/scheduler_worker.py) checks quota on
       its own timer while enabled=True; when the bot reports quota <=
       threshold, it re-runs cleanup + re-add for every active file (same
       tags, no new prompts needed - they were already collected at start).
    5. "stop" just turns the periodic monitor off. Queue and active-files
       history are left alone, so restarting later is a fresh decision, not
       a silent resume of possibly-stale state.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.automation import otp_schedule
from app.db import repo
from app.db.base import session_scope
from app.integrations.telegram_user import UserbotError, get_userbot
from app.logging_conf import get_logger
from app.security import rel_path, safe_path

log = get_logger(__name__)

SETTING_KEY = "otp_bot_automation"
QUEUE_KEY = f"{SETTING_KEY}_queue"
ACTIVE_KEY = f"{SETTING_KEY}_active"
LAST_RESULT_KEY = f"{SETTING_KEY}_last_result"
LAST_START_KEY = f"{SETTING_KEY}_last_start"
LAST_TAG_KEY = f"{SETTING_KEY}_last_tag"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,            # periodic quota-monitor/refill loop on/off
    "target_bot": "@PBDxbot",
    "add_command_template": "/fan -t {tag} -l {limit} -c {count}",
    "limit": 4,
    "count": 4,
    # /st ("/stats") over /myquota: it reports live stock PER COUNTRY in one
    # reply, so a single round-trip answers for every country being run
    # instead of needing one check each.
    "quota_command": "/st",
    # Delete the stock command and its reply after reading them. On a short
    # interval this traffic otherwise buries the chat the owner reads.
    "tidy_stock_messages": True,
    "quota_threshold": 0,        # refill when active quota <= this
    "cleanup_command": "/useddelete",
    # Force-delete (Owner-level "/frcd <country> <uid>"). /useddelete only
    # clears used/expired numbers; when a country's stock must be wiped
    # outright before re-adding, this is the command that does it.
    "force_delete_command": "/frcd {country} {uid}",
    "force_delete_before_add": False,
    "force_delete_uid": "",      # owner's own uid; blank disables force-delete
    "interval_minutes": 10,
    "default_tag": "",           # if set, every new file auto-tags with this, never asks
    # Lifecycle limits. 0 = no limit, which is the old behaviour: run until
    # the owner says stop.
    "max_refills": 0,
    "run_minutes": 0,
    # Clearing numbers off the bot at the end is irreversible there, so it
    # is opt-in even when a run does finish on its own.
    "delete_when_done": False,
    "delete_done_command": "/frcd {country} {uid}",
    "thread_id": "",             # dedicated chat thread; "" = not bound yet
    # A file dropped in the thread is a request to run it. Asking "shall I
    # start?" after every upload is the owner repeating themselves.
    "auto_start": True,
    # Before adding, read the bot's own stock: a country that already holds
    # numbers does not need them sent again (the bot answers such an add with
    # pure duplicates). Monitoring still starts, so the refill happens the
    # moment it actually runs out.
    "skip_add_if_stocked": True,
    "awaiting_tag_entry_id": None,  # set while a "what tag for X" prompt is pending
}

# Seeded as the dedicated thread's first message so it gets a recognisable
# title in the thread list (list_threads titles a thread from its first user
# message) instead of showing up as a nameless "New chat".
THREAD_TITLE = "\U0001F501 OTP Bot - send number files here"

# Known tag/region tokens the agent recognises straight out of a filename, so
# it can decide the tag itself instead of asking every single time - e.g.
# "numbers_BD_batch2.txt" -> "BD" with zero owner interaction.
_KNOWN_TAG_TOKENS = {
    "BD", "IN", "PK", "NG", "KE", "ID", "US", "UK", "GENERAL", "GEN",
}

# The live /myquota reply looks like:
#     📊 Your quota
#     Active : 0
#     Limit  : unlimited (disabled)
# Comma-grouped counts ("1,234") appear once the number gets large, so they
# have to be accepted here or a healthy quota reads as an unparseable one.
_QUOTA_RE = re.compile(r"active\s*[:\-]?\s*([\d,]+)", re.IGNORECASE)

# /st ("/stats") reports live stock PER COUNTRY under its own header, which
# is what makes one command enough for any number of countries:
#
#   \U0001F30D Country Stock (yours):
#     \U0001F1E8\U0001F1EB Central African Republic: 90712 (+416 taken)
#
# Only lines below that header count. The section above it has a global
# "\u2022 Available: 416036" line that would otherwise be read as a country.
_STOCK_HEADER_RE = re.compile(r"country\s+stock", re.IGNORECASE)
_STOCK_LINE_RE = re.compile(
    r"^[\s\u2022\-]*"            # bullet / indent
    r"[^\w(]*"                    # flag emoji and other leading symbols
    r"(?P<country>[A-Za-z][A-Za-z .'\-]*[A-Za-z])"
    r"\s*[:\-]\s*"
    r"(?P<count>[\d,]+)"
)


def _parse_country_stock(text: str) -> dict[str, int]:
    """Live per-country stock from a /st reply.

    Returns {} when the reply has no stock section, which the caller treats
    as "not the answer I asked for" rather than "every country is empty" -
    misreading a progress notice as zero stock would trigger a pointless
    refill of everything.
    """
    if not text:
        return {}

    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _STOCK_HEADER_RE.search(line):
            start = index + 1
            break
    if start is None:
        return {}

    stock: dict[str, int] = {}
    for line in lines[start:]:
        if not line.strip():
            # A blank line ends the section; anything after it belongs to
            # some other part of the message.
            if stock:
                break
            continue
        match = _STOCK_LINE_RE.match(line)
        if not match:
            continue
        country = match.group("country").strip()
        try:
            stock[country] = int(match.group("count").replace(",", ""))
        except ValueError:
            continue
    return stock

def _stock_for(stock: dict[str, int], country: str) -> int | None:
    """Look up one country in a /st reply.

    Matched case-insensitively, and by prefix as a fallback, because the
    name we stored came from the phone-prefix table while the one in the
    reply came from the bot - they agree today but need not agree exactly
    ("Congo (DRC)" vs "Congo"), and a missed match reads as zero stock and
    triggers a pointless re-add.
    """
    wanted = " ".join(country.strip().casefold().split())
    lowered = {" ".join(k.strip().casefold().split()): v for k, v in stock.items()}
    if wanted in lowered:
        return lowered[wanted]

    def letters(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    target = letters(wanted)
    if not target:
        return None
    for name, value in lowered.items():
        stored = letters(name)
        if stored and (stored.startswith(target) or target.startswith(stored)):
            return value
    return None


_FAILURE_PHRASES = ("no valid", "error", "failed", "invalid", "not found", "denied")

# "✅ 53412 added, 46588 duplicates skipped" / "Central African Republic: 0 added (20000 dup)"
_ADDED_RE = re.compile(r"([\d,]+)\s+added", re.IGNORECASE)


def _parse_added_count(text: str) -> int | None:
    """How many NEW numbers an add actually contributed.

    This is the difference between "the command worked" and "the file still
    has something to give": a spent file reports "0 added (20000 dup)" -
    a perfectly successful command that changed nothing. Without this the
    automation would keep re-uploading a dead file every cycle forever and
    the owner would never learn the stock is gone.
    """
    if not text:
        return None
    match = _ADDED_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


KNOWN_COUNTRIES_KEY = "otp_bot_known_countries"


# Everything the owner can change from a chat, with the validation each
# needs. Kept as data rather than a wall of if/elif so Telegram, the web UI
# and the help text cannot drift apart - they all read this one table.
SETTABLE_FIELDS: dict[str, dict[str, Any]] = {
    "target_bot": {
        "type": "str",
        "label": "Which bot to drive",
        "example": "@PBDxbot",
    },
    "quota_command": {
        "type": "str",
        "label": "Stock-check command",
        "example": "/st",
    },
    "cleanup_command": {
        "type": "str",
        "label": "Cleanup command",
        "example": "/useddelete",
    },
    "add_command_template": {
        "type": "str",
        "label": "Add command ({tag}, {limit}, {count}, {country})",
        "example": "/fan -t {tag} -l {limit} -c {count}",
    },
    "force_delete_command": {
        "type": "str",
        "label": "Force-delete command ({country}, {uid})",
        "example": "/frcd {country} {uid}",
    },
    "force_delete_uid": {
        "type": "str",
        "label": "Your user id for /frcd",
        "example": "123456789",
    },
    "force_delete_before_add": {
        "type": "bool",
        "label": "Wipe the country before adding",
        "example": "on / off",
    },
    "interval_minutes": {
        "type": "int",
        "label": "Default check interval (minutes)",
        "min": 1,
        "max": 1440,
        "example": "10",
    },
    "quota_threshold": {
        "type": "int",
        "label": "Refill when stock <= this",
        "min": 0,
        "max": 10_000_000,
        "example": "0",
    },
    "limit": {
        "type": "int",
        "label": "Per-add limit",
        "min": 1,
        "max": 10000,
        "example": "4",
    },
    "count": {
        "type": "int",
        "label": "Cooldown seconds",
        "min": 1,
        "max": 10000,
        "example": "4",
    },
    "max_refills": {
        "type": "int",
        "label": "Stop after N re-adds (0 = no limit)",
        "min": 0,
        "max": 10000,
        "example": "5",
    },
    "run_minutes": {
        "type": "int",
        "label": "Stop after N minutes (0 = no limit)",
        "min": 0,
        "max": 100000,
        "example": "120",
    },
    "delete_when_done": {
        "type": "bool",
        "label": "Delete the numbers off the bot when finished",
        "example": "on / off",
    },
    "delete_done_command": {
        "type": "str",
        "label": "Delete-when-done command ({country}, {uid})",
        "example": "/frcd {country} {uid}",
    },
    "auto_start": {
        "type": "bool",
        "label": "Start by itself when a file is uploaded",
        "example": "on / off",
    },
    "skip_add_if_stocked": {
        "type": "bool",
        "label": "Skip the add when the country already has stock",
        "example": "on / off",
    },
    "stop_at": {
        "type": "time",
        "label": "Stop at this time (Dubai, HH:MM; blank = never)",
        "example": "23:30",
    },
    "paused": {
        "type": "bool",
        "label": "Paused (per country: off without losing its file)",
        "example": "on / off",
    },
    "tidy_stock_messages": {
        "type": "bool",
        "label": "Delete the stock command + reply after each check",
        "example": "on / off",
    },
    "default_tag": {
        "type": "str",
        "label": "Default service (blank = ask/auto)",
        "example": "WhatsApp",
    },
    "enabled": {
        "type": "bool",
        "label": "Automation running",
        "example": "on / off",
    },
}

_TRUE_WORDS = {"on", "true", "yes", "1", "haa", "ha", "chalu"}
_FALSE_WORDS = {"off", "false", "no", "0", "na", "bondho"}


def coerce_setting(key: str, raw: str) -> Any:
    """Parse and validate one setting. Raises ValueError with a message meant
    to be shown straight to the owner.
    """
    spec = SETTABLE_FIELDS.get(key)
    if spec is None:
        raise ValueError(f"'{key}' ta kono setting na. /otpset likhe list dekho.")

    value = raw.strip()
    if spec["type"] == "time":
        from app.automation import otp_schedule

        # Blank clears it - "stop at nothing" has to be expressible.
        if value.lower() in {"", "never", "off", "none", "kokhono na"}:
            return ""
        return otp_schedule.parse_stop_time(value)

    if spec["type"] == "bool":
        lowered = value.casefold()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ValueError(f"{key}: 'on' othoba 'off' likho.")

    if spec["type"] == "int":
        try:
            number = int(value.replace(",", ""))
        except ValueError:
            raise ValueError(f"{key}: ekta number lagbe (jemon {spec['example']}).") from None
        low, high = spec.get("min", 0), spec.get("max", 10**9)
        if not low <= number <= high:
            raise ValueError(f"{key}: {low} theke {high} er moddhe hote hobe.")
        return number

    # Strings: a template that loses its placeholders silently stops working,
    # so check the ones the sender actually substitutes.
    if key == "add_command_template" and "{tag}" not in value:
        raise ValueError("add_command_template e {tag} thakte hobe.")
    if key == "force_delete_command" and "{country}" not in value:
        raise ValueError("force_delete_command e {country} thakte hobe.")
    if key == "delete_done_command" and "{country}" not in value:
        raise ValueError("delete_done_command e {country} thakte hobe.")
    if key == "target_bot" and not value.startswith("@"):
        raise ValueError("target_bot '@' diye shuru hobe (jemon @PBDxbot).")
    return value[:200]


async def set_setting_from_chat(key: str, raw: str) -> str:
    """Apply one setting and describe the result in one line."""
    value = coerce_setting(key, raw)
    await save_config({key: value})
    shown = "on" if value is True else "off" if value is False else (value or "(blank)")
    line = f"\u2705 {key} = {shown}"

    # These two silently END a run, and a small number is very easy to read
    # as "every N" rather than "stop after N". Saying so at the moment it is
    # set is the difference between a deliberate short run and waking up to
    # a task that stopped minutes after you went to bed.
    if key == "run_minutes" and value:
        line += (
            f"\n\u26A0\uFE0F Mane: shuru hoar {value} min por task NIJEI BONDHO hobe."
            "\n   Sara raat chalate chaile: /otpset run_minutes 0"
        )
    if key == "max_refills" and value:
        line += (
            f"\n\u26A0\uFE0F Mane: {value} bar re-add er por task NIJEI BONDHO hobe."
            "\n   Limit chara chalate chaile: /otpset max_refills 0"
        )
    if key == "delete_when_done" and value is True:
        line += (
            "\n\u26A0\uFE0F Task shesh hole oi country-r number bot theke MUCHE jabe."
        )
    if key == "stop_at":
        from app.automation import otp_schedule

        clock = otp_schedule.clock_now()
        if value:
            line += (
                f"\n\u23F0 Dubai time {value} hole task bondho hobe."
                f"\n   Ekhon: {clock['dubai']} Dubai / {clock['utc']} UTC"
            )
        else:
            line += "\n\u267E\uFE0F Kono stop time nai."
    return line


async def describe_settings() -> str:
    """Current values for everything settable, with how to change them."""
    from app.automation import otp_schedule

    cfg = await get_config()
    clock = otp_schedule.clock_now()
    lines = [
        "\u2699\uFE0F OTP-bot settings",
        f"\U0001F551 {clock['dubai_full']}  |  {clock['utc']} UTC",
        "",
    ]
    for key, spec in SETTABLE_FIELDS.items():
        current = cfg.get(key, "")
        if current is True:
            shown = "on"
        elif current is False:
            shown = "off"
        else:
            shown = str(current) if current != "" else "(blank)"
        lines.append(f"\u2022 {key} = {shown}")
        lines.append(f"    {spec['label']}")
    lines += [
        "",
        "Bodlate: /otpset <key> <value>",
        "Jemon: /otpset interval_minutes 15",
        "       /otpset target_bot @PBDxbot",
        "       /otpset force_delete_before_add on",
        "",
        "Ek country-r jonno alada: /otpset <country> <key> <value>",
        "Jemon: /otpset Bangladesh interval_minutes 5",
    ]
    return "\n".join(lines)


async def set_country_setting_from_chat(country: str, key: str, raw: str) -> str:
    """Per-country override of the same fields."""
    from app.automation import otp_schedule

    if key not in otp_schedule.OVERRIDABLE:
        allowed = ", ".join(otp_schedule.OVERRIDABLE)
        raise ValueError(f"'{key}' country-r jonno set kora jay na. Jegula jay: {allowed}")

    value = coerce_setting(key, raw)
    canonical = await canonical_country(country)
    await otp_schedule.set_country_settings(canonical, {key: value})
    if key == "interval_minutes":
        await otp_schedule.arm_country(canonical, int(value))
    shown = "on" if value is True else "off" if value is False else value
    return f"\u2705 {canonical}: {key} = {shown}"


PENDING_INPUT_KEY = f"{SETTING_KEY}_pending_input"


async def get_pending_input() -> dict[str, Any] | None:
    """A question the owner is mid-way through answering, if any.

    Custom values (an interval or restock level not on the button list) need
    a free-text answer, and the next message in the OTP thread is that
    answer - so it has to be remembered across messages rather than parsed
    out of a command.
    """
    async with session_scope() as session:
        stored = await repo.get_setting(session, PENDING_INPUT_KEY)
    return dict(stored) if stored else None


async def set_pending_input(field: str | None, country: str | None = None) -> None:
    async with session_scope() as session:
        await repo.set_setting(
            session,
            PENDING_INPUT_KEY,
            {"field": field, "country": country} if field else {},
        )


async def handle_pending_input(text: str) -> str | None:
    """Consume a free-text answer to a custom-value question.

    Returns None when nothing was pending, so ordinary chat is unaffected.
    """
    pending = await get_pending_input()
    if not pending or not pending.get("field"):
        return None

    field, country = pending["field"], pending.get("country")
    if is_skip_trigger(text):
        await set_pending_input(None)
        return "Thik ache, bad dilam."

    try:
        if country:
            reply = await set_country_setting_from_chat(country, field, text)
        else:
            reply = await set_setting_from_chat(field, text)
    except ValueError as exc:
        # Keep the question open: the owner meant to answer it, they just
        # typed something unusable, and dropping it would lose the context.
        return f"\u274C {exc}\n\nAbar likho, othoba 'bad' bolo."

    await set_pending_input(None)
    return reply


async def get_known_countries() -> dict[str, Any]:
    """Countries the TARGET BOT itself reports, learned from /st replies.

    The phone-prefix table is only a guess used to split a file; the bot's
    own spelling is the authority, because that is what /frcd, /setlimit and
    the stock report all key on. A name we invented that the bot does not
    use would silently never match its stock line.
    """
    async with session_scope() as session:
        stored = await repo.get_setting(session, KNOWN_COUNTRIES_KEY)
    return dict(stored or {})


async def learn_countries(stock: dict[str, int]) -> list[str]:
    """Record the countries seen in a /st reply. Returns newly-seen names."""
    if not stock:
        return []

    known = await get_known_countries()
    names = dict(known.get("names", {}))
    new: list[str] = []
    for name, count in stock.items():
        key = " ".join(name.strip().casefold().split())
        if key not in names:
            new.append(name)
        names[key] = {"name": name, "last_stock": count, "seen_at": _now_iso()}

    known["names"] = names
    known["updated_at"] = _now_iso()
    async with session_scope() as session:
        await repo.set_setting(session, KNOWN_COUNTRIES_KEY, known)
    return new


async def canonical_country(name: str) -> str:
    """Map a guessed country name onto the bot's own spelling when we have
    seen it. Falls back to the guess, so a brand-new country still works -
    it just gets corrected the first time the bot reports it.
    """
    known = (await get_known_countries()).get("names", {})
    key = " ".join(name.strip().casefold().split())
    if key in known:
        return known[key]["name"]

    # Prefix match on letters only, so punctuation and abbreviation styles
    # do not block a match ("Central African Rep." vs "...Republic",
    # "Congo (DRC)" vs "Congo").
    def letters(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    target = letters(key)
    if not target:
        return name.strip()
    for stored_key, entry in known.items():
        stored = letters(stored_key)
        if stored and (stored.startswith(target) or target.startswith(stored)):
            return entry["name"]
    return name.strip()


def _looks_like_failure(reply_text: str) -> bool:
    lowered = reply_text.lower()
    return any(phrase in lowered for phrase in _FAILURE_PHRASES)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _sleep(seconds: float) -> None:
    """Thin wrapper around asyncio.sleep so tests can monkeypatch it to zero -
    the real delays exist only to give the target bot time to reply before we
    read its messages back.

    NOTE: this must call asyncio.sleep, never _sleep - an earlier bulk rename
    of asyncio.sleep -> _sleep rewrote this body too and made the function
    infinitely recursive. Tests monkeypatch _sleep, so only production hit it,
    surfacing as a bare "maximum recursion depth exceeded" with no traceback.
    """
    await asyncio.sleep(seconds)


@dataclass(slots=True)
class CycleResult:
    ok: bool
    action: str                     # "skipped" | "added" | "error"
    quota_reply: str = ""
    active_quota: int | None = None
    add_reply: str = ""
    # Live per-country stock from /st, so one command answers for every
    # country at once instead of one round-trip each.
    country_stock: dict[str, int] = field(default_factory=dict)
    # Countries whose file has nothing left to give (0 added, all duplicates)
    # - the owner has to send a fresh file for these.
    exhausted: list[dict[str, Any]] = field(default_factory=list)
    # Countries the bot reported that we had not seen before.
    new_countries: list[str] = field(default_factory=list)
    # Countries that hit their own finish line (max re-adds / time limit).
    finished: list[dict[str, Any]] = field(default_factory=list)
    files_processed: list[str] = field(default_factory=list)
    error: str = ""
    ran_at: str = field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# Config (settings that apply to every run: target bot, timing, templates)
# --------------------------------------------------------------------------- #
async def get_config() -> dict[str, Any]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, SETTING_KEY)
    merged = dict(DEFAULT_CONFIG)
    if stored:
        merged.update(stored)
    return merged


async def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    current = await get_config()
    current.update({k: v for k, v in patch.items() if k in DEFAULT_CONFIG})
    async with session_scope() as session:
        await repo.set_setting(session, SETTING_KEY, current)
    return current


async def get_last_result() -> dict[str, Any] | None:
    async with session_scope() as session:
        return await repo.get_setting(session, LAST_RESULT_KEY)


async def _save_last_result(result: CycleResult) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, LAST_RESULT_KEY, asdict(result))


async def get_last_start_result() -> dict[str, Any] | None:
    async with session_scope() as session:
        return await repo.get_setting(session, LAST_START_KEY)


async def _get_last_tag(country: str | None = None) -> str | None:
    """The tag last used - for THIS country when one is given.

    Service/tag is a per-country decision in practice (Bangladesh stock is
    not sold as the same service as Nigeria stock), so asking for a specific
    country must NOT fall back to the global last tag: doing so silently
    files a new country's numbers under whatever service happened to be used
    last, which is exactly the mix-up this split exists to prevent.
    """
    async with session_scope() as session:
        stored = await repo.get_setting(session, LAST_TAG_KEY)
    if not stored:
        return None
    if country:
        return (stored.get("by_country") or {}).get(country)
    return stored.get("tag")


async def _set_last_tag(tag: str, country: str | None = None) -> None:
    async with session_scope() as session:
        stored = await repo.get_setting(session, LAST_TAG_KEY) or {}
        by_country = dict(stored.get("by_country") or {})
        if country:
            by_country[country] = tag
        await repo.set_setting(
            session, LAST_TAG_KEY, {"tag": tag, "by_country": by_country}
        )


def _infer_tag_from_filename(name: str) -> str | None:
    """Read the tag straight off the filename when it's obvious, e.g.
    "numbers_BD_batch2.txt" -> "BD". Lets the agent decide on its own
    instead of asking every time a file with a self-describing name arrives.
    """
    stem = name.rsplit(".", 1)[0]
    for token in re.split(r"[ _\-]+", stem):
        if token.upper() in _KNOWN_TAG_TOKENS:
            return token.upper()
    return None


async def _decide_tag(name: str, country: str | None = None) -> str | None:
    """Best-effort autonomous tag decision for a newly queued file, in order:
    1. the tag last used for THIS country (strongest signal - same country,
       same service, new batch is the normal case)
    2. filename says it explicitly (the owner named it that way)
    3. an owner-configured default_tag applies to everything
    Returns None only when none of the above give anything - that's the one
    case worth actually asking about.

    Deliberately does NOT fall back to "the last tag used for anything" when
    the country is known: service is a per-country decision, so inheriting
    Nigeria's answer for a Bangladesh file would silently file the numbers
    under the wrong service - worse than asking one short question.
    """
    if country:
        remembered = await _get_last_tag(country)
        if remembered:
            return remembered

    inferred = _infer_tag_from_filename(name)
    if inferred:
        return inferred
    config = await get_config()
    if config.get("default_tag"):
        return config["default_tag"]
    if country:
        return None
    return await _get_last_tag()


# --------------------------------------------------------------------------- #
# Dedicated chat thread: one conversation that IS the OTP-bot workspace.
#
# Without this the owner has to remember which of several chat threads a
# number file belongs in, and a file dropped into a general conversation
# looks identical to one meant for this automation. Binding a specific
# thread makes "where do I send the files?" answer itself - anything sent
# in that thread is for the OTP bot, anything elsewhere is not.
# --------------------------------------------------------------------------- #
async def get_thread_id() -> str:
    return (await get_config()).get("thread_id") or ""


async def is_otp_thread(chat_id: int) -> bool:
    """True when the given chat's CURRENT thread is the dedicated OTP one."""
    bound = await get_thread_id()
    if not bound:
        return False
    async with session_scope() as session:
        row = await repo.ensure_session(session, chat_id)
        return row.current_thread_id == bound


async def ensure_thread(chat_id: int) -> str:
    """Create (once) and switch to the dedicated OTP-bot thread, returning it.

    Reuses the existing thread if one is already bound and still present, so
    calling this repeatedly never spawns duplicate near-identical threads.
    """
    bound = await get_thread_id()
    async with session_scope() as session:
        # A chat that has never been seen has no chat_sessions row yet, and
        # reset_session is an UPDATE - without this it would silently affect
        # zero rows and leave the chat on the default thread.
        await repo.ensure_session(session, chat_id)

        if bound:
            existing = {
                t["thread_id"]: t
                for t in await repo.list_threads(session, chat_id, limit=200)
            }
            if bound in existing:
                await repo.switch_thread(session, chat_id, bound)
                # Backfill the title for a thread that predates the titling
                # fix (or lost its seed some other way) - otherwise it stays
                # a nameless "New chat" forever, which is the one thing this
                # thread is supposed to prevent.
                if existing[bound]["title"] in ("New chat", ""):
                    await repo.add_message(
                        session, chat_id=chat_id, role="user",
                        content=THREAD_TITLE, thread_id=bound,
                    )
                return bound

        new_thread = await repo.reset_session(session, chat_id)
        # Seeded with role="user" deliberately: list_threads titles a thread
        # from its first USER message, so an assistant-role seed would leave
        # this showing as a nameless "New chat" in the thread list - exactly
        # the "which chat was it again?" problem this thread exists to solve.
        await repo.add_message(
            session, chat_id=chat_id, role="user",
            content=THREAD_TITLE, thread_id=new_thread,
        )
        await repo.add_message(
            session, chat_id=chat_id, role="assistant",
            content=(
                "Send number files here (one at a time is fine), then say "
                "\"start\" when you're done. Anything sent in this thread goes "
                "straight into the OTP-bot queue - files sent in other threads "
                "are left alone."
            ),
            thread_id=new_thread,
        )
    await save_config({"thread_id": new_thread})
    return new_thread


# --------------------------------------------------------------------------- #
# File queue: uploaded, waiting for a tag + the start() call
# --------------------------------------------------------------------------- #
async def _load_files(key: str) -> list[dict[str, Any]]:
    async with session_scope() as session:
        stored = await repo.get_setting(session, key)
    return list(stored.get("files", [])) if stored else []


async def _save_files(key: str, files: list[dict[str, Any]]) -> None:
    async with session_scope() as session:
        await repo.set_setting(session, key, {"files": files})


async def get_queue() -> list[dict[str, Any]]:
    return await _load_files(QUEUE_KEY)


async def get_active_files() -> list[dict[str, Any]]:
    return await _load_files(ACTIVE_KEY)


async def enqueue_file(path: str, name: str) -> dict[str, Any]:
    """Analyse an uploaded file and queue one entry PER COUNTRY found in it.

    A single upload routinely mixes countries (a "numbers" export can hold
    Dominican Republic and Bangladesh side by side) and the target bot keeps
    stock per country, so adding the file whole would file every number under
    one country's tag. Each country therefore becomes its own queue entry
    with its own split-out file, and each carries the service/tag last used
    for that country so the common case needs no questions at all.
    """
    from app.automation import phone_countries

    source = safe_path(path, must_exist=True)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    grouped = phone_countries.split_by_country(lines)

    queue = await get_queue()
    created: list[dict[str, Any]] = []
    batch_id = secrets.token_hex(4)

    for country, numbers in grouped.items():
        # Prefer the spelling the target bot itself uses, when we have seen
        # it: every per-country command (/frcd, /setlimit) and the stock
        # report key on the bot's name, not on our prefix table's guess.
        country = await canonical_country(country)
        split_path = path
        if len(grouped) > 1:
            # Write the country's own file so the bot only ever receives
            # numbers that match the country/tag the command declares.
            safe_country = re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_") or "unknown"
            stem = name.rsplit(".", 1)[0]
            split_name = f"{stem}__{safe_country}.txt"
            target = safe_path(f"uploads/{split_name}")
            target.write_text("\n".join(numbers) + "\n", encoding="utf-8")
            split_path = rel_path(target)

        entry = {
            "id": secrets.token_hex(4),
            "batch_id": batch_id,
            "path": split_path,
            "name": name if len(grouped) == 1 else f"{name} [{country}]",
            "source_name": name,
            "country": country,
            "count": len(numbers),
            "tag": await _decide_tag(name, country),
            "uploaded_at": _now_iso(),
        }
        queue.append(entry)
        created.append(entry)

    await _save_files(QUEUE_KEY, queue)
    return {
        "batch_id": batch_id,
        "entries": created,
        "countries": {c: len(n) for c, n in grouped.items()},
    }


async def set_queue_tag(entry_id: str, tag: str) -> dict[str, Any] | None:
    queue = await get_queue()
    for entry in queue:
        if entry["id"] == entry_id:
            entry["tag"] = tag.strip()[:60]
            await _save_files(QUEUE_KEY, queue)
            # Remembered per country, so the next Bangladesh file does not
            # inherit the tag that happened to be used for a Nigeria file.
            await _set_last_tag(entry["tag"], entry.get("country"))
            return entry
    return None


async def remove_from_queue(entry_id: str) -> bool:
    queue = await get_queue()
    remaining = [e for e in queue if e["id"] != entry_id]
    if len(remaining) == len(queue):
        return False
    await _save_files(QUEUE_KEY, remaining)
    # A prompt pointing at a now-deleted entry would hang the conversation:
    # get_awaiting_tag_entry() returns None (the id no longer resolves) while
    # the config still claims one is pending, so clear it explicitly.
    config = await get_config()
    if config.get("awaiting_tag_entry_id") == entry_id:
        await set_awaiting_tag_entry(None)
    return True


async def remove_active_file(entry_id: str) -> dict[str, Any] | None:
    """Stop monitoring one already-started file.

    Removing it here does not un-add the numbers the bot already has - it
    only means future refill cycles stop re-adding this file. Emptying the
    active set also turns the monitor off, since there is then nothing left
    for it to do.
    """
    active = await get_active_files()
    removed = next((e for e in active if e["id"] == entry_id), None)
    if removed is None:
        return None
    remaining = [e for e in active if e["id"] != entry_id]
    await _save_files(ACTIVE_KEY, remaining)
    if not remaining:
        await save_config({"enabled": False})
    return removed


async def remove_by_country(country: str) -> list[dict[str, Any]]:
    """Drop every queued AND active entry for one country.

    Country is how the owner actually thinks about this ("bad the Bangladesh
    one"), and after splitting, one country can span several entries from
    different uploads - so matching by country removes all of them rather
    than leaving stragglers behind.
    """
    wanted = country.strip().casefold()
    removed: list[dict[str, Any]] = []

    queue = await get_queue()
    keep_queue = [e for e in queue if (e.get("country") or "").casefold() != wanted]
    removed += [e for e in queue if (e.get("country") or "").casefold() == wanted]
    if len(keep_queue) != len(queue):
        await _save_files(QUEUE_KEY, keep_queue)

    active = await get_active_files()
    keep_active = [e for e in active if (e.get("country") or "").casefold() != wanted]
    removed += [e for e in active if (e.get("country") or "").casefold() == wanted]
    if len(keep_active) != len(active):
        await _save_files(ACTIVE_KEY, keep_active)
        if not keep_active:
            await save_config({"enabled": False})

    if removed:
        config = await get_config()
        if config.get("awaiting_tag_entry_id") in {e["id"] for e in removed}:
            await set_awaiting_tag_entry(None)
    return removed


async def clear_queue() -> int:
    """Throw away everything waiting to be started. Active files are left
    alone - "clear the queue" should not silently stop running work.
    """
    queue = await get_queue()
    if queue:
        await _save_files(QUEUE_KEY, [])
        await set_awaiting_tag_entry(None)
    return len(queue)


async def next_untagged_entry() -> dict[str, Any] | None:
    for entry in await get_queue():
        if not entry.get("tag"):
            return entry
    return None


async def get_awaiting_tag_entry() -> dict[str, Any] | None:
    """The queue entry currently being asked about, if a tag prompt is open."""
    config = await get_config()
    awaiting_id = config.get("awaiting_tag_entry_id")
    if not awaiting_id:
        return None
    for entry in await get_queue():
        if entry["id"] == awaiting_id:
            return entry
    return None


async def set_awaiting_tag_entry(entry_id: str | None) -> None:
    await save_config({"awaiting_tag_entry_id": entry_id})


class AddRejected(Exception):
    """The target bot's own reply says the add did not work.

    A distinct type rather than RuntimeError: RecursionError, ValueError and
    friends all subclass Exception too, and catching a broad built-in here
    once caused a real crash (infinite recursion in _sleep) to be reported to
    the owner as "the bot rejected your file", sending the investigation in
    completely the wrong direction.
    """


# --------------------------------------------------------------------------- #
# Helpers shared by start() and the periodic refill cycle
# --------------------------------------------------------------------------- #
_QUOTA_RE_MATCH = _QUOTA_RE


def _parse_quota(text: str) -> int | None:
    match = _QUOTA_RE_MATCH.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


async def _last_bot_message(target: str, *, limit: int = 5) -> dict[str, Any] | None:
    """The most recent message FROM the bot (not from us).

    read_messages returns newest-first, so the newest bot message is the
    first non-outgoing entry. Iterating in reverse here once returned the
    OLDEST message in the window instead, which is why a quota check could
    sit at "no reply" while the answer was already sitting in the chat.
    """
    messages = await get_userbot().read_messages(target, limit)
    for message in messages:
        if not message.get("out"):
            return message
    return None


async def _wait_for_new_bot_reply(
    target: str,
    *,
    after_id: int | None,
    timeout_s: float = 90.0,
    poll_s: float = 3.0,
    matches: Any | None = None,
) -> dict[str, Any] | None:
    """Wait for a bot message NEWER than ``after_id`` (optionally one whose
    text satisfies ``matches``).

    A fixed sleep-then-read is not enough: a large file can take the target
    bot well over a minute to ingest, and reading too early returns its
    PREVIOUS message - which is how an add once got reported with the
    /useddelete reply ("Deleted 4,796 used numbers") as its result.

    Recency alone is not enough either: the bot emits progress/completion
    messages of its own, so a quota check can land on the tail end of an
    earlier add ("Fast Add Complete!") and fail to parse. ``matches`` lets
    the caller say what the answer should look like, and unmatched messages
    are skipped rather than mistaken for the reply.
    """
    waited = 0.0
    newest_seen = after_id
    while waited < timeout_s:
        await _sleep(poll_s)
        waited += poll_s
        message = await _last_bot_message(target)
        if message is None:
            continue
        message_id = int(message.get("id", 0))
        if newest_seen is not None and message_id <= newest_seen:
            continue
        if matches is not None and not matches(message.get("text") or ""):
            # Something new, but not the answer we asked for - remember it so
            # we keep moving forward instead of re-examining it every poll.
            newest_seen = message_id
            continue
        return message
    return None


async def _force_delete_country(target: str, country: str, cfg: dict[str, Any]) -> str:
    """Owner-level /frcd for one country, when configured.

    /useddelete only removes used/expired numbers. Re-adding a country whose
    stock is still live would just pile duplicates on top (the bot reports
    them as "dup"), so when the owner wants a genuine replace rather than a
    top-up, this wipes that country's numbers first.
    """
    uid = str(cfg.get("force_delete_uid") or "").strip()
    if not uid:
        return ""
    command = cfg["force_delete_command"].format(country=country, uid=uid)

    previous = await _last_bot_message(target)
    previous_id = int(previous.get("id", 0)) if previous else None
    await get_userbot().send_message(target, command)
    reply = await _wait_for_new_bot_reply(target, after_id=previous_id, timeout_s=45.0)
    return (reply.get("text") or "") if reply else ""


async def _tidy_messages(target: str, message_ids: list[int | None]) -> None:
    """Remove the automation's own bookkeeping messages from the chat.

    Only ever called with ids this code sent or read back itself, and never
    allowed to fail a run: keeping the chat clean is cosmetic, whereas
    losing a cycle over it is not.
    """
    ids = [int(m) for m in message_ids if m]
    if not ids:
        return
    try:
        userbot = get_userbot()
        deleter = getattr(userbot, "delete_messages", None)
        if deleter is None:
            return
        await deleter(target, ids)
    except Exception:  # noqa: BLE001 - tidying is cosmetic
        log.warning("otp_bot_tidy_failed", extra={"ids": ids})


async def _finish_country(
    target: str, entry: dict[str, Any], cfg: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Retire one country: stop monitoring it and optionally clear its
    numbers off the target bot.

    Deleting is opt-in (delete_when_done). It is irreversible on the bot's
    side, so it must never be what happens by default just because a run
    ended.
    """
    country = entry.get("country") or entry["name"]
    outcome: dict[str, Any] = {
        "country": country,
        "name": entry["name"],
        "reason": reason,
        "deleted": False,
        "delete_reply": "",
        "error": "",
    }

    if cfg.get("delete_when_done"):
        command = (cfg.get("delete_done_command") or "").strip()
        uid = str(cfg.get("force_delete_uid") or "").strip()
        if not command:
            outcome["error"] = "no delete command configured"
        elif "{uid}" in command and not uid:
            # Sending "/frcd Bangladesh {uid}" literally would be a silent
            # no-op that looks like it worked.
            outcome["error"] = "delete needs your user id (force_delete_uid)"
        else:
            try:
                previous = await _last_bot_message(target)
                previous_id = int(previous.get("id", 0)) if previous else None
                await get_userbot().send_message(
                    target, command.format(country=country, uid=uid)
                )
                reply = await _wait_for_new_bot_reply(
                    target, after_id=previous_id, timeout_s=60.0
                )
                outcome["deleted"] = True
                outcome["delete_reply"] = (reply or {}).get("text", "")[:300]
            except UserbotError as exc:
                outcome["error"] = f"delete failed: {exc}"
            except Exception as exc:  # noqa: BLE001 - finishing must not crash a cycle
                log.exception("otp_bot_finish_delete_error")
                outcome["error"] = f"delete failed ({type(exc).__name__}): {exc}"

    # Stop monitoring it either way - the run is over even if the delete
    # could not be done.
    await remove_active_file(entry["id"])
    await otp_schedule.clear_run_state(country)
    return outcome


async def _add_one_file(target: str, entry: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Send one file as a reply-based add; returns the bot's reply text.
    Raises AddRejected if the bot's own reply indicates the add failed
    (e.g. "No valid phone numbers found") - a delivered message is not the
    same as a successful add, and silently reporting success on a rejected
    file would hide the exact bug this automation exists to avoid.
    """
    target_path = safe_path(entry["path"], must_exist=True)

    if cfg.get("force_delete_before_add") and entry.get("country"):
        await _force_delete_country(target, entry["country"], cfg)
        await _sleep(2)

    # Remember where the conversation stood before we touch it, so we can
    # tell this file's reply apart from whatever the bot said last.
    previous = await _last_bot_message(target)
    previous_id = int(previous.get("id", 0)) if previous else None

    sent_file = await get_userbot().send_file(target, str(target_path))
    await _sleep(2)

    command_text = cfg["add_command_template"].format(
        tag=entry.get("tag") or "General",
        limit=cfg["limit"],
        count=cfg["count"],
        country=entry.get("country") or "",
    )
    await get_userbot().send_message(target, command_text, reply_to=sent_file.get("message_id"))

    reply = await _wait_for_new_bot_reply(target, after_id=previous_id)
    if reply is None:
        raise AddRejected(
            f"{entry['name']}: no reply from {target} within the wait window - "
            "the add may or may not have gone through, check the chat"
        )
    text = reply.get("text") or ""
    if text and _looks_like_failure(text):
        raise AddRejected(f"{entry['name']}: bot rejected the add ({text[:200]})")
    return text


# --------------------------------------------------------------------------- #
# start() / stop(): owner-triggered, not on the scheduler's timer
# --------------------------------------------------------------------------- #
async def refresh_countries_from_bot() -> dict[str, Any]:
    """Ask the bot for its stock and learn the country list from the reply.

    Lets the owner populate the country list without waiting for a refill
    cycle - and confirms the bot is reachable at the same time.
    """
    cfg = await get_config()
    target = cfg["target_bot"]
    try:
        before = await _last_bot_message(target)
        before_id = int(before.get("id", 0)) if before else None
        sent = await get_userbot().send_message(target, cfg["quota_command"])
        reply = await _wait_for_new_bot_reply(
            target,
            after_id=before_id,
            timeout_s=45.0,
            matches=lambda text: bool(_parse_country_stock(text)),
        )
        if cfg.get("tidy_stock_messages", True):
            await _tidy_messages(target, [sent.get("message_id"), (reply or {}).get("id")])
        if reply is None:
            return {"ok": False, "error": "no stock reply from the bot", "countries": {}}

        stock = _parse_country_stock(reply["text"])
        new = await learn_countries(stock)
        return {"ok": True, "countries": stock, "new": new, "error": ""}
    except UserbotError as exc:
        return {"ok": False, "error": f"telegram account not linked: {exc}", "countries": {}}
    except Exception as exc:  # noqa: BLE001 - never crash a UI call
        log.exception("otp_bot_refresh_countries_error")
        return {"ok": False, "error": f"unexpected error ({type(exc).__name__}): {exc}", "countries": {}}


async def _stock_snapshot(target: str, cfg: dict[str, Any]) -> dict[str, int]:
    """Current per-country stock, or {} if the bot did not answer.

    Returning {} rather than raising keeps a failed read from blocking a
    start: worst case the add happens when it might not have been needed,
    which is far better than refusing to run at all.
    """
    try:
        before = await _last_bot_message(target)
        before_id = int(before.get("id", 0)) if before else None
        sent = await get_userbot().send_message(target, cfg["quota_command"])
        reply = await _wait_for_new_bot_reply(
            target,
            after_id=before_id,
            timeout_s=45.0,
            matches=lambda text: bool(_parse_country_stock(text)),
        )
        if cfg.get("tidy_stock_messages", True):
            await _tidy_messages(target, [sent.get("message_id"), (reply or {}).get("id")])
        if reply is None:
            return {}
        stock = _parse_country_stock(reply["text"])
        await learn_countries(stock)
        return stock
    except Exception:  # noqa: BLE001 - a failed read must not block the start
        log.exception("otp_bot_stock_snapshot_error")
        return {}


async def start_automation() -> dict[str, Any]:
    """Consume every queued (and tagged) file: cleanup once, then add each
    file as its own reply-based command. On success the queue becomes the
    new active_files set (replacing whatever was active before) and the
    periodic monitor (enabled) turns on.
    """
    cfg = await get_config()
    queue = await get_queue()

    if not queue:
        return {"ok": False, "error": "no files queued - upload one first", "missing_tags": []}

    missing = [e for e in queue if not e.get("tag")]
    if missing:
        return {
            "ok": False,
            "error": "every queued file needs a tag before starting",
            "missing_tags": [
                {
                    "id": e["id"],
                    "name": e["name"],
                    "country": e.get("country"),
                    "count": e.get("count"),
                }
                for e in missing
            ],
        }

    target = cfg["target_bot"]
    result: dict[str, Any] = {"ok": False, "target_bot": target, "files": [], "error": "", "ran_at": _now_iso()}

    try:
        # Read the bot's own stock once, before adding anything. A country
        # that is already stocked does not need the same numbers sent again -
        # the bot answers that with pure duplicates and the file is burned
        # for nothing. Monitoring still starts, so the file is added the
        # moment the country actually runs out.
        stock: dict[str, int] = {}
        if cfg.get("skip_add_if_stocked", True):
            stock = await _stock_snapshot(target, cfg)

        needs_add: list[dict[str, Any]] = []
        for entry in queue:
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            have = _stock_for(stock, country) if stock else None
            threshold = int(entry_cfg.get("quota_threshold") or 0)
            if have is not None and have > threshold:
                # Already stocked: hold the file, watch the country.
                entry["skipped_add"] = True
                entry["stock_at_start"] = have
                result["files"].append({
                    "name": entry["name"],
                    "country": entry.get("country"),
                    "count": entry.get("count"),
                    "tag": entry["tag"],
                    "interval_minutes": entry_cfg.get("interval_minutes"),
                    "added": False,
                    "stock": have,
                    "reply": f"already has {have:,} - kept for when it runs out",
                })
            else:
                entry.pop("skipped_add", None)
                needs_add.append(entry)

        # Only tidy up if something is actually going to be added; otherwise
        # this is a pointless command in a chat the owner reads.
        if needs_add:
            await get_userbot().send_message(target, cfg["cleanup_command"])
            await _sleep(2)

        for entry in needs_add:
            country = entry.get("country") or entry["name"]
            # Each country runs on its own settings from the very first add,
            # not just from the second cycle onwards.
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            reply = await _add_one_file(target, entry, entry_cfg)
            result["files"].append({
                "name": entry["name"],
                "country": entry.get("country"),
                "count": entry.get("count"),
                "tag": entry["tag"],
                "interval_minutes": entry_cfg.get("interval_minutes"),
                "added": True,
                "stock": _stock_for(stock, country) if stock else None,
                "reply": reply,
            })

        # Arm every country, added or not: the whole point of skipping an add
        # is that monitoring still runs and refills it later.
        for entry in queue:
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            await otp_schedule.arm_country(
                country, int(entry_cfg.get("interval_minutes", 10))
            )
            # Fresh run: the refill count and the time limit both start now,
            # not from whenever this country last ran.
            await otp_schedule.begin_run(country)

        await save_config({"enabled": True, "awaiting_tag_entry_id": None})
        # MERGE into the active set, never replace it. A second upload used to
        # wipe every country already running - the owner adds Nigeria and
        # silently loses the Bangladesh run started an hour ago. A new file
        # for a country already running supersedes that country's entry only.
        existing = await get_active_files()
        fresh_countries = {(e.get("country") or e["name"]) for e in queue}
        kept = [e for e in existing if (e.get("country") or e["name"]) not in fresh_countries]
        await _save_files(ACTIVE_KEY, kept + queue)
        await _save_files(QUEUE_KEY, [])
        result["kept_running"] = [e.get("country") or e["name"] for e in kept]
        # Surfaced so the owner is told up front when this run will end -
        # a silent finish minutes later looks identical to a crash.
        result["limits"] = {
            "max_refills": int(cfg.get("max_refills") or 0),
            "run_minutes": int(cfg.get("run_minutes") or 0),
            "delete_when_done": bool(cfg.get("delete_when_done")),
        }
        result["ok"] = True

    except UserbotError as exc:
        result["error"] = f"telegram account not linked or errored: {exc}"
    except FileNotFoundError as exc:
        result["error"] = str(exc)
    except AddRejected as exc:
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - start() must never crash the caller
        # log.exception keeps the real traceback in the container logs; the
        # owner-facing string alone is not enough to debug a crash like the
        # recursive-_sleep one, which read simply as "maximum recursion depth
        # exceeded" with no indication of where.
        log.exception("otp_bot_start_error")
        result["error"] = f"unexpected error ({type(exc).__name__}): {exc}"

    async with session_scope() as session:
        await repo.set_setting(session, LAST_START_KEY, result)
    return result


async def maybe_auto_start() -> dict[str, Any] | None:
    """Start by itself once every queued file knows its tag.

    A file dropped into the OTP thread is a request to run it - making the
    owner type "start" afterwards is asking a question whose answer is
    always yes. Returns None when it did not act, so callers can fall back
    to their normal "queued, say start" reply.
    """
    cfg = await get_config()
    if not cfg.get("auto_start", True):
        return None

    queue = await get_queue()
    if not queue:
        return None
    # A file still waiting on its tag is not ready; the tag answer itself
    # calls back in here once it lands.
    if any(not e.get("tag") for e in queue):
        return None

    result = await start_automation()
    result["auto"] = True
    return result


async def set_cleanup_mode(mode: str) -> dict[str, Any]:
    """Pick between "tidy up" and "wipe and replace" as one decision.

    Exposed as a single choice because that is how the owner thinks about it;
    the two config flags underneath (which command, and whether /frcd runs at
    all) always have to move together, and letting them drift apart is how
    you end up with force-delete "enabled" but silently inert.
    """
    if mode == "force":
        return await save_config({"force_delete_before_add": True})
    return await save_config({"force_delete_before_add": False})


async def stop_automation() -> dict[str, Any]:
    """Turn the periodic monitor off. Queue and active_files are untouched -
    starting again later is always a fresh, explicit decision (the owner
    asked for this specifically: a new upload after stop must not silently
    reuse old files).
    """
    cfg = await save_config({"enabled": False})
    return {"ok": True, "enabled": cfg["enabled"]}


# --------------------------------------------------------------------------- #
# Deterministic chat triggers - plain words, no LLM, work identically from
# Telegram or the web dashboard chat (both call app.agent.conversation.
# handle_message with the same text). This is the whole point: the owner
# must be able to run this even while every LLM provider is down.
# --------------------------------------------------------------------------- #
_START_WORDS = {
    "start", "shuru", "shuru koro", "shuru korbo", "start koro", "cholo",
    "run", "go", "begin", "done", "shesh", "sesh", "shesh hoise", "sesh hoise",
    "finish", "finished",
}
_STOP_WORDS = {
    "stop", "off", "bondho", "bondho koro", "bondho kore dao", "thamao",
    "pause", "stop koro",
}
_RESUME_PHRASES = ("age-r file", "agerfile", "purono file", "same file", "old file", "agerta")

# Removing things. "skip"/"bad dao" while a tag question is open means "not
# this one", which is the moment the owner most often realises a file should
# not go in at all.
_SKIP_WORDS = {
    "skip", "bad", "bad dao", "baddao", "bad de", "baddo", "cancel", "no",
    "na", "eta na", "eta bad", "remove", "delete", "bad koro", "skip koro",
}

# Offered as buttons so the common answers are one tap instead of typing.
# The owner can still type anything else - these are shortcuts, not a
# closed list.
SERVICE_CHOICES = ("WhatsApp", "Telegram", "Facebook", "Google", "Instagram", "Signal")
INTERVAL_CHOICES = (5, 10, 15, 30, 60)
# Restock points offered as buttons. 0 means "wait until it is actually
# empty"; anything above refills while numbers are still left, so the
# country never goes dead between checks.
THRESHOLD_CHOICES = (0, 100, 200, 500, 1000, 5000)
# Cleanup style, phrased as the decision rather than the command: "just tidy
# up" vs "wipe and replace". The second needs /frcd, which needs the uid.
CLEANUP_CHOICES = (
    ("used", "\U0001F9F9 Used/expired only (/useddelete)"),
    ("force", "\U0001F5D1 Wipe country first (/frcd)"),
)
_CLEAR_PHRASES = ("sob bad", "sob remove", "clear queue", "queue clear",
                  "sob cancel", "clear all", "sob delete")
# "<country> bad dao" / "remove <country>" - the country is whatever is left
# once the removal word is taken out.
_REMOVE_PREFIXES = ("remove ", "delete ", "bad dao ", "bad koro ", "bad ")
_REMOVE_SUFFIXES = (" bad dao", " bad koro", " bad", " remove koro", " remove",
                    " delete koro", " delete", " cancel")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_skip_trigger(text: str) -> bool:
    return _normalize(text) in _SKIP_WORDS


def is_clear_trigger(text: str) -> bool:
    return any(phrase in _normalize(text) for phrase in _CLEAR_PHRASES)


def country_to_remove(text: str) -> str | None:
    """Extract the country from "<country> bad dao" / "remove <country>".

    Returns None when the message is not a removal request, so an ordinary
    sentence mentioning a country never deletes anything.
    """
    norm = _normalize(text)
    if is_clear_trigger(norm) or is_skip_trigger(norm):
        return None

    for prefix in _REMOVE_PREFIXES:
        if norm.startswith(prefix):
            rest = norm[len(prefix):].strip()
            return rest or None
    for suffix in _REMOVE_SUFFIXES:
        if norm.endswith(suffix):
            rest = norm[: -len(suffix)].strip()
            return rest or None
    return None


def is_start_trigger(text: str) -> bool:
    return _normalize(text) in _START_WORDS


def is_stop_trigger(text: str) -> bool:
    return _normalize(text) in _STOP_WORDS


def is_resume_trigger(text: str) -> bool:
    norm = _normalize(text)
    return any(phrase in norm for phrase in _RESUME_PHRASES)


async def _notify_owner(text: str) -> None:
    """Best-effort Telegram ping regardless of which channel (Telegram or the
    web dashboard) actually triggered the change - the owner asked to be
    told on Telegram either way.
    """
    try:
        from app.config import get_settings
        from app.telegram.notifier import Notifier

        settings = get_settings()
        if settings.owner_chat_id:
            await Notifier().send(settings.owner_chat_id, text)
    except Exception:  # noqa: BLE001 - a notify failure must never break the flow
        log.exception("otp_bot_notify_failed")


def _format_start_success(result: dict[str, Any]) -> str:
    added = [f for f in result["files"] if f.get("added", True)]
    held = [f for f in result["files"] if not f.get("added", True)]

    head = "\u2705 Auto-start" if result.get("auto") else "\u2705 Shuru hoye geche"
    lines = [f"{head} - {len(added)} file add kora holo {result['target_bot']}-e."]
    for f in added:
        country = f.get("country") or "?"
        lines.append(f"  {country} - {f['count']} number (tag: {f['tag']})")
        lines.append(f"     {str(f['reply'])[:150]}")
    # Files held back are not failures - the country already had numbers, so
    # sending them now would only produce duplicates. Say so plainly, or it
    # reads as "my file was ignored".
    for f in held:
        country = f.get("country") or "?"
        lines.append(
            f"  \u23F8 {country} - {f['count']} number rakha holo "
            f"(bot e ekhon {f.get('stock', 0):,} ache)"
        )
        lines.append("     Shesh hoar shathe shathe auto add hobe.")
    if result.get("kept_running"):
        lines.append(f"  \u267B\uFE0F Age theke cholche: {', '.join(result['kept_running'])}")
    lines.append("")
    # State the finish conditions explicitly. A run that stops on its own is
    # correct behaviour, but only if the owner knows it will - otherwise it
    # reads as "the automation broke overnight".
    limits = result.get("limits") or {}
    max_refills, run_minutes = limits.get("max_refills"), limits.get("run_minutes")
    if max_refills or run_minutes:
        parts = []
        if max_refills:
            parts.append(f"{max_refills} bar re-add")
        if run_minutes:
            parts.append(f"{run_minutes} min")
        lines.append(f"\u26A0\uFE0F Ei task {' othoba '.join(parts)} por NIJEI BONDHO hobe.")
        if limits.get("delete_when_done"):
            lines.append("   Tarpor oi country-r number bot theke muche jabe.")
        lines.append("   Limit chara chalate: /otpset run_minutes 0, /otpset max_refills 0")
    else:
        lines.append("\u267E\uFE0F Kono time/count limit nai - 'stop' na bola porjonto cholbe.")
    lines.append(
        "Periodically stock check hobe, khali hole nijei cleanup+re-add korbe."
    )
    return "\n".join(lines)


def tag_question(entry: dict[str, Any]) -> str:
    """Public wrapper for the tag prompt (used by the upload handlers)."""
    return _tag_question(entry)


def _tag_question(entry: dict[str, Any]) -> str:
    """Ask for one country's service/tag, showing what was actually detected.

    The owner needs to see WHICH country and how many numbers before naming a
    service - the same upload can contain several countries, and the answer
    differs per country.
    """
    country = entry.get("country") or "?"
    count = entry.get("count")
    detail = f"{count} number" if count else "numbers"
    return (
        f"\U0001F30D {country} - {detail}\n"
        f"(file: {entry.get('source_name') or entry['name']})\n\n"
        "Ei country-r jonno kon service/tag e add korbo? (jemon: WhatsApp, Telegram)"
    )


async def handle_start_trigger() -> str:
    """Owner said "start"/"done"/etc. Ask for any missing tag first (one
    question at a time), otherwise actually start.
    """
    entry = await next_untagged_entry()
    if entry is not None:
        await set_awaiting_tag_entry(entry["id"])
        return _tag_question(entry)

    queue = await get_queue()
    if not queue:
        active = await get_active_files()
        if active:
            # Nothing new was queued, but files are already active - "start"
            # after "stop" with no new upload almost always means "resume
            # what I had going". Low-risk and fully reversible (stop undoes
            # it), so just do it instead of asking.
            return await handle_resume_trigger()
        return "Kono file queue-e nai. Age file upload koro, tarpor 'start' bolo."

    result = await start_automation()
    if result["ok"]:
        text = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text}")
        return text
    return f"\u274C Start korte parlam na: {result['error']}"


def format_start_result(result: dict[str, Any]) -> str:
    """Public wrapper: callers outside this module format a start result
    through here rather than reaching for the private helper."""
    return _format_start_success(result)


async def _continue_after_tagging(prefix: str) -> str:
    """Either ask the next question or, if nothing is left to ask, start."""
    next_entry = await next_untagged_entry()
    if next_entry is not None:
        await set_awaiting_tag_entry(next_entry["id"])
        return f"{prefix}\n\n{_tag_question(next_entry)}"

    if not await get_queue():
        return f"{prefix}\n\nQueue ekhon khali. Notun file dao, tarpor 'start' bolo."

    result = await start_automation()
    if result["ok"]:
        text_out = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text_out}")
        return f"{prefix}\n\n{text_out}"
    return f"{prefix}\n\n\u274C Start korte parlam na: {result['error']}"


async def handle_tag_answer(text: str) -> str:
    """Owner just answered a "what tag for X" question."""
    entry = await get_awaiting_tag_entry()
    if entry is None:
        return "Kono file tag-er jonno wait kortese na. 'start' bolle shuru hobe."

    label = entry.get("country") or entry["name"]

    # "skip" / "bad dao" here means "not this one" - the tag question is
    # exactly when the owner notices a country they did not mean to send.
    if is_skip_trigger(text):
        await remove_from_queue(entry["id"])
        return await _continue_after_tagging(f"\U0001F5D1 {label} bad deoa holo.")

    await set_queue_tag(entry["id"], text)
    await set_awaiting_tag_entry(None)
    return await _continue_after_tagging(f"\u2705 {label} -> {text.strip()[:60]}")


async def handle_remove_country(country: str) -> str:
    """Owner said "<country> bad dao" - drop it from queue and active alike."""
    removed = await remove_by_country(country)
    if not removed:
        queue = await get_queue()
        active = await get_active_files()
        known = sorted({e.get("country") or "?" for e in queue + active})
        available = ", ".join(known) if known else "(kichu nai)"
        return f"'{country}' khuje pelam na.\nEkhon ache: {available}"

    total = sum(e.get("count") or 0 for e in removed)
    label = removed[0].get("country") or country
    text = f"\U0001F5D1 {label} bad deoa holo ({len(removed)} entry, {total} number)."

    # If that removal answered the open question, keep the flow moving.
    if await get_awaiting_tag_entry() is None and await next_untagged_entry() is not None:
        return await _continue_after_tagging(text)
    return text


async def handle_clear_queue() -> str:
    """Owner said "sob bad dao" - empty the not-yet-started queue."""
    count = await clear_queue()
    if not count:
        return "Queue emnitei khali."
    return (
        f"\U0001F5D1 Queue clear kora holo ({count} entry bad).\n"
        "Cholte thaka file gulo (active) thik-i ache - oigula bondho korte 'stop' bolo."
    )


async def handle_stop_trigger() -> str:
    cfg = await get_config()
    if not cfg["enabled"]:
        return "Automation already off."
    await stop_automation()
    await _notify_owner("\U0001F534 OTP-bot automation bondho kora holo.")
    return (
        "\u23F8\uFE0F Automation bondho kora holo. Notun file dile ba 'start' bolle "
        "abar jiggesh korbo kon file diye shuru korbo."
    )


async def handle_resume_trigger() -> str:
    """Owner said "age-r file diye shuru koro" (resume with previously active
    files, already tagged, without needing a fresh upload).
    """
    active = await get_active_files()
    if not active:
        return "Kono age-r active file nai - notun file upload koro, tarpor 'start' bolo."
    await _save_files(QUEUE_KEY, active)
    result = await start_automation()
    if result["ok"]:
        text = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation resumed (age-r file diye).\n\n{text}")
        return text
    return f"\u274C Resume korte parlam na: {result['error']}"


# --------------------------------------------------------------------------- #
# The periodic cycle (called by the scheduler on its own timer, and by the
# dashboard's manual "Run now" button)
# --------------------------------------------------------------------------- #
async def run_cycle(config: dict[str, Any] | None = None, *, force: bool = False) -> CycleResult:
    """One stock-check-and-refill pass over the ACTIVE files. Never raises -
    always returns a result, even on failure.

    Countries share the target bot's single chat, so stock is read once per
    pass; what differs per country is the settings used to re-add it (limit,
    count, tag, cleanup mode), which come from otp_schedule.

    force=True ignores the per-country timers. The scheduler leaves it False
    so each country keeps its own pace, but a human pressing "Check now" is
    asking for a check right now - answering "not due yet" would make the
    button look broken.
    """
    cfg = config or await get_config()
    target = cfg["target_bot"]
    result = CycleResult(ok=False, action="error")

    active_files = await get_active_files()
    if not active_files:
        result.error = "no active files - run start() first"
        await _save_last_result(result)
        return result

    # Only refill the countries whose own timer has come up. With a single
    # shared interval a slow country was being re-added on the fast one's
    # clock, which is pure noise for the target bot.
    countries = [e.get("country") or e["name"] for e in active_files]
    if force:
        # A manual check looks at everything, and still re-arms the timers so
        # the next automatic pass is measured from now.
        ready = countries
        for country in countries:
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            await otp_schedule.arm_country(
                country, int(entry_cfg.get("interval_minutes", 10))
            )
    else:
        ready = await otp_schedule.due_countries(countries, cfg)
    if not ready:
        result.ok = True
        result.action = "not due yet"
        await _save_last_result(result)
        return result

    due_files = [
        e for e in active_files if (e.get("country") or e["name"]) in set(ready)
    ]

    try:
        # Anchor on the last bot message before asking, so a slow answer is
        # waited for rather than the previous one being mistaken for it.
        before_quota = await _last_bot_message(target)
        before_quota_id = int(before_quota.get("id", 0)) if before_quota else None

        sent_stock = await get_userbot().send_message(target, cfg["quota_command"])
        # Only a message that actually carries per-country stock counts as
        # the answer: the bot also posts its own add/progress notices, and
        # one of those arriving first previously produced a misread.
        stock_msg = await _wait_for_new_bot_reply(
            target,
            after_id=before_quota_id,
            timeout_s=45.0,
            matches=lambda text: bool(_parse_country_stock(text)) or _parse_quota(text) is not None,
        )

        # Tidy up the check itself: the stock command and its reply are
        # bookkeeping, and on a 2-minute interval they bury the chat the
        # owner actually reads. Done after the reply is captured, so the
        # data is already in hand.
        if cfg.get("tidy_stock_messages", True):
            await _tidy_messages(
                target,
                [sent_stock.get("message_id"), (stock_msg or {}).get("id")],
            )

        if stock_msg is None:
            result.error = (
                f"no reply from {target} to {cfg['quota_command']} "
                "(check the userbot is linked and the command still works)"
            )
            await _save_last_result(result)
            return result

        result.quota_reply = stock_msg["text"]
        stock = _parse_country_stock(stock_msg["text"])
        result.country_stock = stock
        # The bot's own list is the authority on what countries exist and how
        # they are spelled - learn it rather than trusting our prefix guess.
        result.new_countries = await learn_countries(stock)
        # Kept for the existing UI/notification surface, which still shows a
        # single headline number.
        result.active_quota = _parse_quota(stock_msg["text"])
        if result.active_quota is None and stock:
            result.active_quota = sum(stock.values())

        if not stock:
            # A plain /myquota reply has no per-country breakdown. Fall back
            # to the old all-or-nothing behaviour rather than refusing to
            # work, so an owner who kept /myquota configured is not stranded.
            if result.active_quota is None:
                result.error = "could not read stock from the reply"
                await _save_last_result(result)
                return result
            if result.active_quota > cfg["quota_threshold"]:
                result.ok = True
                result.action = "skipped"
                await _save_last_result(result)
                return result
            refill = due_files
        else:
            # Per-country decision: only refill the countries that are
            # actually empty. A country with stock left does not need
            # touching just because a sibling ran out.
            refill = []
            for entry in due_files:
                country = entry.get("country") or entry["name"]
                entry_cfg = await otp_schedule.effective_config(country, cfg)
                have = _stock_for(stock, country)
                if have is None:
                    # Not listed at all means the bot is holding none of it.
                    have = 0
                if have <= int(entry_cfg.get("quota_threshold", 0)):
                    refill.append(entry)

            if not refill:
                result.ok = True
                result.action = "skipped - every due country still has stock"
                await _save_last_result(result)
                return result

        # Clean up once, then re-add the countries that ran out - each with
        # its OWN limit/count/tag/cleanup mode.
        await get_userbot().send_message(target, cfg["cleanup_command"])
        await _sleep(2)

        replies: list[str] = []
        for entry in refill:
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)

            # Has this country reached its own finish line? Checked before
            # the re-add, so "3 refills" means 3, not 4.
            done_reason = await otp_schedule.finished_reason(country, cfg)
            if done_reason:
                finished = await _finish_country(target, entry, entry_cfg, done_reason)
                result.finished.append(finished)
                continue

            reply = await _add_one_file(target, entry, entry_cfg)
            await otp_schedule.record_refill(country)
            result.files_processed.append(entry["name"])
            if reply:
                replies.append(f"{country}: {reply}")

            # The file is spent when the bot had no stock AND the re-add
            # contributed nothing new - every number in it is already known.
            added = _parse_added_count(reply)
            if added == 0:
                result.exhausted.append({
                    "country": country,
                    "name": entry["name"],
                    "had_stock": _stock_for(stock, country) or 0,
                })

        result.add_reply = "\n".join(replies)
        result.ok = True
        result.action = f"added ({len(refill)} countr{'y' if len(refill) == 1 else 'ies'})"
        await _save_last_result(result)
        return result

    except UserbotError as exc:
        result.error = f"telegram account not linked or errored: {exc}"
        await _save_last_result(result)
        return result
    except FileNotFoundError as exc:
        result.error = str(exc)
        await _save_last_result(result)
        return result
    except Exception as exc:  # noqa: BLE001 - a cycle must never crash the runner
        log.exception("otp_bot_cycle_error")
        result.error = f"unexpected error ({type(exc).__name__}): {exc}"
        await _save_last_result(result)
        return result
