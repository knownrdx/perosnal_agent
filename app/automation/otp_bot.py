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
    3. start() sends, for every queued file: the file, then a reply to that
       file's message with the add-numbers command carrying its own tag.
       Successful files move from the queue into ACTIVE_FILES and the
       periodic monitor turns on. No cleanup command is ever sent.
    4. The scheduler (see app/workers/scheduler_worker.py) checks stock on
       ONE shared timer (interval_minutes, every country at once - a single
       /st answers for all of them) while enabled=True; a country at or
       below its threshold is re-added (same tags, no new prompts needed -
       they were already collected at start).
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

from app.automation import country_names, otp_schedule
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

# "start immediately", stored so it can be told apart from an empty string
# left behind by an older version that wrote start_at on every save.
START_NOW = "now"

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
    # No cleanup command exists any more: the owner does not want /useddelete
    # recycling used numbers back into stock, so nothing like it is sent.
    # Force-delete (Owner-level "/frcd <country> <uid>") is the one way to
    # wipe a country's stock outright before re-adding, and it is opt-in.
    "force_delete_command": "/frcd {country} {uid}",
    "force_delete_before_add": False,
    "force_delete_uid": "",      # owner's own uid; blank disables force-delete
    # ONE check interval for every country. A single /st reports stock for
    # all of them at once, so there is one shared timer, not one per country.
    "interval_minutes": 1,
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
    # Wall-clock stop time (Dubai, "HH:MM"), inherited by every country
    # unless it sets its own. "" = never.
    "stop_at": "",
    # Wall-clock START time (Dubai, "HH:MM"), same inheritance. "" = begin
    # immediately. 04:00 is the owner's default: the target bot is quietest
    # then, so a file uploaded during the day waits for the small hours
    # instead of competing with the evening traffic. A caption that names a
    # time overrides it, and "ekhoni" / the Now button clears it.
    "start_at": "04:00",
    # How long a fresh upload runs for when the owner says nothing about it.
    # 20h covers "upload tonight, still going tomorrow evening" without the
    # run being immortal; 0 would mean forever, which is what silently
    # happened before and is why runs were found still going days later.
    "default_run_minutes": 1200,
    # Ask about timing once per upload instead of assuming the default. Turn
    # this off and every upload just takes default_run_minutes.
    "ask_run_time": True,
    # Before adding, read the bot's own stock: a country that already holds
    # numbers does not need them sent again (the bot answers such an add with
    # pure duplicates). Monitoring still starts, so the refill happens the
    # moment it actually runs out.
    "skip_add_if_stocked": True,
    "awaiting_tag_entry_id": None,  # set while a "what tag for X" prompt is pending
    # Bumped when a stored config needs a one-off correction on load; see
    # get_config(). A fresh install starts at the current version.
    "config_version": 3,
}

CONFIG_VERSION = 3

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
#
# The name is anything from the first letter up to the separator before the
# count. It used to be ASCII letters, spaces and .'- only, which silently
# dropped "Congo (DRC)", "Myanmar (Burma)", "C\u00f4te d'Ivoire", "Bosnia &
# Herzegovina", "T\u00fcrkiye" - and a dropped line reads as ZERO stock, so that
# country was cleaned up and re-added on every single cycle.
_STOCK_LINE_RE = re.compile(
    r"^[\s\u2022\-]*"            # bullet / indent
    r"[^\w(]*"                    # flag emoji and other leading symbols
    r"(?P<country>[^\W\d_][^:\n]*?)"
    r"\s*[:\-]\s*"
    r"(?P<count>\d[\d,]*)"
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

    The name we hold came from the phone-prefix table; the one in the reply
    came from the bot, and they do not always agree ("Congo (DRC)" vs "DR
    Congo"). A missed match reads as ZERO stock and triggers a re-add of the
    whole file on every cycle - the observed symptom was 16,499 duplicates
    skipped every five minutes, forever. See app/automation/country_names.py
    for why this is not a prefix comparison.
    """
    return country_names.find(stock, country)


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


# One country's line in an add result:
#     "Bangladesh: 0 added (5 dup)"
#     "🇨🇫 Central African Republic: 100000 added"
# Same name shape as a /st stock line (flag, then anything up to the colon).
_ADDED_LINE_RE = re.compile(
    r"^[\s•\-]*"
    r"[^\w(]*"
    r"(?P<country>[^\W\d_][^:\n]*?)"
    r"\s*:\s*"
    r"(?P<count>\d[\d,]*)\s+added",
    re.IGNORECASE | re.MULTILINE,
)
# "Total: 5 added" and the like are a summary, not a country.
_ADDED_LINE_LABELS = frozenset({
    "total", "totals", "done", "result", "results", "summary", "status",
    "added", "new", "numbers",
})


def _added_by_country(text: str) -> dict[str, int]:
    """Per-country counts from an add result, {} when it has none.

    "✅ 53412 added, 46588 duplicates skipped" has no country on it, so it
    yields {} - the caller then reads it the old, country-blind way.
    """
    found: dict[str, int] = {}
    for match in _ADDED_LINE_RE.finditer(text or ""):
        name = match.group("country").strip()
        if not name or name.casefold() in _ADDED_LINE_LABELS:
            continue
        try:
            found.setdefault(name, int(match.group("count").replace(",", "")))
        except ValueError:
            continue
    return found


def _added_for(text: str, country: str | None) -> int | None:
    """How many numbers an add contributed for ONE country.

    The chat is shared by every country's add, so a slow bot's "Bangladesh:
    0 added" can be the text another country's wait ends up holding. Reading
    that as the other country's count wrote off a good file (never refilled
    again) while the spent one kept being re-sent. So: this country's own
    line when the reply has per-country lines, None when it has lines but
    none of them is this country, and the plain count otherwise.
    """
    lines = _added_by_country(text)
    if lines and country:
        return country_names.find(lines, country)
    return _parse_added_count(text)


KNOWN_COUNTRIES_KEY = "otp_bot_known_countries"

# Shown whenever something tries to give one country its own interval.
INTERVAL_IS_GLOBAL = (
    "The check interval is the same for every country. "
    "Change it with: /otpset interval_minutes <minutes>"
)


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
        "label": "Check interval for all countries (minutes)",
        "min": 1,
        "max": 1440,
        "example": "1",
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
    "start_at": {
        "type": "time",
        "label": "Start at this time (Dubai, HH:MM; blank = right away)",
        "example": "21:00",
    },
    "default_run_minutes": {
        "type": "int",
        "label": "Default run length for a new upload (minutes, 0 = no limit)",
        "min": 0,
        "max": 100000,
        "example": "1200",
    },
    "ask_run_time": {
        "type": "bool",
        "label": "Ask how long a new upload should run",
        "example": "on / off",
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
        raise ValueError(f"'{key}' isn't a setting. Send /otpset to see the full list.")

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
        raise ValueError(f"{key}: please use 'on' or 'off'.")

    if spec["type"] == "int":
        try:
            number = int(value.replace(",", ""))
        except ValueError:
            raise ValueError(f"{key}: needs a number (e.g. {spec['example']}).") from None
        low, high = spec.get("min", 0), spec.get("max", 10**9)
        if not low <= number <= high:
            raise ValueError(f"{key}: must be between {low:,} and {high:,}.")
        return number

    # Strings: a template that loses its placeholders silently stops working,
    # so check the ones the sender actually substitutes.
    if key == "add_command_template" and "{tag}" not in value:
        raise ValueError("add_command_template must contain {tag}.")
    if key == "force_delete_command" and "{country}" not in value:
        raise ValueError("force_delete_command must contain {country}.")
    if key == "delete_done_command" and "{country}" not in value:
        raise ValueError("delete_done_command must contain {country}.")
    if key == "target_bot" and not value.startswith("@"):
        raise ValueError("target_bot must start with '@' (e.g. @PBDxbot).")
    return value[:200]


def _num(value: Any) -> str:
    """A count with thousands separators; '?' when there is none."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"


def _numbers(value: Any) -> str:
    """'1 number' / '7,625 numbers' - how counts read in owner messages."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "numbers"
    return f"{count:,} number{'' if count == 1 else 's'}"


async def set_setting_from_chat(key: str, raw: str) -> str:
    """Apply one setting and describe the result in one line."""
    value = coerce_setting(key, raw)
    await save_config({key: value})
    shown = "on" if value is True else "off" if value is False else (value or "(blank)")
    line = f"\u2705 Saved: {key} = {shown}"
    notes: list[str] = []

    # These two silently END a run, and a small number is very easy to read
    # as "every N" rather than "stop after N". Saying so at the moment it is
    # set is the difference between a deliberate short run and waking up to
    # a task that stopped minutes after you went to bed.
    if key == "run_minutes" and value:
        notes += [
            f"\u26A0\uFE0F Heads-up: the run will STOP by itself "
            f"{format_run_minutes(int(value))} after it starts.",
            "To keep it running all night: /otpset run_minutes 0",
        ]
    if key == "max_refills" and value:
        notes += [
            f"\u26A0\uFE0F Heads-up: the run will STOP by itself after "
            f"{int(value):,} re-add{'' if int(value) == 1 else 's'}.",
            "To run with no limit: /otpset max_refills 0",
        ]
    if key == "delete_when_done" and value is True:
        notes.append(
            "\u26A0\uFE0F Heads-up: when a run finishes, that country's numbers "
            "will be DELETED from the bot."
        )
    if key == "stop_at":
        from app.automation import otp_schedule

        clock = otp_schedule.clock_now()
        if value:
            notes += [
                f"\u23F0 The run will stop at {value} Dubai time.",
                f"It's now {clock['dubai']} Dubai / {clock['utc']} UTC.",
            ]
        else:
            notes.append("\u267E\uFE0F No stop time set.")
    if notes:
        line += "\n\n" + "\n".join(notes)
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
        "To change one: /otpset <key> <value>",
        "e.g. /otpset interval_minutes 15",
        "     /otpset target_bot @PBDxbot",
        "     /otpset force_delete_before_add on",
        "",
        "Just for one country: /otpset <country> <key> <value>",
        "e.g. /otpset Bangladesh quota_threshold 200",
        "(The check interval is shared by every country.)",
    ]
    return "\n".join(lines)


async def set_country_setting_from_chat(country: str, key: str, raw: str) -> str:
    """Per-country override of the same fields."""
    from app.automation import otp_schedule

    if key == "interval_minutes":
        # One /st answers for every country, so there is one shared check
        # timer - a per-country interval no longer means anything.
        raise ValueError(INTERVAL_IS_GLOBAL)
    if key not in otp_schedule.OVERRIDABLE:
        allowed = ", ".join(otp_schedule.OVERRIDABLE)
        raise ValueError(f"'{key}' can't be set per country. These can: {allowed}")

    value = coerce_setting(key, raw)
    canonical = await canonical_country(country)
    if key == "paused":
        # Through set_paused, so resuming re-arms the timer and starts a
        # fresh run for one that already ran out - written directly, a
        # finished country came back only to be finished again next pass.
        await otp_schedule.set_paused(canonical, bool(value))
    else:
        await otp_schedule.set_country_settings(canonical, {key: value})
    shown ="on" if value is True else "off" if value is False else value
    return f"\u2705 Saved for {canonical}: {key} = {shown}"


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
        return "\U0001F44D Okay, skipped \u2014 nothing was changed."

    try:
        if country:
            reply = await set_country_setting_from_chat(country, field, text)
        else:
            reply = await set_setting_from_chat(field, text)
    except ValueError as exc:
        if country and field == "interval_minutes":
            # A per-country interval question left over from before the
            # interval became global: no answer can ever satisfy it, so
            # close it instead of asking again forever.
            await set_pending_input(None)
            return f"\u274C {exc}"
        # Keep the question open: the owner meant to answer it, they just
        # typed something unusable, and dropping it would lose the context.
        return f"\u274C {exc}\n\nPlease try again, or say \"skip\" to cancel."

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
        if country_names.same_country(stored_key, key):
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


# Deadlines for one Telegram call. A call that never returns used to hold the
# conversation lock forever, and from then on every scheduler cycle queued up
# behind it. Uploading a large file legitimately takes longer than a message.
_BOT_CALL_TIMEOUT_S = 120.0
_SEND_FILE_TIMEOUT_S = 300.0
# How long a scheduled cycle waits for the conversation before giving up for
# this round. Owner-initiated start/refresh still wait their turn.
_CYCLE_LOCK_WAIT_S = 600.0


async def _bot_call(awaitable: Any, timeout_s: float) -> Any:
    """Await one userbot call, but never for longer than ``timeout_s``.

    A timeout is reported as a UserbotError, which every caller already
    handles as "Telegram failed" - so a hung call ends the operation cleanly
    and releases the lock instead of stalling everything behind it.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout_s)
    except asyncio.TimeoutError as exc:
        raise UserbotError(f"Telegram did not answer within {timeout_s:g} s") from exc


# One conversation with the target bot at a time. Every step here reads
# "the bot's newest message after X" as the answer to what it just sent, so
# two flows interleaving in that chat - the scheduler's refill and a Start
# pressed in Telegram - read each other's replies: an add reported with the
# stock table as its result, a stock check misread as zero. Keyed per event
# loop because an asyncio.Lock is bound to the loop it is first used on.
_conversation_locks: dict[int, asyncio.Lock] = {}


def _conversation_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _conversation_locks.get(id(loop))
    if lock is None:
        _conversation_locks.clear()  # a previous loop's lock is dead weight
        lock = _conversation_locks[id(loop)] = asyncio.Lock()
    return lock


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
    # Countries whose wall-clock start time arrived during this cycle, so
    # this pass is where their numbers were actually sent for the first time.
    started_now: list[dict[str, Any]] = field(default_factory=list)
    # Countries whose start time arrived while the bot still had stock for
    # them: started (monitored from now on) but the file is HELD until the
    # stock runs low, instead of wiping live numbers to add it.
    held_at_start: list[dict[str, Any]] = field(default_factory=list)
    # Per-country add failures. The cycle carries on with the other
    # countries instead of abandoning them all over one slow reply.
    add_errors: list[str] = field(default_factory=list)
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
        # Not a setting any more (config v3): no cleanup command is ever sent.
        merged.pop("cleanup_command", None)
        # A stored "" for start_at predates this setting existing: the field
        # was written on every save, so an install that never chose a start
        # time still holds an empty string, and that empty string would mask
        # the 04:00 default forever. An owner who genuinely wants uploads to
        # begin at once says so per upload ("ekhoni shuru") or clears it from
        # the panel, which writes START_NOW rather than "".
        if stored.get("start_at", START_NOW) == "":
            merged["start_at"] = DEFAULT_CONFIG["start_at"]

        version = int(stored.get("config_version") or 1)
        if version < CONFIG_VERSION:
            if version < 2:
                # Version 2: the owner asked for two things to stop for good.
                #   - /useddelete before every add. It put used numbers back
                #     into stock. (Version 3 removed the setting entirely.)
                #   - Wiping a country off the bot when its run ends. A
                #     finished run is now HELD (file kept, country paused).
                # Applied once and persisted, so the wipe can still be turned
                # back on deliberately afterwards.
                merged["delete_when_done"] = False
            if version < 3:
                # Version 3: ONE check for every country. A single /st
                # reports stock for all of them, so the old per-country
                # intervals (and the 10/60-minute defaults behind them) only
                # left countries waiting on stale timers. The cleanup
                # command is gone too (popped above), so nothing like
                # /useddelete can ever be sent again.
                merged["interval_minutes"] = 1
            merged["config_version"] = CONFIG_VERSION
            # Persisted once, BEFORE the per-country fixes below: those call
            # back into code that reads the config, which must already see
            # the new version rather than migrate a second time.
            async with session_scope() as session:
                await repo.set_setting(session, SETTING_KEY, merged)
            if version < 2:
                # The same goes for a country's own override (e.g. left by
                # the "One-shot burst" preset): turning only the global flag
                # off still had that country /frcd'd at the end of its run.
                for key, overrides in (await otp_schedule.get_all_country_settings()).items():
                    if overrides.get("delete_when_done") is True:
                        await otp_schedule.set_country_settings(
                            str(overrides.get("display_name") or key),
                            {"delete_when_done": False},
                        )
            if version < 3:
                # Every per-country interval goes, the old staggered due
                # times go with them, and the shared check is due right now
                # so nothing keeps waiting out an old 60-minute timer.
                await otp_schedule.migrate_to_shared_timer()
            log.info("otp_bot_config_migrated", extra={"version": CONFIG_VERSION})
    return merged


async def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    current = await get_config()
    current.update({k: v for k, v in patch.items() if k in DEFAULT_CONFIG})
    async with session_scope() as session:
        await repo.set_setting(session, SETTING_KEY, current)
    if "interval_minutes" in patch:
        # A shorter interval takes effect now: the shared check is pulled in
        # to one new interval from now instead of waiting out the old one.
        # (A longer one applies from the next check; nothing is postponed.)
        await otp_schedule.arm_country("", otp_schedule.interval_of(current))
    if "start_at" in patch:
        # Runs still waiting on the old default follow the new one; runs
        # already going are never paused by it.
        # Read back rather than taken from the patch: a stored "" is read
        # as the default start time, and the gate must match what runs see.
        effective = (await get_config()).get("start_at") or ""
        await otp_schedule.regate_waiting_runs(str(effective))
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


async def enqueue_file(
    path: str, name: str, caption: str = ""
) -> dict[str, Any]:
    """Analyse an uploaded file and queue one entry PER COUNTRY found in it.

    A single upload routinely mixes countries (a "numbers" export can hold
    Dominican Republic and Bangladesh side by side) and the target bot keeps
    stock per country, so adding the file whole would file every number under
    one country's tag. Each country therefore becomes its own queue entry
    with its own split-out file, and each carries the service/tag last used
    for that country so the common case needs no questions at all.

    ``caption`` is whatever the owner typed alongside the file. Anything it
    states - service, country, start/stop time, run length - is taken as the
    answer to a question that would otherwise be asked. Saying it once in the
    caption and being asked it again anyway is the bot wasting the owner's
    time.
    """
    from app.automation import otp_caption, phone_countries

    source = safe_path(path, must_exist=True)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    grouped = phone_countries.split_by_country(lines)

    stated = otp_caption.parse_caption(caption)

    # A country named in the caption OVERRIDES what the numbers look like:
    # the owner knows which stock this file is for, and a prefix table does
    # not (routing prefixes, ported ranges, a country the bot lists under a
    # name of its own). Only applied when the file is one bucket - with a
    # genuinely mixed file the per-number split is the more informative
    # answer and overriding it would merge distinct stock.
    if stated.get("country") and len(grouped) == 1:
        only = next(iter(grouped))
        if only != stated["country"]:
            grouped = {stated["country"]: grouped[only]}

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
            "tag": stated.get("tag") or await _decide_tag(name, country),
            "uploaded_at": _now_iso(),
        }
        if caption.strip():
            entry["caption"] = caption.strip()[:300]
        queue.append(entry)
        created.append(entry)

        # Timing the caption stated is a per-country setting, so it survives
        # into the run exactly like one set from a button.
        #
        # A field the caption did NOT state is CLEARED rather than left
        # alone. A start time is a one-off instruction about one upload
        # ("tonight, start at 21:00"), not a standing property of the
        # country, and leaving it set meant the next file for that country
        # silently inherited it - uploaded at noon, it sat there until 21:00
        # for no reason the owner could see. Clearing falls back to the
        # global default (start_at below), which is what a plain upload
        # should follow.
        timing: dict[str, Any] = {
            "start_at": stated.get("start_at"),
        }
        if "stop_at" in stated:
            timing["stop_at"] = stated["stop_at"]
        if "run_minutes" in stated:
            timing["run_minutes"] = int(stated["run_minutes"])
        await otp_schedule.set_country_settings(country, timing)
        if stated.get("tag"):
            await _set_last_tag(stated["tag"], country)

    await _save_files(QUEUE_KEY, queue)
    return {
        "batch_id": batch_id,
        "entries": created,
        "countries": {c: len(n) for c, n in grouped.items()},
        "caption": stated,
        "caption_note": otp_caption.describe(stated),
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


# --------------------------------------------------------------------------- #
# "How long should this run?" - asked once per upload, per country.
#
# Before this, an upload with no explicit limit ran until the owner
# remembered to stop it, which in practice meant runs found still going days
# later. The default is stated rather than silently applied: the owner is
# told "20h" and can take it, change it, or say there is no limit at all.
# --------------------------------------------------------------------------- #
RUNTIME_ASK_KEY = f"{SETTING_KEY}_awaiting_runtime"

# Offered as buttons. 0 means "no limit", which has to stay expressible -
# some runs genuinely should go until stopped by hand.
RUNTIME_CHOICES = (240, 480, 720, 1200, 1440, 2880, 0)


def format_run_minutes(minutes: int) -> str:
    """Human-readable run length, e.g. 1200 -> '20h'."""
    minutes = int(minutes or 0)
    if minutes <= 0:
        return "no limit"
    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days} day{'' if days == 1 else 's'}"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    if minutes > 60:
        return f"{minutes // 60}h {minutes % 60}m"
    return f"{minutes} min"


def parse_run_minutes(text: str) -> int:
    """Read a run length from free text. Raises ValueError with a message.

    Accepts the way the owner actually writes it: "20h", "20 ghonta",
    "90 min", "2 din", a bare number (minutes), and the various ways of
    saying "no limit at all".
    """
    from app.automation import otp_caption

    raw = text.strip()
    if not raw:
        raise ValueError("How long should it run? e.g. 20h — or say 'no limit'.")

    lowered = " ".join(raw.casefold().split())
    if any(phrase in lowered for phrase in otp_caption._NEVER_PHRASES):
        return 0
    if lowered in {"0", "no", "na", "never", "kichu na"}:
        return 0

    match = otp_caption._DURATION_RE.search(raw)
    if match:
        minutes = otp_caption._duration_minutes(int(match.group(1)), match.group(2))
        if minutes > 0:
            return minutes

    digits = raw.replace(",", "").strip()
    if digits.isdigit():
        # A bare number is minutes - consistent with run_minutes everywhere
        # else, and the buttons cover the common hour values anyway.
        value = int(digits)
        if value > 100_000:
            raise ValueError("That's too long — the maximum is 100,000 min.")
        return value

    raise ValueError(
        "Sorry, I didn't catch that. Try '20h', '90 min', '2 days' or 'no limit'."
    )


async def get_awaiting_runtime() -> dict[str, Any] | None:
    """The upload batch currently being asked about, if that prompt is open."""
    async with session_scope() as session:
        stored = await repo.get_setting(session, RUNTIME_ASK_KEY)
    return dict(stored) if stored else None


async def set_awaiting_runtime(countries: list[str] | None) -> None:
    async with session_scope() as session:
        await repo.set_setting(
            session,
            RUNTIME_ASK_KEY,
            {"countries": list(countries)} if countries else {},
        )


async def countries_needing_runtime(entries: list[dict[str, Any]]) -> list[str]:
    """Which of these countries have no run limit of their own yet.

    A country whose caption already said "20h", or that carries a limit from
    an earlier run, is not asked again - the question exists to stop runs
    being immortal by accident, not to be answered twice.
    """
    cfg = await get_config()
    if not cfg.get("ask_run_time", True):
        return []

    needing: list[str] = []
    for entry in entries:
        country = entry.get("country") or entry.get("name") or ""
        if not country or country in needing:
            continue
        own = await otp_schedule.get_country_settings(country)
        if own.get("run_minutes") or own.get("stop_at"):
            continue
        if int(cfg.get("run_minutes") or 0) or str(cfg.get("stop_at") or "").strip():
            # A global limit already applies; the country inherits it.
            continue
        needing.append(country)
    return needing


def runtime_question(countries: list[str], default_minutes: int) -> str:
    """Ask how long this upload should run, stating the default plainly."""
    from app.automation import otp_schedule

    who = ", ".join(countries[:4]) + (" ..." if len(countries) > 4 else "")
    clock = otp_schedule.clock_now()
    return (
        f"\u23F3 How long should {who} run?\n"
        "\n"
        f"\u2022 Default: {format_run_minutes(default_minutes)} "
        "(used if you skip this)\n"
        f"\u2022 Time now: {clock['dubai']} Dubai\n"
        "\n"
        "Tap a button below, or type e.g. '20h', '90 min', '2 days' or 'no limit'."
    )


async def apply_runtime_answer(countries: list[str], minutes: int) -> str:
    """Store the answer for every country in the batch and describe it."""
    for country in countries:
        await otp_schedule.set_country_settings(
            country, {"run_minutes": int(minutes)}
        )
    await set_awaiting_runtime(None)
    who = ", ".join(countries[:4]) + (" ..." if len(countries) > 4 else "")
    if minutes <= 0:
        return f"\u267E\uFE0F {who}: no time limit \u2014 runs until you say \"stop\"."
    return f"\u23F3 {who}: runs for {format_run_minutes(minutes)}, then stops by itself."


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
    messages = await _bot_call(get_userbot().read_messages(target, limit), _BOT_CALL_TIMEOUT_S)
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
    settle_s: float | None = None,
    keep_unmatched: bool = False,
) -> dict[str, Any] | None:
    """Wait for a bot message NEWER than ``after_id`` (optionally one whose
    text satisfies ``matches``).

    Every message newer than the anchor is examined, oldest first - not
    just the newest one. The bot can answer and then post something else
    within one poll (a notice, the next progress line), and looking only at
    the newest message skipped the actual answer forever.

    ``settle_s`` is for replies whose final form we can recognise but
    whose wording may vary: a new message that does not match is accepted
    anyway once nothing newer has appeared, and its text has not changed,
    for that long. It handles a bot that edits "Processing..." into the
    result (same id, new text) without ever hanging on a wording we did
    not anticipate. Without it, a non-matching message is skipped.

    ``keep_unmatched`` keeps the newest non-matching message WITHOUT ever
    accepting it early: the whole window is waited for a match, and only
    when it runs out is that message returned instead of None. That is what
    an add needs - a progress line that sits unchanged for a while is still
    a progress line, and settling on it let the real result land during the
    next country's wait and be credited to that country.
    """
    waited = 0.0
    pending: dict[str, Any] | None = None
    pending_key: tuple[int, str] | None = None
    pending_since = 0.0
    while waited < timeout_s:
        await _sleep(poll_s)
        waited += poll_s
        messages = await _bot_call(get_userbot().read_messages(target, 10), _BOT_CALL_TIMEOUT_S)
        fresh = sorted(
            (
                m for m in messages
                if not m.get("out")
                and (after_id is None or int(m.get("id", 0)) > after_id)
            ),
            key=lambda m: int(m.get("id", 0)),
        )
        for message in fresh:
            if matches is None or matches(message.get("text") or ""):
                return message
        if fresh and (settle_s is not None or keep_unmatched):
            newest = fresh[-1]
            key = (int(newest.get("id", 0)), newest.get("text") or "")
            if key != pending_key:
                pending, pending_key, pending_since = newest, key, waited
            elif settle_s is not None and waited - pending_since >= settle_s:
                return pending
    # Out of time: a reply we could not classify still beats "no reply".
    return pending


def _is_add_result(text: str) -> bool:
    """Does this look like the bot's FINAL answer to an add, rather than a
    progress line? Used to wait past "Processing..." to the real result."""
    if _parse_added_count(text) is not None or _looks_like_failure(text):
        return True
    return bool(re.search(r"complete|success|duplicate|\bdup\b", text, re.IGNORECASE))


def _is_add_result_for(text: str, country: str | None) -> bool:
    """_is_add_result, but only for THIS country's add.

    A result that names countries ("Bangladesh: 0 added (5 dup)") answers
    for those countries only: if none of them is ours it is some earlier,
    slower add finishing, and must not end our wait. A result that names no
    country ("✅ 53412 added, ...") cannot be told apart and counts as before.
    """
    lines = _added_by_country(text)
    if lines and country:
        return country_names.find(lines, country) is not None
    return _is_add_result(text)


def _add_wait_seconds(entry: dict[str, Any]) -> float:
    """How long to wait for an add's answer. A 60,000-number file takes the
    bot far longer than a 300-number one; a fixed 90 s timed out on big
    files and reported a perfectly good add as "no reply"."""
    count = int(entry.get("count") or 0)
    return 90.0 + min(count / 500.0, 210.0)


async def _force_delete_country(target: str, country: str, cfg: dict[str, Any]) -> str:
    """Owner-level /frcd for one country, when configured.

    Re-adding a country whose stock is still live would just pile duplicates
    on top (the bot reports them as "dup"), so when the owner wants a genuine
    replace rather than a top-up, this wipes that country's numbers first.
    It is the only "clean-up" this module ever sends, and only when
    force_delete_before_add is on.
    """
    uid = str(cfg.get("force_delete_uid") or "").strip()
    if not uid:
        return ""
    command = cfg["force_delete_command"].format(country=country, uid=uid)

    previous = await _last_bot_message(target)
    previous_id = int(previous.get("id", 0)) if previous else None
    await _bot_call(get_userbot().send_message(target, command), _BOT_CALL_TIMEOUT_S)
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
        await _bot_call(deleter(target, ids), _BOT_CALL_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - tidying is cosmetic
        log.warning("otp_bot_tidy_failed", extra={"ids": ids})


async def _finish_country(
    target: str, entry: dict[str, Any], cfg: dict[str, Any], reason: str
) -> dict[str, Any]:
    """End one country's run: stop adding to it, but HOLD its file.

    The file stays in the active set and the country is paused, so the run
    can be picked up again with one tap (resume) and nothing about it is
    lost. Removing the entry outright - what this used to do - threw the
    file away the moment a time limit passed, and the owner's complaint was
    exactly that: "it deletes the file instead of holding it".

    Clearing the country's numbers off the target bot is a separate, opt-in
    decision (delete_when_done). It is irreversible on the bot's side, so it
    must never be what happens by default just because a run ended.
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
                await _bot_call(
                    get_userbot().send_message(
                        target, command.format(country=country, uid=uid)
                    ),
                    _BOT_CALL_TIMEOUT_S,
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

    # Hold it either way - the run is over even if the delete could not be
    # done. Paused, not removed: the file and the country's settings stay.
    await _update_active_entry(
        entry["id"], finished_at=_now_iso(), finished_reason=reason,
    )
    await otp_schedule.mark_finished(country, reason)
    outcome["held"] = True
    return outcome


async def _update_active_entry(entry_id: str, **changes: Any) -> dict[str, Any] | None:
    """Set (or, with None, remove) fields on one stored active entry."""
    active = await get_active_files()
    for stored in active:
        if stored["id"] == entry_id:
            for key, value in changes.items():
                if value is None:
                    stored.pop(key, None)
                else:
                    stored[key] = value
            await _save_files(ACTIVE_KEY, active)
            return stored
    return None


async def clear_finished(country: str) -> None:
    """Called when a held country is resumed: it is an ordinary running
    country again, so the "finished" and "file used up" marks come off."""
    wanted = " ".join(country.strip().casefold().split())
    active = await get_active_files()
    changed = False
    for stored in active:
        name = " ".join((stored.get("country") or stored["name"]).casefold().split())
        if name == wanted:
            for key in ("finished_at", "finished_reason", "exhausted_at"):
                if stored.pop(key, None) is not None:
                    changed = True
    if changed:
        await _save_files(ACTIVE_KEY, active)


async def _start_scheduled(
    target: str,
    entries: list[dict[str, Any]],
    cfg: dict[str, Any],
    stock: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Begin the countries whose wall-clock start time has just arrived.

    Same rule as a manual start: a country the bot still holds stock for is
    NOT added to. The file is held and the country starts being monitored,
    so it is added the moment stock runs low. Adding regardless - what this
    used to do - meant that with "wipe before add" on, a scheduled start
    /frcd-ed a country's live numbers just to put the new file in.

    ``stock`` is the /st reading of this same cycle; None means unknown
    (old /myquota-style reply), in which case the file is added as before.

    Returns (started_with_add, held). Never raises - a failed add is
    reported and the country is retried next cycle rather than dropped.
    """
    started: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    if not entries:
        return started, held

    for entry in entries:
        country = entry.get("country") or entry["name"]
        entry_cfg = await otp_schedule.effective_config(country, cfg)
        have = _stock_for(stock, country) if stock else None
        threshold = int(entry_cfg.get("quota_threshold") or 0)

        if have is not None and have > threshold and cfg.get("skip_add_if_stocked", True):
            # The run clock starts now even though nothing was sent: the
            # start time has arrived, and from here on it is monitored and
            # refilled like any running country.
            await otp_schedule.begin_run(country)
            await _update_active_entry(
                entry["id"], waiting_start=None, held=True, stock_at_start=have,
            )
            entry.pop("waiting_start", None)
            held.append({"country": country, "stock": have, "count": entry.get("count")})
            continue

        outcome: dict[str, Any] = {
            "country": country,
            "count": entry.get("count"),
            "tag": entry.get("tag"),
            "added": False,
            "error": "",
        }
        try:
            outcome["reply"] = (await _add_one_file(target, entry, entry_cfg))[:300]
            outcome["added"] = True
        except (UserbotError, AddRejected, FileNotFoundError) as exc:
            outcome["error"] = str(exc)[:300]
        except Exception as exc:  # noqa: BLE001 - one bad country must not
            log.exception("otp_bot_wait_add_error")   # kill the whole cycle
            outcome["error"] = f"{type(exc).__name__}: {exc}"[:300]
        started.append(outcome)

        if outcome["added"]:
            # The run clock starts NOW, not when the file was uploaded: a
            # "20h" run scheduled for 21:00 means twenty hours from 21:00.
            # No gate is recorded because the wait is over - passing the
            # start time again here would make the country wait a second
            # time, tomorrow, having just begun.
            await otp_schedule.begin_run(country)
            await _update_active_entry(entry["id"], waiting_start=None, held=None)
            entry.pop("waiting_start", None)

    return started, held


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

    sent_file = await _bot_call(
        get_userbot().send_file(target, str(target_path)), _SEND_FILE_TIMEOUT_S
    )
    await _sleep(2)

    command_text = cfg["add_command_template"].format(
        tag=entry.get("tag") or "General",
        limit=cfg["limit"],
        count=cfg["count"],
        country=entry.get("country") or "",
    )
    await _bot_call(
        get_userbot().send_message(target, command_text, reply_to=sent_file.get("message_id")),
        _BOT_CALL_TIMEOUT_S,
    )

    # Wait for the RESULT ("... added", "Fast Add Complete!", a rejection),
    # not the first thing the bot says: a progress line taken as the answer
    # hid how many numbers were really added, so a spent file was never
    # noticed. No settling on an unrecognised message: a "Processing..."
    # that sat unchanged for 20 s was once taken as the answer, and the real
    # "Bangladesh: 0 added" then landed in the NEXT country's wait and was
    # credited to it. Only a result for THIS country (or one naming no
    # country) ends the wait; when the window runs out, the newest thing the
    # bot said is still returned rather than nothing.
    country = entry.get("country")
    reply = await _wait_for_new_bot_reply(
        target,
        after_id=previous_id,
        timeout_s=_add_wait_seconds(entry),
        matches=lambda text: _is_add_result_for(text, country),
        settle_s=None,
        keep_unmatched=True,
    )
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
    async with _conversation_lock():
        return await _refresh_countries_locked()


async def _refresh_countries_locked() -> dict[str, Any]:
    cfg = await get_config()
    target = cfg["target_bot"]
    try:
        before = await _last_bot_message(target)
        before_id = int(before.get("id", 0)) if before else None
        sent = await _bot_call(
            get_userbot().send_message(target, cfg["quota_command"]), _BOT_CALL_TIMEOUT_S
        )
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
        sent = await _bot_call(
            get_userbot().send_message(target, cfg["quota_command"]), _BOT_CALL_TIMEOUT_S
        )
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


async def start_automation(*, respect_schedule: bool = False) -> dict[str, Any]:
    """Consume every queued (and tagged) file: add each one as its own
    reply-based command, or hold it when its country is still stocked. The
    started files are merged into the active set and the periodic monitor
    (enabled) turns on.

    Starts NOW by default, which is what the name says and what pressing
    Start means. ``respect_schedule`` is for the one caller that is not an
    instruction - maybe_auto_start(), reacting to an upload nobody said
    anything about. That is where a default start time belongs; answering
    an explicit "shuru" with "sure, in six hours" is not.
    """
    async with _conversation_lock():
        return await _start_automation_locked(respect_schedule=respect_schedule)


async def _is_running(country: str, cfg: dict[str, Any]) -> bool:
    """Is this country currently being monitored (started, not finished)?"""
    wanted = " ".join(country.strip().casefold().split())
    for entry in await get_active_files():
        name = " ".join((entry.get("country") or entry["name"]).casefold().split())
        if name != wanted or entry.get("finished_at") or entry.get("waiting_start"):
            continue
        state = await otp_schedule.get_run_state(country)
        if state and not state.get("finished_at") and await otp_schedule.has_started(country, cfg):
            return True
    return False


async def _start_automation_locked(*, respect_schedule: bool) -> dict[str, Any]:
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
    # Files that went through (added, held or scheduled) leave the queue;
    # files that failed stay in it, so a retry never re-sends the ones that
    # already worked.
    started: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    errors: list[str] = []
    gates: dict[str, str] = {}
    # Set when an error outside the per-file handling cut the start short:
    # the files it never reached are neither started nor failed.
    aborted = False

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

            # A country with a wall-clock start time must not touch the
            # target bot yet. Arm it, show it, add nothing - the whole point
            # of "start at 21:00" is that nothing happens before 21:00.
            start_at = str(entry_cfg.get("start_at") or "").strip() if respect_schedule else ""
            if otp_schedule._is_now(start_at):
                start_at = ""
            own_start = (await otp_schedule.get_country_settings(country)).get("start_at")
            if start_at and own_start is None and await _is_running(country, cfg):
                # A fresh file for a country that is ALREADY running, with no
                # time stated for it: the default start time is for new runs,
                # never a reason to pause this one. Gating it here stopped a
                # live country being refilled until 04:00 the next morning.
                start_at = ""
            gates[country] = start_at
            if start_at:
                entry["waiting_start"] = start_at
                begins = otp_schedule.next_occurrence(start_at)
                started.append(entry)
                result["files"].append({
                    "name": entry["name"],
                    "country": entry.get("country"),
                    "count": entry.get("count"),
                    "tag": entry["tag"],
                    "interval_minutes": cfg.get("interval_minutes"),  # shared by every country
                    "added": False,
                    "waiting_start": start_at,
                    "starts_at": begins.isoformat() if begins else None,
                    "reply": f"starts at {start_at} Dubai time",
                    **_limits_of(entry_cfg),
                })
                continue
            entry.pop("waiting_start", None)

            have = _stock_for(stock, country) if stock else None
            threshold = int(entry_cfg.get("quota_threshold") or 0)
            if have is not None and have > threshold:
                # Already stocked: hold the file, watch the country.
                entry["skipped_add"] = True
                entry["held"] = True
                entry["stock_at_start"] = have
                started.append(entry)
                result["files"].append({
                    "name": entry["name"],
                    "country": entry.get("country"),
                    "count": entry.get("count"),
                    "tag": entry["tag"],
                    "interval_minutes": cfg.get("interval_minutes"),  # shared by every country
                    "added": False,
                    "stock": have,
                    "reply": f"bot already has {have:,} — kept until stock runs low",
                    **_limits_of(entry_cfg),
                })
            else:
                entry.pop("skipped_add", None)
                entry.pop("held", None)
                needs_add.append(entry)

        for entry in needs_add:
            country = entry.get("country") or entry["name"]
            # Each country runs on its own settings from the very first add,
            # not just from the second cycle onwards.
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            try:
                reply = await _add_one_file(target, entry, entry_cfg)
            except (AddRejected, FileNotFoundError) as exc:
                # One file failing must not lose track of the others: the
                # ones before it are already on the bot and need monitoring.
                failed.append(entry)
                errors.append(str(exc))
                continue
            started.append(entry)
            result["files"].append({
                "name": entry["name"],
                "country": entry.get("country"),
                "count": entry.get("count"),
                "tag": entry["tag"],
                "interval_minutes": cfg.get("interval_minutes"),  # shared by every country
                "added": True,
                "stock": _stock_for(stock, country) if stock else None,
                "reply": reply,
                **_limits_of(entry_cfg),
            })

    except UserbotError as exc:
        aborted = True
        errors.append(f"telegram account not linked or errored: {exc}")
    except Exception as exc:  # noqa: BLE001 - start() must never crash the caller
        # log.exception keeps the real traceback in the container logs; the
        # owner-facing string alone is not enough to debug a crash like the
        # recursive-_sleep one, which read simply as "maximum recursion depth
        # exceeded" with no indication of where.
        aborted = True
        log.exception("otp_bot_start_error")
        errors.append(f"unexpected error ({type(exc).__name__}): {exc}")

    if errors and not started:
        # Nothing went through: leave the queue exactly as it was, so a
        # retry is safe and nothing half-started is being monitored.
        result["error"] = "; ".join(errors)
        async with session_scope() as session:
            await repo.set_setting(session, LAST_START_KEY, result)
        return result

    for entry in started:
        country = entry.get("country") or entry["name"]
        # A country whose last run finished is paused (held). A new file
        # for it is the owner asking for a new run, so it comes off hold.
        was_finished = bool((await otp_schedule.get_run_state(country)).get("finished_at"))
        # Make sure the shared check is armed, added or not: the whole point
        # of holding a file is that monitoring still runs and adds it later.
        # This never pushes the shared check later for countries already
        # running - it only arms it if nothing is due sooner.
        await otp_schedule.arm_country(country, int(cfg.get("interval_minutes") or 1))
        # Fresh run: the refill count and the time limit both start now,
        # not from whenever this country last ran. For a country waiting
        # on a start time the clock is re-based when that time actually
        # arrives (see _start_scheduled) - what begin_run records here is
        # the anchor the start time is measured from, and the gate this
        # particular run must wait for.
        await otp_schedule.begin_run(country, start_at=gates.get(country, ""))
        if was_finished:
            await otp_schedule.set_paused(country, False)

    await save_config({"enabled": True, "awaiting_tag_entry_id": None})
    # MERGE into the active set, never replace it. A second upload used to
    # wipe every country already running - the owner adds Nigeria and
    # silently loses the Bangladesh run started an hour ago. A new file
    # for a country already running supersedes that country's entry only.
    existing = await get_active_files()
    fresh_countries = {(e.get("country") or e["name"]) for e in started}
    kept = [e for e in existing if (e.get("country") or e["name"]) not in fresh_countries]
    await _save_files(ACTIVE_KEY, kept + started)
    # Take out ONLY what went through, from the queue as it is NOW. Writing
    # back the failed list instead dropped every file an abort never reached
    # (in neither list) and every file uploaded while this start held the
    # lock - each upload is its own task and queues itself meanwhile.
    started_ids = {e.get("id") for e in started}
    current_queue = await get_queue()
    await _save_files(QUEUE_KEY, [e for e in current_queue if e.get("id") not in started_ids])
    if aborted:
        failed_ids = {e.get("id") for e in failed}
        failed = failed + [
            e for e in queue if e.get("id") not in started_ids and e.get("id") not in failed_ids
        ]
        result["partial"] = True
    result["kept_running"] = [e.get("country") or e["name"] for e in kept]
    # Surfaced so the owner is told up front when this run will end -
    # a silent finish minutes later looks identical to a crash.
    result["limits"] = {
        "max_refills": int(cfg.get("max_refills") or 0),
        "run_minutes": int(cfg.get("run_minutes") or 0),
        "delete_when_done": bool(cfg.get("delete_when_done")),
    }
    # Some files went in and some did not: the started ones are monitored,
    # the failed ones wait in the queue for a retry.
    result["failed"] = [
        {"name": e["name"], "country": e.get("country"), "count": e.get("count")}
        for e in failed
    ]
    result["error"] = "; ".join(errors)
    result["ok"] = True

    async with session_scope() as session:
        await repo.set_setting(session, LAST_START_KEY, result)
    return result


def _limits_of(entry_cfg: dict[str, Any]) -> dict[str, Any]:
    """The finish conditions that apply to one country, for the start
    message. Taken from the country's EFFECTIVE settings: reading only the
    global ones told the owner "no time limit" for a country whose own
    20h limit was about to end it."""
    return {
        "run_minutes": int(entry_cfg.get("run_minutes") or 0),
        "max_refills": int(entry_cfg.get("max_refills") or 0),
        "stop_at": str(entry_cfg.get("stop_at") or ""),
    }


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
    # Nor is one still waiting on "how long should this run?" - starting
    # first and asking after would leave a gap where the run has no limit.
    if await countries_needing_runtime(queue):
        return None

    result = await start_automation(respect_schedule=True)
    result["auto"] = True
    return result


async def set_cleanup_mode(mode: str) -> dict[str, Any]:
    """Pick between "add on top" ("used") and "wipe and replace" ("force").

    Exposed as a single choice because that is how the owner thinks about it.
    "used" keeps its old key for compatibility but sends nothing at all - no
    cleanup command exists any more; "force" turns /frcd on before each add.
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
_STATUS_WORDS = {
    "status", "ki obostha", "kemon cholche", "koto", "stock", "stock koto",
    "ki hocche", "koto ache", "report", "ekhon ki", "chole", "cholche",
    "kaj korche", "update", "info",
}
_HELP_WORDS = {
    "help", "sahajjo", "ki korte pari", "commands", "command", "cmd",
    "ki ki kora jay", "kivabe", "how", "?",
}


def is_status_trigger(text: str) -> bool:
    """A plain "what's happening" question, answerable with no model."""
    return _normalize(text) in _STATUS_WORDS


def is_help_trigger(text: str) -> bool:
    return _normalize(text) in _HELP_WORDS


async def handle_status_trigger() -> str:
    """Live status, read straight from state - no LLM, no waiting.

    This is the single most common question in the thread; routing it
    through a rate-limited provider made the bot look slow and stupid for
    information it already has in hand.
    """
    from app.telegram import otp_panel

    return await otp_panel.status_text()


def help_text() -> str:
    """Everything the thread understands. Explained in English; the example
    inputs stay in the Banglish/Bengali the owner actually types."""
    return (
        "\U0001F501 OTP-bot \u2014 what you can tell me\n"
        "\n"
        "Send a number file and it starts by itself. Anything you write in the\n"
        "file's caption is taken as the answer \u2014 I only ask about what's missing.\n"
        "\n"
        "Caption examples:\n"
        "\u2022 bangladesh whatsapp 20h \u2014 country + service + run time\n"
        "\u2022 shuru 21:00 bondho 06:00 \u2014 start and stop time (Dubai)\n"
        "\u2022 telegram, kono stop nai \u2014 service, no time limit\n"
        "\u2022 nigeria 2 din \u2014 country + how many days\n"
        "\u2022 rat 9 ta porjonto \u2014 everyday phrasing works too\n"
        "\n"
        "Bengali script works exactly the same:\n"
        "\u2022 \u09ac\u09be\u0982\u09b2\u09be\u09a6\u09c7\u09b6 \u09b9\u09cb\u09df\u09be\u099f\u09b8\u0985\u09cd\u09af\u09be\u09aa \u09e8\u09e6 \u0998\u09a8\u09cd\u099f\u09be\n"
        "\u2022 \u09b6\u09c1\u09b0\u09c1 \u09e8\u09e7:\u09e6\u09e6 \u09ac\u09a8\u09cd\u09a7 \u09e6\u09ec:\u09e6\u09e6\n"
        "\u2022 \u09b8\u0995\u09be\u09b2 \u09ec\u099f\u09be \u09a5\u09c7\u0995\u09c7 \u09b0\u09be\u09a4 \u09e7\u09e7\u099f\u09be \u09aa\u09b0\u09cd\u09af\u09a8\u09cd\u09a4\n"
        "\n"
        "Messages:\n"
        "\u2022 start / shuru \u2014 start running\n"
        "\u2022 stop / bondho \u2014 stop\n"
        "\u2022 status \u2014 what's running right now\n"
        "\u2022 20h / 90 min / limit nai \u2014 run time (when I ask)\n"
        "\u2022 <country> bad dao \u2014 drop that country\n"
        "\u2022 sob bad dao \u2014 empty the queue\n"
        "\n"
        "New files start at 04:00 (Dubai) by default. To start right away, write\n"
        "\"ekhoni shuru\" in the caption, or use \u25b6\ufe0f On at in the /otpbot panel.\n"
        "\n"
        "Commands:\n"
        "\u2022 /otpbot \u2014 button panel (everything is there)\n"
        "\u2022 /otpset \u2014 view / change settings\n"
        "\u2022 /otppreset \u2014 list / save / apply presets\n"
        "\n"
        "The buttons cover interval, restock point, country on/off, start time,\n"
        "stop time, run time, deleting a file and presets."
    )


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
# One shared check interval for every country (minutes).
INTERVAL_CHOICES = (1, 2, 5, 10, 15, 30)
# Restock points offered as buttons. 0 means "wait until it is actually
# empty"; anything above refills while numbers are still left, so the
# country never goes dead between checks.
THRESHOLD_CHOICES = (0, 100, 200, 500, 1000, 5000)
# What happens to a country's stock before its file is added, phrased as the
# decision rather than the command: leave it alone, or wipe and replace. Only
# the second sends anything (/frcd, which needs the uid); there is no
# "/useddelete" option - no cleanup command is ever sent.
CLEANUP_CHOICES = (
    ("used", "\U0001F9F9 Don't wipe before adding"),
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
    """Lower-cased, whitespace-collapsed text for trigger matching.

    Also transliterates Bengali script into the Banglish the trigger word
    lists are written in, and folds Bengali digits to ASCII. Every trigger in
    this module funnels through here, so "বন্ধ করো" stops the run exactly like
    "bondho koro" does - without duplicating a Bengali spelling into each of
    the dozen word lists below.

    This matters more here than anywhere else in the codebase: these triggers
    exist so the OTP thread keeps working when the LLM is down, and a message
    the patterns cannot read would be handed to the very model they are meant
    to bypass.
    """
    from app.agent import language

    return " ".join(language.normalise(text).strip().lower().split())


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


def _ends_when(file: dict[str, Any], limits: dict[str, Any]) -> str:
    """When one started country stops by itself - "after 20h or 5 re-adds",
    "at 23:30 Dubai time" - or "" when nothing will stop it.

    Read from the file's own (effective, per-country) limits. Results stored
    before those fields existed fall back to the global ones.
    """
    run_minutes = int(file.get("run_minutes", limits.get("run_minutes")) or 0)
    max_refills = int(file.get("max_refills", limits.get("max_refills")) or 0)
    stop_at = str(file.get("stop_at") or "").strip()
    if otp_schedule._is_now(stop_at):
        stop_at = ""

    after: list[str] = []
    if run_minutes > 0:
        after.append(format_run_minutes(run_minutes))
    if max_refills > 0:
        after.append(f"{max_refills:,} re-add{'' if max_refills == 1 else 's'}")
    parts: list[str] = []
    if after:
        parts.append("after " + " or ".join(after))
    if stop_at:
        parts.append(f"at {stop_at} Dubai time")
    return " or ".join(parts)


def _format_start_success(result: dict[str, Any]) -> str:
    files = list(result.get("files") or [])
    added = [f for f in files if f.get("added", True)]
    waiting = [f for f in files if f.get("waiting_start")]
    held = [f for f in files if not f.get("added", True) and not f.get("waiting_start")]
    target = result.get("target_bot") or "the bot"

    if result.get("auto"):
        head = "\u2705 Auto-started"
    elif result.get("resumed"):
        head = "\u2705 Resumed"
    else:
        head = "\u2705 Started"
    if added:
        head += f" \u2014 {len(added)} file{'' if len(added) == 1 else 's'} sent to {target}"
    else:
        head += f" \u2014 nothing sent to {target} yet"

    body: list[str] = []
    for f in added:
        line = f"\u2022 {f.get('country') or '?'} \u2014 {_numbers(f.get('count'))}"
        if f.get("tag"):
            line += f" ({f['tag']})"
        new = _parse_added_count(str(f.get("reply") or ""))
        if new is not None:
            line += f", {new:,} added"
        body.append(line)
    # Files held back are not failures - the country already had numbers, so
    # sending them now would only produce duplicates. Say so plainly, or it
    # reads as "my file was ignored".
    for f in held:
        body.append(
            f"\u23F8 Held: {f.get('country') or '?'} \u2014 {_numbers(f.get('count'))} kept "
            f"(bot still has {_num(f.get('stock', 0))}). "
            "It will be added automatically when stock runs low."
        )
    # Scheduled countries have not been touched at all yet - saying so is the
    # difference between "waiting as asked" and "my file was ignored".
    for f in waiting:
        tag = f" ({f['tag']})" if f.get("tag") else ""
        body.append(
            f"\u23F0 Scheduled: {f.get('country') or '?'} \u2014 {_numbers(f.get('count'))}{tag}, "
            f"starts at {f['waiting_start']} Dubai time."
        )
    if result.get("kept_running"):
        body.append(
            f"\u267B\uFE0F Still running from before: {', '.join(result['kept_running'])}"
        )

    # Some files went in and some did not: the failed ones are still queued.
    failed_section: list[str] = []
    failed = list(result.get("failed") or [])
    if failed:
        failed_section.append(
            f"\u274C {len(failed)} file{'' if len(failed) == 1 else 's'} could not be sent "
            "\u2014 kept in the queue for a retry:"
        )
        for f in failed:
            failed_section.append(
                f"\u2022 {f.get('country') or f.get('name') or '?'} \u2014 {_numbers(f.get('count'))}"
            )
        if result.get("error"):
            failed_section.append(f"Error: {str(result['error'])[:300]}")
        failed_section.append(
            f"Say \"start\" to try {'it' if len(failed) == 1 else 'them'} again."
        )

    # State the finish conditions explicitly, per country. A run that stops
    # on its own is correct behaviour, but only if the owner knows it will -
    # otherwise it reads as "the automation broke overnight".
    limits = result.get("limits") or {}
    ends: dict[str, str] = {}
    for f in added + held + waiting:
        country = f.get("country") or f.get("name") or "?"
        ends.setdefault(country, _ends_when(f, limits))
    limited = [(country, when) for country, when in ends.items() if when]
    unlimited = [country for country, when in ends.items() if not when]

    footer: list[str] = []
    if not limited:
        footer.append(
            "\u267E\uFE0F No time or re-add limit \u2014 it keeps running until you say \"stop\"."
        )
    else:
        if len(ends) == 1:
            country, when = limited[0]
            footer.append(f"\u23F3 {country} stops by itself {when}.")
        else:
            listed = ", ".join(f"{country} {when}" for country, when in limited)
            lead = "Each country stops by itself" if not unlimited else "Stops by itself"
            footer.append(f"\u23F3 {lead} ({listed}).")
        if unlimited:
            footer.append(
                f"\u267E\uFE0F No limit for {', '.join(unlimited)} \u2014 "
                "runs until you say \"stop\"."
            )
        if limits.get("delete_when_done"):
            footer.append("\U0001F5D1 When a run ends, its numbers are deleted from the bot.")
    footer.append("Stock is checked automatically; empty countries are refilled.")

    sections = [[head], body, failed_section, footer]
    return "\n\n".join("\n".join(section) for section in sections if section)


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
    detail = _numbers(count) if count else "numbers"
    return (
        f"\U0001F30D {country} — {detail}\n"
        f"File: {entry.get('source_name') or entry['name']}\n"
        "\n"
        "Which service (tag) should I add these under? (e.g. WhatsApp, Telegram)"
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
        return (
            "\U0001F4ED Nothing in the queue yet.\n"
            "\n"
            "Upload a number file first, then say \"start\"."
        )

    needing = await countries_needing_runtime(queue)
    if needing:
        cfg = await get_config()
        await set_awaiting_runtime(needing)
        return runtime_question(needing, int(cfg.get("default_run_minutes") or 0))

    result = await start_automation()
    if result["ok"]:
        text = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text}")
        return text
    return f"\u274C Couldn't start: {result['error']}"


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

    queue = await get_queue()
    if not queue:
        return f"{prefix}\n\nThe queue is empty now. Send a new file, then say \"start\"."

    # Tags are settled; the remaining question is how long it should run.
    # Asked once for the whole batch, and only for the countries that do not
    # already have an answer from their caption or a previous run.
    needing = await countries_needing_runtime(queue)
    if needing:
        cfg = await get_config()
        await set_awaiting_runtime(needing)
        return (
            f"{prefix}\n\n"
            + runtime_question(needing, int(cfg.get("default_run_minutes") or 0))
        )

    result = await start_automation()
    if result["ok"]:
        text_out = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text_out}")
        return f"{prefix}\n\n{text_out}"
    return f"{prefix}\n\n\u274C Couldn't start: {result['error']}"


async def handle_runtime_answer(text: str) -> str:
    """Owner answered the "how long should this run?" question."""
    pending = await get_awaiting_runtime()
    countries = list((pending or {}).get("countries") or [])
    if not countries:
        return "There's no run-time question open right now."

    if is_skip_trigger(text):
        # Skipping takes the stated default rather than leaving the run
        # unlimited - an unanswered question must not quietly become the
        # riskier option.
        cfg = await get_config()
        minutes = int(cfg.get("default_run_minutes") or 0)
        note = await apply_runtime_answer(countries, minutes)
        return await _continue_after_runtime(f"{note} (default)")

    try:
        minutes = parse_run_minutes(text)
    except ValueError as exc:
        return f"\u274C {exc}\n\nPlease try again \u2014 or say \"skip\" to use the default."

    note = await apply_runtime_answer(countries, minutes)
    return await _continue_after_runtime(note)


async def _continue_after_runtime(prefix: str) -> str:
    """The run length is settled - start, unless something else is pending."""
    next_entry = await next_untagged_entry()
    if next_entry is not None:
        await set_awaiting_tag_entry(next_entry["id"])
        return f"{prefix}\n\n{_tag_question(next_entry)}"

    if not await get_queue():
        return f"{prefix}\n\nThe queue is empty \u2014 send a new file."

    result = await start_automation()
    if result["ok"]:
        text_out = _format_start_success(result)
        await _notify_owner(f"\U0001F7E2 OTP-bot automation started.\n\n{text_out}")
        return f"{prefix}\n\n{text_out}"
    return f"{prefix}\n\n\u274C Couldn't start: {result['error']}"


async def handle_tag_answer(text: str) -> str:
    """Owner just answered a "what tag for X" question."""
    entry = await get_awaiting_tag_entry()
    if entry is None:
        return "No file is waiting for a service right now. Say \"start\" to begin."

    label = entry.get("country") or entry["name"]

    # "skip" / "bad dao" here means "not this one" - the tag question is
    # exactly when the owner notices a country they did not mean to send.
    if is_skip_trigger(text):
        await remove_from_queue(entry["id"])
        return await _continue_after_tagging(f"\U0001F5D1 {label} removed.")

    await set_queue_tag(entry["id"], text)
    await set_awaiting_tag_entry(None)
    return await _continue_after_tagging(f"\u2705 {label} \u2192 {text.strip()[:60]}")


async def handle_remove_country(country: str) -> str:
    """Owner said "<country> bad dao" - drop it from queue and active alike."""
    removed = await remove_by_country(country)
    if not removed:
        queue = await get_queue()
        active = await get_active_files()
        known = sorted({e.get("country") or "?" for e in queue + active})
        available = ", ".join(known) if known else "(nothing)"
        return f"\U0001F50D Couldn't find '{country}'.\n\nIn the list now: {available}"

    total = sum(e.get("count") or 0 for e in removed)
    label = removed[0].get("country") or country
    entries = f"{len(removed)} entr{'y' if len(removed) == 1 else 'ies'}"
    text = f"\U0001F5D1 {label} removed ({entries}, {_numbers(total)})."

    # If that removal answered the open question, keep the flow moving.
    if await get_awaiting_tag_entry() is None and await next_untagged_entry() is not None:
        return await _continue_after_tagging(text)
    return text


async def handle_clear_queue() -> str:
    """Owner said "sob bad dao" - empty the not-yet-started queue."""
    count = await clear_queue()
    if not count:
        return "\U0001F4ED The queue is already empty."
    return (
        f"\U0001F5D1 Queue cleared \u2014 {count} entr{'y' if count == 1 else 'ies'} removed.\n"
        "\n"
        "Files that are already running are untouched. Say \"stop\" to stop those too."
    )


async def handle_stop_trigger() -> str:
    cfg = await get_config()
    if not cfg["enabled"]:
        return "\u23F8\uFE0F The automation is already off."
    await stop_automation()
    await _notify_owner("\U0001F534 OTP-bot automation stopped.")
    return (
        "\u23F8\uFE0F Automation stopped.\n"
        "\n"
        "Your files are kept. To run again, send a new file or say \"start\" "
        "to resume the current ones."
    )


async def handle_resume_trigger() -> str:
    """Owner said "age-r file diye shuru koro" (resume with previously active
    files, already tagged, without needing a fresh upload).
    """
    active = await get_active_files()
    if not active:
        return (
            "\U0001F4ED There are no previous files to resume.\n"
            "\n"
            "Upload a new file, then say \"start\"."
        )
    # Copied in beside whatever is already queued, never over it.
    queue = await get_queue()
    queued_ids = {e.get("id") for e in queue}
    copies = [e for e in active if e.get("id") not in queued_ids]
    await _save_files(QUEUE_KEY, queue + copies)
    result = await start_automation()
    if not result["ok"]:
        # Nothing started: the copies must not linger in the queue as
        # duplicates of files that are still in the active set.
        copied_ids = {e.get("id") for e in copies}
        remaining = [e for e in await get_queue() if e.get("id") not in copied_ids]
        await _save_files(QUEUE_KEY, remaining)
    if result["ok"]:
        result["resumed"] = True
        text = _format_start_success(result)
        await _notify_owner(
            f"\U0001F7E2 OTP-bot automation resumed with the previous files.\n\n{text}"
        )
        return text
    return f"\u274C Couldn't resume: {result['error']}"


# --------------------------------------------------------------------------- #
# The periodic cycle (called by the scheduler on its own timer, and by the
# dashboard's manual "Run now" button)
# --------------------------------------------------------------------------- #
async def run_cycle(config: dict[str, Any] | None = None, *, force: bool = False) -> CycleResult:
    """One stock-check-and-refill pass over the ACTIVE files. Never raises -
    always returns a result, even on failure.

    ONE check for every country: a single /st reports stock for all of them,
    so there is one shared timer (interval_minutes) and, when it comes up,
    every started, unpaused country is looked at from that one reading. What
    still differs per country is what happens next - hold, refill or finish,
    with its own threshold/limit/count/tag/wipe mode from otp_schedule.

    force=True ignores the shared timer (and re-arms it from now). The
    scheduler leaves it False, but a human pressing "Check now" is asking
    for a check right now - answering "not due yet" would make the button
    look broken.

    The wait for the conversation is bounded (_CYCLE_LOCK_WAIT_S): if some
    other operation still holds it, this round reports "busy" and the next
    one tries again, instead of every cycle queueing up behind it.
    """
    lock = _conversation_lock()
    try:
        await asyncio.wait_for(lock.acquire(), _CYCLE_LOCK_WAIT_S)
    except asyncio.TimeoutError:
        log.warning("otp_bot_cycle_busy", extra={"waited_s": _CYCLE_LOCK_WAIT_S})
        result = CycleResult(
            ok=False,
            action="error",
            error="busy: another operation with the bot is still running",
        )
        await _save_last_result(result)
        return result
    try:
        return await _run_cycle_locked(config, force=force)
    finally:
        lock.release()


async def _run_cycle_locked(config: dict[str, Any] | None, *, force: bool) -> CycleResult:
    cfg = config or await get_config()
    target = cfg["target_bot"]
    result = CycleResult(ok=False, action="error")

    active_files = await get_active_files()
    if not active_files:
        result.error = "no active files - run start() first"
        await _save_last_result(result)
        return result

    # One shared timer for every country: when it comes up, every started,
    # unpaused country is due together (one /st answers for all of them).
    # Per-country timers left countries staggered - one checked now, the
    # next an hour later - for no benefit. A finished (held) country is
    # paused, so due_countries leaves it out.
    countries = list(dict.fromkeys(e.get("country") or e["name"] for e in active_files))
    if force:
        # A manual check looks at everything, and still re-arms the shared
        # timer so the next automatic pass is measured from now. A country
        # still waiting for its start time is exempt: "Check now" means check
        # what is running, not "override the schedule I just set".
        ready = [
            c for c in countries
            if await otp_schedule.has_started(c, cfg)
            and not await otp_schedule.is_paused(c)
        ]
        if ready:
            await otp_schedule.arm_shared_check(int(cfg.get("interval_minutes") or 1))
    else:
        ready = await otp_schedule.due_countries(countries, cfg)
    if not ready:
        result.ok = True
        result.action = "not due yet"
        await _save_last_result(result)
        return result

    ready_set = set(ready)
    due_files = [
        e for e in active_files
        if (e.get("country") or e["name"]) in ready_set and not e.get("finished_at")
    ]

    try:
        # Read stock FIRST, before anything is added. Everything below - a
        # scheduled start, a refill, a finish - is decided from this one
        # reading. Anchor on the last bot message before asking, so a slow
        # answer is waited for rather than the previous one mistaken for it.
        before_quota = await _last_bot_message(target)
        before_quota_id = int(before_quota.get("id", 0)) if before_quota else None

        sent_stock = await _bot_call(
            get_userbot().send_message(target, cfg["quota_command"]), _BOT_CALL_TIMEOUT_S
        )
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
        # bookkeeping, and on a 1-minute interval they bury the chat the
        # owner actually reads. Done after the reply is captured, so the
        # data is already in hand.
        if cfg.get("tidy_stock_messages", True):
            await _tidy_messages(
                target,
                [sent_stock.get("message_id"), (stock_msg or {}).get("id")],
            )

        if stock_msg is None:
            # Nothing is decided blind: no add, no start, no finish. A
            # scheduled start in particular waits for a reading - starting
            # without one is exactly how a still-stocked country got wiped.
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
        if not stock and result.active_quota is None:
            result.error = "could not read stock from the reply"
            await _save_last_result(result)
            return result

        # 1. Finish lines, for every due country that is running - not only
        # the empty ones. A 20h run ends at 20h even if the bot still has
        # numbers; checking only before a refill left "finished" countries
        # running for as long as their stock happened to last.
        running: list[dict[str, Any]] = []
        for entry in due_files:
            if entry.get("waiting_start"):
                continue
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            done_reason = await otp_schedule.finished_reason(country, cfg)
            if done_reason:
                result.finished.append(
                    await _finish_country(target, entry, entry_cfg, done_reason)
                )
            else:
                running.append(entry)

        # 2. Scheduled starts whose time has come: hold if still stocked,
        # add if not.
        newly_started = [e for e in due_files if e.get("waiting_start")]
        if newly_started:
            result.started_now, result.held_at_start = await _start_scheduled(
                target, newly_started, cfg, stock or None,
            )

        # 3. Refills: only countries actually at or below their own restock
        # point, and never a file already known to be used up.
        refill: list[dict[str, Any]] = []
        for entry in running:
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            if stock:
                have = _stock_for(stock, country)
                # Not listed at all means the bot is holding none of it.
                have = 0 if have is None else have
            else:
                # A plain /myquota reply has no per-country breakdown: fall
                # back to the old all-or-nothing reading.
                have = int(result.active_quota or 0)
            if have > int(entry_cfg.get("quota_threshold", 0)):
                continue
            if entry.get("exhausted_at"):
                # Re-sending a file the bot already answered with "0 added"
                # changes nothing - it just floods the bot every minute. The
                # owner was told once; a new file for this country clears it.
                continue
            refill.append(entry)

        if not refill:
            result.ok = True
            if result.started_now or result.held_at_start or result.finished:
                result.action = "started/finished"
            elif not stock:
                # Single-number /myquota reply: there is no per-country
                # picture to describe, just "healthy".
                result.action = "skipped"
            else:
                result.action = "skipped - every due country still has stock"
            await _save_last_result(result)
            return result

        # Re-add the countries that ran low - each with its OWN
        # limit/count/tag/wipe mode. No cleanup command is sent first.
        replies: list[str] = []
        for entry in refill:
            country = entry.get("country") or entry["name"]
            entry_cfg = await otp_schedule.effective_config(country, cfg)
            try:
                reply = await _add_one_file(target, entry, entry_cfg)
            except (AddRejected, FileNotFoundError) as exc:
                # One country's slow or rejected add must not leave every
                # other empty country unfilled until the next pass.
                result.add_errors.append(f"{country}: {exc}"[:300])
                continue
            await otp_schedule.record_refill(country)
            if entry.get("held"):
                await _update_active_entry(entry["id"], held=None)
            result.files_processed.append(entry["name"])
            if reply:
                replies.append(f"{country}: {reply}")

            # The file is spent when the bot had no stock AND the re-add
            # contributed nothing new - every number in it is already known.
            # Read THIS country's count only: another country's "0 added"
            # must never write this file off.
            added = _added_for(reply, entry.get("country"))
            if added == 0:
                await _update_active_entry(entry["id"], exhausted_at=_now_iso())
                result.exhausted.append({
                    "country": country,
                    "name": entry["name"],
                    "had_stock": _stock_for(stock, country) or 0 if stock else 0,
                })

        result.add_reply = "\n".join(replies)
        done = len(result.files_processed)
        result.action = f"added ({done} countr{'y' if done == 1 else 'ies'})"
        if result.add_errors:
            # Reported as a failure so the owner hears about it, but only
            # after every other country has been dealt with.
            result.ok = False
            result.error = "; ".join(result.add_errors)
        else:
            result.ok = True
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
