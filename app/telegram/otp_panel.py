"""Inline-keyboard layer for the OTP-number automation.

Kept apart from bot.py because it is a self-contained conversation: the
keyboards, the callback payloads they carry, and the handlers that consume
them all have to agree, and having them in one file makes that checkable at
a glance instead of spread across a 900-line module.

Callback data is `otp:<action>:<argument>`. Telegram caps callback_data at 64
bytes, so arguments are always short ids or enum-ish tokens - never a file
name or a country, both of which can easily blow the limit.
"""

from __future__ import annotations

import html
import re
import zlib
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.automation import otp_bot

PREFIX = "otp"

# Telegram REJECTS a message over 4096 characters - it does not truncate it -
# and the error lands in aiogram's log, not the chat, so an over-long status
# or start report simply never arrived. The cap is counted in UTF-16 units,
# where every emoji here is two, hence the margin below the real limit.
MESSAGE_LIMIT = 4000

# One /st reports every country at once, so there is ONE check interval for
# all of them. Buttons rendered before that (per-country "Check interval")
# still sit in chat history; tapping one answers with this instead of failing.
INTERVAL_IS_GLOBAL = "The check interval is now the same for all countries"


def country_key(country: str) -> str:
    """A short, stable id for one country, for use in callback_data.

    Buttons used to carry a country's POSITION in the active+queue list.
    That list reorders constantly - every start moves the restarted country
    to the end, a finished or removed country shifts everything after it -
    so a button rendered a minute earlier acted on a different country,
    "Remove" included. A hash of the name means the same country for as
    long as the button exists, and fits the 64-byte cap however long the
    name ("Saint Vincent And The Grenadines") is.
    """
    flat = " ".join(country.casefold().split())
    return "k" + format(zlib.crc32(flat.encode("utf-8")), "08x")


def is_all_countries(token: str) -> bool:
    """True for the global pickers' target (-1: On at / Off at / run length)
    rather than one country's key."""
    return token.strip().startswith("-")


def resolve_country(token: str, names: list[str]) -> str | None:
    """The country a button's token refers to, or None if it is gone.

    A bare number is a button rendered before keys existed: still honoured
    so old messages keep working, but bounds-checked both ways - a negative
    index used to wrap round to the LAST country.
    """
    token = token.strip()
    if token.startswith("k"):
        return next((name for name in names if country_key(name) == token), None)
    try:
        index = int(token)
    except ValueError:
        return None
    return names[index] if 0 <= index < len(names) else None


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Break a reply into Telegram-sized pieces, at line breaks where possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            # One absurdly long line (a bot reply pasted whole): hard-cut it.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    # Telegram also rejects an empty message, which a run of blank lines at a
    # chunk boundary would otherwise produce.
    return [chunk for chunk in chunks if chunk.strip()] or [text[:limit]]



def upload_problem(data: bytes) -> str | None:
    """Why an upload to the OTP thread cannot be a numbers file, or None.

    Shared by the Telegram and web upload paths. Anything is read as text
    downstream: a spreadsheet becomes binary noise the country splitter
    files under "Unknown", which auto-start then sends to the target bot
    under a real tag. NUL bytes never occur in a UTF-8 text export, so they
    are the tell (a UTF-16 export carries a BOM and is let through).
    """
    if b"\x00" in data[:8192] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return (
            "\u274C This is not a text file (xlsx / zip / pdf?), so I did not queue it.\n"
            "\n"
            "\u2022 Send the numbers as a .txt file\n"
            "\u2022 One number per line"
        )
    if not re.search(rb"\d", data):
        return (
            "\u274C There are no numbers in this file, so I did not queue it.\n"
            "\n"
            "\u2022 Check that you sent the right file\n"
            "\u2022 Send the numbers as a .txt file, one per line"
        )
    return None


def _rows(buttons: list[InlineKeyboardButton], per_row: int = 2) -> list[list[InlineKeyboardButton]]:
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def service_keyboard(entry_id: str) -> InlineKeyboardMarkup:
    """Pick the service/tag for one country, or drop that country entirely."""
    buttons = [
        InlineKeyboardButton(
            text=service, callback_data=f"{PREFIX}:svc:{entry_id}:{service}"
        )
        for service in otp_bot.SERVICE_CHOICES
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(text="\U0001F5D1 Remove", callback_data=f"{PREFIX}:skip:{entry_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def interval_keyboard(current: int | None = None) -> InlineKeyboardMarkup:
    """The ONE check interval, shared by every country."""
    buttons = [
        InlineKeyboardButton(
            # A tick on the active one, so the current setting is visible
            # without a second message explaining what it already is.
            text=(f"\u2705 {minutes} min" if minutes == current else f"{minutes} min"),
            callback_data=f"{PREFIX}:int:{minutes}",
        )
        for minutes in otp_bot.INTERVAL_CHOICES
    ]
    rows = _rows(buttons, per_row=3)
    # A typed value (7 min, say) has no button of its own to tick, so the
    # Custom button carries it - otherwise the current setting is invisible.
    custom = current is not None and current not in otp_bot.INTERVAL_CHOICES
    rows.append([
        InlineKeyboardButton(
            text=(
                f"\u2705 Custom ({current} min)" if custom else "\u270F\uFE0F Custom"
            ),
            callback_data=f"{PREFIX}:intc:",
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def threshold_keyboard(current: int | None = None) -> InlineKeyboardMarkup:
    """Default restock point for every country without its own setting."""
    buttons = [
        InlineKeyboardButton(
            text=(
                ("\u2705 " if t == current else "")
                + ("When empty" if t == 0 else f"{t:,} left")
            ),
            callback_data=f"{PREFIX}:thr:{t}",
        )
        for t in otp_bot.THRESHOLD_CHOICES
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Custom", callback_data=f"{PREFIX}:thrc:"
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def cleanup_keyboard(force_enabled: bool = False) -> InlineKeyboardMarkup:
    buttons = []
    for mode, label in otp_bot.CLEANUP_CHOICES:
        active = (mode == "force") == bool(force_enabled)
        buttons.append([
            InlineKeyboardButton(
                text=(f"\u2705 {label}" if active else label),
                callback_data=f"{PREFIX}:clean:{mode}",
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def control_keyboard(enabled: bool) -> InlineKeyboardMarkup:
    """The main panel: start/stop, a manual check, and the settings pickers."""
    toggle = (
        InlineKeyboardButton(text="\u23F8 Stop", callback_data=f"{PREFIX}:stop:")
        if enabled
        else InlineKeyboardButton(text="\u25B6\uFE0F Start", callback_data=f"{PREFIX}:start:")
    )
    return InlineKeyboardMarkup(inline_keyboard=[
        [toggle, InlineKeyboardButton(text="\U0001F504 Check now", callback_data=f"{PREFIX}:run:")],
        [
            InlineKeyboardButton(
                text="\u23F1 Check interval", callback_data=f"{PREFIX}:ask_int:"
            ),
            InlineKeyboardButton(text="\U0001F4E6 Restock at", callback_data=f"{PREFIX}:ask_thr:"),
        ],
        [
            InlineKeyboardButton(text="\U0001F30D Per country", callback_data=f"{PREFIX}:ask_country:"),
            InlineKeyboardButton(text="\u23FB Country on/off", callback_data=f"{PREFIX}:ask_onoff:"),
        ],
        [
            InlineKeyboardButton(text="\u25B6\uFE0F On at", callback_data=f"{PREFIX}:ask_startall:"),
            InlineKeyboardButton(text="\u23F0 Off at", callback_data=f"{PREFIX}:ask_stopall:"),
        ],
        [
            InlineKeyboardButton(text="\u23F3 Run time", callback_data=f"{PREFIX}:ask_runall:"),
            InlineKeyboardButton(text="\U0001F9F9 Cleanup", callback_data=f"{PREFIX}:ask_clean:"),
        ],
        [
            InlineKeyboardButton(text="\U0001F4D0 Presets", callback_data=f"{PREFIX}:ask_preset:"),
            InlineKeyboardButton(text="\U0001F5D1 Delete file", callback_data=f"{PREFIX}:ask_del1:"),
        ],
        [
            InlineKeyboardButton(
                text="\U0001F30E Refresh countries", callback_data=f"{PREFIX}:refreshc:"
            ),
            InlineKeyboardButton(text="\u2699\uFE0F Settings", callback_data=f"{PREFIX}:settings:"),
        ],
        [
            InlineKeyboardButton(text="\U0001F4CB Status", callback_data=f"{PREFIX}:status:"),
            InlineKeyboardButton(text="\U0001F5D1 Clear queue", callback_data=f"{PREFIX}:clearq:"),
        ],
    ])


def country_keyboard(entries: list[dict[str, Any]], action: str) -> InlineKeyboardMarkup | None:
    """One button per country, carrying a short key rather than the name.

    Country names are unbounded and often non-ASCII, and callback_data is
    capped at 64 bytes, so the key is resolved back to a name by the
    handler (see country_key for why it is not the list position).
    """
    seen: list[str] = []
    for entry in entries:
        country = entry.get("country")
        if country and country not in seen:
            seen.append(country)
    if not seen:
        return None
    buttons = [
        InlineKeyboardButton(
            text=country, callback_data=f"{PREFIX}:{action}:{country_key(country)}"
        )
        for country in seen
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=2))


def country_names(entries: list[dict[str, Any]]) -> list[str]:
    """The ordering country_keyboard indexes into - kept in one place so the
    buttons and the handler cannot disagree.
    """
    seen: list[str] = []
    for entry in entries:
        country = entry.get("country")
        if country and country not in seen:
            seen.append(country)
    return seen


def preset_keyboard(names: list[str], country_index: int | str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=name, callback_data=f"{PREFIX}:usepre:{country_index}:{index}"
        )
        for index, name in enumerate(names)
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=2))


def country_threshold_keyboard(country_index: int | str, current: int | None) -> InlineKeyboardMarkup:
    """When to restock this country - by numbers left, not only at zero."""
    buttons = [
        InlineKeyboardButton(
            text=(
                ("\u2705 " if t == current else "")
                + ("When empty" if t == 0 else f"{t:,} left")
            ),
            callback_data=f"{PREFIX}:cthr:{country_index}:{t}",
        )
        for t in otp_bot.THRESHOLD_CHOICES
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Custom",
            callback_data=f"{PREFIX}:cthrc:{country_index}",
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def country_field_keyboard(country_index: int | str, country: str, paused: bool = False) -> InlineKeyboardMarkup:
    """What about this country do you want to change?

    No check interval here: one /st covers every country, so the interval
    is global (main panel -> Check interval).
    """
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text=("\u25B6\uFE0F Resume" if paused else "\u23F8 Pause"),
                callback_data=f"{PREFIX}:cpause:{country_index}",
            ),
            InlineKeyboardButton(
                text="\U0001F5D1 Remove", callback_data=f"{PREFIX}:rmc2:{country_index}"
            ),
        ],
        [
            InlineKeyboardButton(
                text="\U0001F4E6 Restock at",
                callback_data=f"{PREFIX}:pickt:{country_index}",
            ),
        ],
        [
            InlineKeyboardButton(
                text="\u23F0 Start time",
                callback_data=f"{PREFIX}:pickstart:{country_index}",
            ),
            InlineKeyboardButton(
                text="\U0001F6D1 Stop time",
                callback_data=f"{PREFIX}:pickst:{country_index}",
            ),
        ],
        [
            InlineKeyboardButton(
                text="\u23F3 Run time",
                callback_data=f"{PREFIX}:pickrt:{country_index}",
            ),
        ],
        [
            InlineKeyboardButton(
                text="\U0001F4D0 Preset", callback_data=f"{PREFIX}:prec:{country_index}"
            ),
        ],
    ])


def file_delete_keyboard(
    queue: list[dict[str, Any]], active: list[dict[str, Any]]
) -> InlineKeyboardMarkup | None:
    """One button per individual entry, queued or running, to drop just it.

    The country-level buttons elsewhere remove every entry for that country
    at once. That is usually what is wanted, but not when two uploads of the
    same country are in play and only one is wrong - the web dashboard could
    always delete a single entry and Telegram could not, which is the gap
    this closes.
    """
    buttons: list[InlineKeyboardButton] = []
    for entry in queue:
        label = entry.get("country") or entry.get("name") or "?"
        buttons.append(
            InlineKeyboardButton(
                text=f"\U0001F5D1 {label} ({entry.get('count') or 0}) \u00b7 queue",
                callback_data=f"{PREFIX}:rm1:q:{entry['id']}",
            )
        )
    for entry in active:
        label = entry.get("country") or entry.get("name") or "?"
        buttons.append(
            InlineKeyboardButton(
                text=f"\U0001F5D1 {label} ({entry.get('count') or 0}) \u00b7 running",
                callback_data=f"{PREFIX}:rm1:a:{entry['id']}",
            )
        )
    if not buttons:
        return None
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=1))


def stop_time_keyboard(country_index: int | str, current: str = "") -> InlineKeyboardMarkup:
    """Common stop times, in Dubai time - plus a custom option and 'never'."""
    choices = ("06:00", "09:00", "12:00", "18:00", "21:00", "23:30")
    buttons = [
        InlineKeyboardButton(
            text=(f"\u2705 {t}" if t == current else t),
            callback_data=f"{PREFIX}:cstop:{country_index}:{t}",
        )
        for t in choices
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Other time", callback_data=f"{PREFIX}:cstopc:{country_index}"
        ),
        InlineKeyboardButton(
            text=("\u2705 Never" if not current else "\u267E\uFE0F Never"),
            callback_data=f"{PREFIX}:cstop:{country_index}:never",
        ),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def start_time_keyboard(country_index: int | str, current: str = "") -> InlineKeyboardMarkup:
    """When should this country BEGIN, in Dubai time.

    Index -1 sets the global default every country inherits, matching how the
    stop-time picker already works.
    """
    choices = ("06:00", "09:00", "12:00", "18:00", "21:00", "23:00")
    buttons = [
        InlineKeyboardButton(
            text=(f"\u2705 {t}" if t == current else t),
            callback_data=f"{PREFIX}:cstart:{country_index}:{t}",
        )
        for t in choices
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Other time", callback_data=f"{PREFIX}:cstartc:{country_index}"
        ),
        InlineKeyboardButton(
            text=("\u2705 Now" if not current else "\u25B6\uFE0F Now"),
            callback_data=f"{PREFIX}:cstart:{country_index}:now",
        ),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def runtime_keyboard() -> InlineKeyboardMarkup:
    """How long a fresh upload should run. 0 = no limit, kept expressible."""
    buttons = [
        InlineKeyboardButton(
            text=otp_bot.format_run_minutes(minutes),
            callback_data=f"{PREFIX}:rt:{minutes}",
        )
        for minutes in otp_bot.RUNTIME_CHOICES
        if minutes > 0
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Other", callback_data=f"{PREFIX}:rtc:"
        ),
        InlineKeyboardButton(
            text="\u267E\uFE0F No limit", callback_data=f"{PREFIX}:rt:0"
        ),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def country_runtime_keyboard(country_index: int | str, current: int | None) -> InlineKeyboardMarkup:
    """Same choices, for one already-running country."""
    buttons = [
        InlineKeyboardButton(
            text=(
                ("\u2705 " if minutes == (current or 0) else "")
                + otp_bot.format_run_minutes(minutes)
            ),
            callback_data=f"{PREFIX}:crt:{country_index}:{minutes}",
        )
        for minutes in otp_bot.RUNTIME_CHOICES
        if minutes > 0
    ]
    rows = _rows(buttons, per_row=3)
    rows.append([
        InlineKeyboardButton(
            text="\u270F\uFE0F Other", callback_data=f"{PREFIX}:crtc:{country_index}"
        ),
        InlineKeyboardButton(
            text=("\u2705 No limit" if not current else "\u267E\uFE0F No limit"),
            callback_data=f"{PREFIX}:crt:{country_index}:0",
        ),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def country_toggle_keyboard(entries: list[dict[str, Any]], paused: set[str]) -> InlineKeyboardMarkup | None:
    """One on/off button per country, showing its current state."""
    names = country_names(entries)
    if not names:
        return None
    buttons = [
        InlineKeyboardButton(
            text=("\u23F8 " if country not in paused else "\u25B6\uFE0F ") + country,
            callback_data=f"{PREFIX}:cpause:{country_key(country)}",
        )
        for country in names
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=2))


def removal_keyboard(entries: list[dict[str, Any]]) -> InlineKeyboardMarkup | None:
    """One button per country currently queued or active, to drop it."""
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        country = entry.get("country")
        if country and country not in seen:
            seen[country] = entry
    if not seen:
        return None
    buttons = [
        InlineKeyboardButton(
            text=f"\U0001F5D1 {country}",
            # Keyed by entry id, not country name: country names are
            # unbounded (and non-ASCII in places) while callback_data is not.
            callback_data=f"{PREFIX}:rmc:{entry['id']}",
        )
        for country, entry in seen.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=2))

# --------------------------------------------------------------------------- #
# Status panel (Telegram HTML)
# --------------------------------------------------------------------------- #
# Short ASCII codes, not emoji, inside the table: an emoji is two cells wide
# in some monospace fonts and one in others, which knocks every column after
# it out of line. Explained by a legend under the table instead.
STATE_LEGEND = {
    "RUN": "running",
    "OFF": "turned off",
    "WAIT": "waiting for its start time",
    "HELD": "held back",
    "DONE": "finished",
    "USED": "file used up - send a new one",
}
_COUNTRY_WIDTH = 14


_STATE_WIDTH = 4


def _short_country(name: str, marked: bool = False) -> str:
    """The name cut to its column; a trailing '*' (inside the width) marks a
    country with settings of its own."""
    flat = " ".join(name.split())
    width = _COUNTRY_WIDTH - (1 if marked else 0)
    short = flat if len(flat) <= width else flat[: width - 1] + "."
    return short + ("*" if marked else "")


def _number(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


def _span(minutes: Any) -> str:
    """Compact duration for one table cell: 90 -> '1h30m', 2880 -> '2d'."""
    minutes = int(minutes or 0)
    if minutes <= 0:
        return "-"
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    return (f"{days}d" if days else "") + (f"{hours}h" if hours else "") + (
        f"{mins}m" if mins else ""
    )


def _dubai_hhmm(when: Any) -> str:
    from datetime import datetime

    from app.automation import otp_schedule

    try:
        moment = when if isinstance(when, datetime) else datetime.fromisoformat(str(when))
        return moment.astimezone(otp_schedule.DUBAI_TZ).strftime("%H:%M")
    except (TypeError, ValueError):
        return "?"


def html_to_plain(text: str) -> str:
    """The same status for places that show text as-is (web chat, history)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


async def _country_rows(
    active: list[dict[str, Any]], config: dict[str, Any], stock: dict[str, int]
) -> tuple[list[tuple[str, ...]], int | None]:
    """One table row per running country - state, name, sizes, end - plus
    the seconds until the next check (None when nothing is running).

    There is no per-country timing column: one /st checks every country, so
    the next check is the same for all of them and is shown once, in the
    summary. What stays per country is its STATE, including when a waiting
    country begins ("WAIT 21:00").
    """
    from app.automation import otp_schedule

    overview = {
        row["country"]: row
        for row in await otp_schedule.schedule_overview(
            [e.get("country") or e["name"] for e in active], config
        )
    }
    rows: list[tuple[str, ...]] = []
    dues: list[int] = []
    for entry in active:
        country = entry.get("country") or entry["name"]
        row = overview.get(country, {})
        # A finished country is also paused, so finished is checked first -
        # "DONE" says more than "OFF" about why nothing is happening.
        if entry.get("finished_at"):
            state = "DONE"
        elif entry.get("exhausted_at"):
            state = "USED"
        elif row.get("paused"):
            state = "OFF"
        elif not await otp_schedule.has_started(country, config):
            begins = await otp_schedule.pending_start_at(country, config)
            state = "WAIT " + (
                _dubai_hhmm(begins) if begins else str(row.get("start_at") or "?")
            )
        elif entry.get("held"):
            state = "HELD"
        else:
            state = "RUN"
            if row.get("due_in_seconds") is not None:
                dues.append(int(row["due_in_seconds"]))
        ends = str(row.get("stop_at") or "") or _span(row.get("run_minutes"))
        have = otp_bot._stock_for(stock, country) if stock else None
        rows.append((
            state, _short_country(country, bool(row.get("customised"))),
            _number(entry.get("count") or 0), _number(have), ends,
        ))
    # The global timer gives every running country the same due time; min()
    # only matters while an older per-country timer is still winding down.
    return rows, (min(dues) if dues else None)


def _table_line(cells: tuple[str, ...], state_width: int = _STATE_WIDTH) -> str:
    state, country, size, have, ends = cells
    return (
        f"{state:<{state_width}} {country:<{_COUNTRY_WIDTH}} {size:>7} {have:>7} {ends}"
    ).rstrip()


def _next_check(seconds: int) -> str:
    """'now', 'in 45s', 'in 3m' - for the one shared next check."""
    if seconds <= 0:
        return "now"
    if seconds < 60:
        return f"in {seconds}s"
    return f"in {_span(-(-seconds // 60))}"


async def status_text() -> str:
    """The panel, as Telegram HTML: a summary, then one table row per country.

    Send it with parse_mode="HTML". Every value that comes from the owner or
    the target bot is escaped - a country or bot reply containing "<" or "&"
    would otherwise make Telegram reject the whole message. It always fits
    in ONE message: rows that do not fit become "+N more", because a status
    that is too long is not shortened by Telegram but refused outright.
    """
    from app.automation import otp_schedule

    config = await otp_bot.get_config()
    queue = await otp_bot.get_queue()
    active = await otp_bot.get_active_files()
    last = await otp_bot.get_last_result()
    esc = html.escape

    threshold = int(config.get("quota_threshold") or 0)
    restock = f"{threshold:,} left" if threshold else "when empty"
    # Same wording as the Cleanup buttons, so the panel and the picker agree.
    cleanup = dict(otp_bot.CLEANUP_CHOICES).get(
        "force" if config.get("force_delete_before_add") else "used", ""
    )
    interval = max(1, int(config.get("interval_minutes") or 1))
    if last:
        last_check = (
            f"{'OK' if last.get('ok') else 'FAILED'} - {last.get('action') or '?'}"
            f" at {_dubai_hhmm(last.get('ran_at'))}"
        )
    else:
        last_check = "not yet"

    head = [
        "<b>\U0001F501 OTP-bot automation</b>",
        f"\U0001F551 {esc(otp_schedule.clock_now()['dubai_full'])}",
        "",
        f"<b>Running:</b> {'yes' if config.get('enabled') else 'no'}",
        f"<b>Target bot:</b> {esc(str(config.get('target_bot') or ''))}",
        f"<b>Stock command:</b> {esc(str(config.get('quota_command') or ''))}",
        f"<b>Restock point:</b> {esc(restock)}",
        f"<b>Cleanup mode:</b> {esc(cleanup)}",
    ]
    stock = dict((last or {}).get("country_stock") or {})
    rows, due = await _country_rows(active, config, stock) if active else ([], None)
    # One /st checks every country, so the interval and the next check are
    # the same for all of them - said once here, not repeated per row.
    every = f"every {interval} min (all countries)"
    if config.get("enabled") and due is not None:
        head.append(f"<b>Next check:</b> {esc(_next_check(due))} · {esc(every)}")
    else:
        head.append(f"<b>Checks:</b> {esc(every)}")
    head.append(f"<b>Last check:</b> {esc(last_check)}")
    if last and last.get("error"):
        head.append(f"<b>Error:</b> {esc(str(last['error'])[:200])}")

    header = ("St", "Country", "File", "Stock", "Ends")
    state_width = max([_STATE_WIDTH] + [len(r[0]) for r in rows])
    legend = [f"{code} {meaning}" for code, meaning in STATE_LEGEND.items()
              if any(r[0].split()[0] == code for r in rows)]
    if any(r[1].endswith("*") for r in rows):
        legend.append("* custom settings")
    queue_lines = [
        f"\u2022 {esc(e.get('country') or e['name'])} "
        f"({_number(e.get('count') or 0)}) - {esc(e.get('tag') or 'no service yet')}"
        for e in queue
    ]

    def render(shown_rows: int, shown_queue: int) -> str:
        out = list(head)
        if rows:
            body = [_table_line(header, state_width)] + [
                _table_line(r, state_width) for r in rows[:shown_rows]
            ]
            if shown_rows < len(rows):
                body.append(f"+{len(rows) - shown_rows} more")
            # Padded on the raw text, escaped afterwards: escaping first
            # would count "&amp;" as five characters and skew the column.
            out += [
                "",
                f"<b>Running now ({len(rows)})</b>",
                "<pre>" + esc("\n".join(body)) + "</pre>",
                "<i>" + esc(" \u00B7 ".join(legend)) + "</i>",
            ]
        if queue_lines:
            out += ["", f"<b>Waiting to start ({len(queue_lines)})</b>"]
            out += queue_lines[:shown_queue]
            if shown_queue < len(queue_lines):
                out.append(f"+{len(queue_lines) - shown_queue} more")
        if not rows and not queue_lines:
            out += ["", "Nothing queued or running - send a numbers file here."]
        return "\n".join(out)

    shown_rows, shown_queue = len(rows), len(queue_lines)
    text = render(shown_rows, shown_queue)
    # The waiting list goes first: what is running is what the owner opened
    # the panel to see.
    while len(text) > MESSAGE_LIMIT and (shown_rows or shown_queue):
        if shown_queue:
            shown_queue -= 1
        else:
            shown_rows -= 1
        text = render(shown_rows, shown_queue)
    return text
