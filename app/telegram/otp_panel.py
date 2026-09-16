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

from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.automation import otp_bot

PREFIX = "otp"


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
        InlineKeyboardButton(text="\U0001F5D1 Bad dao", callback_data=f"{PREFIX}:skip:{entry_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def interval_keyboard(current: int | None = None) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            # A tick on the active one, so the current setting is visible
            # without a second message explaining what it already is.
            text=(f"\u2705 {minutes} min" if minutes == current else f"{minutes} min"),
            callback_data=f"{PREFIX}:int:{minutes}",
        )
        for minutes in otp_bot.INTERVAL_CHOICES
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=3))


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
            InlineKeyboardButton(text="\u23F1 Interval", callback_data=f"{PREFIX}:ask_int:"),
            InlineKeyboardButton(text="\U0001F9F9 Cleanup", callback_data=f"{PREFIX}:ask_clean:"),
        ],
        [
            InlineKeyboardButton(text="\U0001F30D Per country", callback_data=f"{PREFIX}:ask_country:"),
            InlineKeyboardButton(text="\U0001F4D0 Presets", callback_data=f"{PREFIX}:ask_preset:"),
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
    """One button per country, carrying a short index rather than the name.

    Country names are unbounded and often non-ASCII, and callback_data is
    capped at 64 bytes, so the index is resolved back to a name by the
    handler against the same ordering.
    """
    seen: list[str] = []
    for entry in entries:
        country = entry.get("country")
        if country and country not in seen:
            seen.append(country)
    if not seen:
        return None
    buttons = [
        InlineKeyboardButton(text=country, callback_data=f"{PREFIX}:{action}:{index}")
        for index, country in enumerate(seen)
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


def preset_keyboard(names: list[str], country_index: int) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=name, callback_data=f"{PREFIX}:usepre:{country_index}:{index}"
        )
        for index, name in enumerate(names)
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=2))


def country_interval_keyboard(country_index: int, current: int | None) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=(f"\u2705 {m} min" if m == current else f"{m} min"),
            callback_data=f"{PREFIX}:cint:{country_index}:{m}",
        )
        for m in otp_bot.INTERVAL_CHOICES
    ]
    return InlineKeyboardMarkup(inline_keyboard=_rows(buttons, per_row=3))


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


async def status_text() -> str:
    """One compact summary of everything the panel can change."""
    from app.automation import otp_schedule

    config = await otp_bot.get_config()
    queue = await otp_bot.get_queue()
    active = await otp_bot.get_active_files()
    last = await otp_bot.get_last_result()

    lines = [
        "\U0001F501 OTP-bot automation",
        "",
        f"Running: {'yes' if config['enabled'] else 'no'}",
        f"Target: {config['target_bot']}",
        f"Stock check: {config['quota_command']}",
        f"Cleanup: {'wipe country first (/frcd)' if config.get('force_delete_before_add') else 'used/expired only'}",
    ]

    if active:
        # Per-country, because that is now the unit of scheduling: each has
        # its own interval and its own next-run time.
        overview = {
            row["country"]: row
            for row in await otp_schedule.schedule_overview(
                [e.get("country") or e["name"] for e in active], config
            )
        }
        lines += ["", "Running now:"]
        for entry in active:
            country = entry.get("country") or entry["name"]
            row = overview.get(country, {})
            due = row.get("due_in_seconds")
            when = "due now" if due is None or due <= 0 else f"in {max(1, due // 60)}m"
            mark = " *" if row.get("customised") else ""
            lines.append(
                f"  {country} ({entry.get('count') or 0}) - {entry.get('tag') or 'General'}"
                f" | every {row.get('interval_minutes')}m, next {when}{mark}"
            )
        if any(overview.get(c, {}).get("customised") for c in overview):
            lines.append("  (* = custom settings for that country)")

    if queue:
        lines += ["", "Waiting to start:"]
        lines += [
            f"  {e.get('country') or e['name']} ({e.get('count') or 0})"
            f" - {e.get('tag') or 'no service yet'}"
            for e in queue
        ]
    if not active and not queue:
        lines += ["", "Nothing queued or running - send a numbers file here."]

    if last:
        outcome = "ok" if last.get("ok") else "FAILED"
        lines += ["", f"Last check: {outcome} ({last.get('action', '')})"]
        stock = last.get("country_stock") or {}
        if stock:
            lines.append("Live stock:")
            lines += [f"  {name}: {count}" for name, count in stock.items()]
        elif last.get("active_quota") is not None:
            lines.append(f"Quota then: {last['active_quota']}")
        if last.get("error"):
            lines.append(f"Error: {last['error'][:200]}")

    return "\n".join(lines)
