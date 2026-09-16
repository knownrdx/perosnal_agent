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
            InlineKeyboardButton(text="\U0001F4CB Status", callback_data=f"{PREFIX}:status:"),
            InlineKeyboardButton(text="\U0001F5D1 Clear queue", callback_data=f"{PREFIX}:clearq:"),
        ],
    ])


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
    config = await otp_bot.get_config()
    queue = await otp_bot.get_queue()
    active = await otp_bot.get_active_files()
    last = await otp_bot.get_last_result()

    lines = [
        "\U0001F501 OTP-bot automation",
        "",
        f"Running: {'yes' if config['enabled'] else 'no'}",
        f"Target: {config['target_bot']}",
        f"Checks every: {config['interval_minutes']} min",
        f"Refill when active \u2264 {config['quota_threshold']}",
        f"Cleanup: {'wipe country first (/frcd)' if config.get('force_delete_before_add') else 'used/expired only'}",
    ]

    if active:
        lines += ["", "Running now:"]
        lines += [
            f"  {e.get('country') or e['name']} ({e.get('count') or 0}) - {e.get('tag') or 'General'}"
            for e in active
        ]
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
        if last.get("active_quota") is not None:
            lines.append(f"Quota then: {last['active_quota']}")
        if last.get("error"):
            lines.append(f"Error: {last['error'][:200]}")

    return "\n".join(lines)
