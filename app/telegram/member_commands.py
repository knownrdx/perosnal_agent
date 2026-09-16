"""Telegram commands for contacts/members and the self-updating skills memory.

Kept apart from capability_commands.py / setup_commands.py so each file
stays readable: this one owns `/members`, `/syncmembers` and `/skills`.
"""

from __future__ import annotations

from aiogram import Dispatcher
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger

log = get_logger(__name__)


def _channel_icon(channel: str) -> str:
    return "\U0001F4AC" if channel == "whatsapp" else "\U0001F465"


def register_member_handlers(dp: Dispatcher, guard) -> None:
    """Attach the commands. ``guard`` is the owner-only async check."""

    # ------------------------------------------------------------------ #
    # Contacts / members
    # ------------------------------------------------------------------ #
    @dp.message(Command("members"))
    async def _members(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        query = (command.args or "").strip()
        async with session_scope() as session:
            rows = (
                await repo.search_contacts(session, query, limit=20)
                if query
                else await repo.list_contacts(session, limit=20)
            )
        if not rows:
            await message.answer(
                "\U0001F465 No contacts yet.\n\nRun /syncmembers to pull them in."
                if not query
                else f"No contacts matching '{query}'."
            )
            return
        header = f"\U0001F465 Contacts matching '{query}':" if query else "\U0001F465 Recent contacts:"
        lines = [header, ""]
        for row in rows:
            who = row.display_name or row.username_or_phone or row.external_id
            extra = f" ({row.username_or_phone})" if row.username_or_phone and row.username_or_phone != who else ""
            lines.append(f"{_channel_icon(row.channel)} {who}{extra}")
        await message.answer("\n".join(lines)[:4000])

    @dp.message(Command("syncmembers"))
    async def _syncmembers(message: Message) -> None:
        if not await guard(message):
            return
        notice = await message.answer("\U0001F504 Syncing contacts...")
        from app.tools.contact_tools import contacts_sync

        try:
            result = await contacts_sync()
        except Exception as exc:  # noqa: BLE001 - report, never crash the bot
            await notice.edit_text(f"\u274C Sync failed: {str(exc)[:300]}")
            return

        telegram = result.get("telegram", {})
        whatsapp = result.get("whatsapp", {})
        lines = [
            f"\u2705 Synced {result.get('total_synced', 0)} contact(s).",
            "",
            f"Telegram: {telegram.get('synced', 0)} synced"
            + ("" if telegram.get("ok") else f" ({telegram.get('reason', 'unavailable')})"),
            f"WhatsApp: {whatsapp.get('synced', 0)} synced"
            + ("" if whatsapp.get("ok") else f" ({whatsapp.get('reason', 'unavailable')})"),
        ]
        await notice.edit_text("\n".join(lines)[:4000])

    # ------------------------------------------------------------------ #
    # Skills memory (coarse, self-updating knowledge synthesis)
    # ------------------------------------------------------------------ #
    @dp.message(Command("skills"))
    async def _skills(message: Message) -> None:
        if not await guard(message):
            return
        async with session_scope() as session:
            rows = await repo.memory_recent(session, limit=20, kinds=["skill"])
        if not rows:
            await message.answer(
                "\U0001F9E9 No skills synthesized yet.\n\n"
                "Skills build up automatically as the agent learns related "
                "lessons across tasks."
            )
            return
        lines = ["\U0001F9E9 Skills:", ""]
        for row in rows:
            topic = row.key.removeprefix("skill_")
            lines.append(f"\u2022 {topic}: {row.value[:220]}")
        await message.answer("\n".join(lines)[:4000])
