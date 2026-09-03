"""Telegram commands for the capabilities the owner configures once.

Kept apart from setup_commands.py so each file stays readable: this one owns
website logins, email accounts, voice notes and the daily briefing.

Every command that carries a secret deletes the owner's message as soon as it
has been processed, so credentials do not sit in Telegram history.
"""

from __future__ import annotations

import contextlib
from typing import Any

from aiogram import Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)


async def _delete_secret(message: Message) -> None:
    """Remove a message that carried a credential."""
    with contextlib.suppress(Exception):
        await message.delete()


def _fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def register_capability_commands(dp: Dispatcher, guard) -> None:
    """Attach the commands. ``guard`` is the owner-only async check."""

    # ------------------------------------------------------------------ #
    # Websites: log in once, then the agent can act there
    # ------------------------------------------------------------------ #
    @dp.message(Command("site"))
    async def _site(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.integrations.web_session import (
            SiteProfile,
            WebSessionError,
            get_session_manager,
        )

        args = (command.args or "").strip()
        parts = args.split()
        action = parts[0].lower() if parts else ""
        manager = get_session_manager()

        if action in {"", "list"}:
            aliases = await manager.list_profiles()
            if not aliases:
                await message.answer(
                    "\U0001F310 No websites saved yet.\n\n"
                    "Add one:\n"
                    "  /site add <alias> <login_url> <username> <password>\n\n"
                    "Example:\n"
                    "  /site add ipnr https://portal.example.com/login arif mypass\n\n"
                    "Then just ask me:\n"
                    '  "log into ipnr and download this month statement"'
                )
                return
            lines = ["\U0001F310 Saved websites:", ""]
            lines += [f"  \u2022 {alias}" for alias in aliases]
            lines += [
                "",
                "Use:  /site test <alias>   |   /site remove <alias>",
                'Or just ask: "log into <alias> and ..."',
            ]
            await message.answer("\n".join(lines))
            return

        if action == "add":
            # /site add <alias> <login_url> <username> <password...>
            if len(parts) < 5:
                await message.answer(
                    "Usage:\n"
                    "  /site add <alias> <login_url> <username> <password>\n\n"
                    "The password is encrypted immediately and this message is deleted.\n"
                    "I never send it to the AI model."
                )
                return
            alias, login_url, username = parts[1], parts[2], parts[3]
            password = " ".join(parts[4:])

            if not login_url.lower().startswith(("http://", "https://")):
                await _delete_secret(message)
                await message.answer("\u274C The login URL must start with http:// or https://")
                return

            try:
                await manager.save_profile(
                    SiteProfile(
                        alias=alias,
                        login_url=login_url,
                        username=username,
                        password=password,
                    )
                )
            except WebSessionError as exc:
                await _delete_secret(message)
                await message.answer(f"\u274C {str(exc)[:250]}")
                return

            await _delete_secret(message)
            await message.answer(
                f"\u2705 Saved '{alias}'.\n\n"
                "Password encrypted, your message deleted.\n\n"
                f"Try:  /site test {alias}\n"
                f'Or:  "log into {alias} and download the latest invoice"'
            )
            return

        if action in {"test", "login"}:
            if len(parts) < 2:
                await message.answer("Usage: /site test <alias>")
                return
            alias = parts[1]
            if not get_settings().enable_browser_tools:
                await message.answer(
                    "\u26A0\uFE0F Browser tools are off in this build.\n"
                    "Rebuild the image with INSTALL_BROWSER=true and set "
                    "ENABLE_BROWSER_TOOLS=true."
                )
                return
            await message.answer(f"\U0001F510 Logging into '{alias}'...")
            try:
                from app.tools.site_tools import site_login

                result = await site_login(alias=alias)
            except Exception as exc:  # noqa: BLE001 - report, never crash the bot
                await message.answer(f"\u274C Login failed: {str(exc)[:300]}")
                return
            reused = " (reused saved session)" if result.get("reused_session") else ""
            await message.answer(
                f"\u2705 Logged into '{alias}'{reused}\n{result.get('url', '')[:200]}"
            )
            return

        if action in {"remove", "delete", "rm"}:
            if len(parts) < 2:
                await message.answer("Usage: /site remove <alias>")
                return
            removed = await manager.delete_profile(parts[1])
            await message.answer(
                f"\U0001F5D1 Removed '{parts[1]}' and its saved cookies."
                if removed
                else f"No site called '{parts[1]}'."
            )
            return

        await message.answer("Unknown action. Use: /site [list|add|test|remove]")

    # ------------------------------------------------------------------ #
    # Email: several accounts, each with its own app password
    # ------------------------------------------------------------------ #
    @dp.message(Command("email"))
    async def _email(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.tools import email_tools

        args = (command.args or "").strip()
        parts = args.split()
        action = parts[0].lower() if parts else ""

        if action in {"", "list", "accounts"}:
            result = await email_tools.email_accounts()
            accounts = result.get("accounts", [])
            if not accounts:
                await message.answer(
                    "\U0001F4EC No email accounts yet.\n\n"
                    "Gmail needs an App Password (not your normal password):\n"
                    "1. Turn on 2-Step Verification\n"
                    "2. Go to myaccount.google.com/apppasswords\n"
                    "3. Create one, copy the 16 characters\n\n"
                    "Then:\n"
                    "  /email add <alias> <your@gmail.com> <app password>\n\n"
                    "Example:\n"
                    "  /email add work arif@gmail.com abcd efgh ijkl mnop\n\n"
                    "You can add as many accounts as you like."
                )
                return
            lines = ["\U0001F4EC Email accounts:", ""]
            for account in accounts:
                lines.append(f"  \u2022 {account['alias']}  -  {account['address']}")
            lines += [
                "",
                "  /email test <alias>     check it connects",
                "  /email remove <alias>   forget it",
                "",
                'Then ask: "any important email today?"',
            ]
            await message.answer("\n".join(lines))
            return

        if action == "add":
            # /email add <alias> <address> <app password, may contain spaces>
            if len(parts) < 4:
                await message.answer(
                    "Usage:\n  /email add <alias> <address> <app password>\n\n"
                    "Gmail app passwords are 16 characters, spaces are fine."
                )
                return
            alias, address = parts[1], parts[2]
            password = "".join(parts[3:])  # Gmail shows it in groups of four

            if "@" not in address:
                await _delete_secret(message)
                await message.answer("\u274C That does not look like an email address.")
                return

            domain = address.rsplit("@", 1)[1].lower()
            imap_host, smtp_host = _servers_for(domain)

            try:
                await email_tools.save_account(
                    alias,
                    address,
                    password,
                    imap_host=imap_host,
                    smtp_host=smtp_host,
                )
            except Exception as exc:  # noqa: BLE001
                await _delete_secret(message)
                await message.answer(f"\u274C {str(exc)[:250]}")
                return

            await _delete_secret(message)
            await message.answer(
                f"\u2705 Saved '{alias}' ({address}).\n"
                "Password encrypted, your message deleted.\n\n"
                f"Checking the connection..."
            )
            try:
                check = await email_tools.test_account(alias)
            except Exception as exc:  # noqa: BLE001
                await message.answer(f"\u26A0\uFE0F Saved, but the check failed: {str(exc)[:250]}")
                return
            if check.get("ok"):
                await message.answer(
                    f"\u2705 Connected. {check.get('mailboxes', 0)} mailboxes visible.\n\n"
                    'Try asking: "any unread email?"'
                )
            else:
                await message.answer(
                    f"\u26A0\uFE0F Saved, but could not sign in:\n{str(check.get('reason'))[:300]}"
                )
            return

        if action == "test":
            if len(parts) < 2:
                await message.answer("Usage: /email test <alias>")
                return
            try:
                check = await email_tools.test_account(parts[1])
            except Exception as exc:  # noqa: BLE001
                await message.answer(f"\u274C {str(exc)[:300]}")
                return
            await message.answer(
                f"\u2705 {parts[1]}: connected, {check.get('mailboxes', 0)} mailboxes"
                if check.get("ok")
                else f"\u274C {parts[1]}: {str(check.get('reason'))[:300]}"
            )
            return

        if action in {"remove", "delete", "rm"}:
            if len(parts) < 2:
                await message.answer("Usage: /email remove <alias>")
                return
            removed = await email_tools.delete_account(parts[1])
            await message.answer(
                f"\U0001F5D1 Removed '{parts[1]}'."
                if removed
                else f"No account called '{parts[1]}'."
            )
            return

        await message.answer("Unknown action. Use: /email [list|add|test|remove]")

    # ------------------------------------------------------------------ #
    # Telegram account: paste a session string
    # ------------------------------------------------------------------ #
    @dp.message(Command("tgstring"))
    async def _tgstring(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.integrations.telegram_user import UserbotError, get_userbot

        args = (command.args or "").strip()
        if not args:
            await message.answer(
                "\U0001F511 Link your Telegram account with a session string\n\n"
                "Use this when /tglogin cannot receive the code (Telegram often "
                "refuses to send codes to a server).\n\n"
                "On your own computer:\n"
                "  pip install telethon\n"
                "  python -c \"from telethon.sync import TelegramClient; "
                "from telethon.sessions import StringSession; "
                "print(TelegramClient(StringSession(), API_ID, 'API_HASH').start().session.save())\"\n\n"
                "Get API_ID / API_HASH from my.telegram.org.\n\n"
                "Then send:\n"
                "  /tgstring <api_id> <api_hash> <session string>\n"
                "or, if you already ran /tglogin once:\n"
                "  /tgstring <session string>"
            )
            return

        parts = args.split()
        api_id, api_hash, session_string = 0, "", ""
        if len(parts) >= 3 and parts[0].isdigit():
            api_id, api_hash = int(parts[0]), parts[1]
            session_string = parts[2]
        else:
            session_string = parts[0]

        try:
            result = await get_userbot().link_string_session(session_string, api_id, api_hash)
        except UserbotError as exc:
            await _delete_secret(message)
            await message.answer(f"\u274C {str(exc)[:300]}")
            return
        except Exception as exc:  # noqa: BLE001
            await _delete_secret(message)
            await message.answer(f"\u274C Unexpected error: {str(exc)[:300]}")
            return

        await _delete_secret(message)
        name = result.get("name") or result.get("username") or result.get("user_id")
        await message.answer(
            f"\u2705 Telegram account linked: {name}\n\n"
            "I can now read your chats, search messages and send as you.\n"
            "Your message was deleted and the session is encrypted."
        )

    @dp.message(Command("tgexport"))
    async def _tgexport(message: Message, command: CommandObject) -> None:
        """Hand the session string back so the owner can keep a backup."""
        if not await guard(message):
            return
        from app.integrations.telegram_user import UserbotError, get_userbot

        try:
            session_string = await get_userbot().export_session_string()
        except UserbotError as exc:
            await message.answer(f"\u274C {str(exc)[:300]}")
            return
        await message.answer(
            "\U0001F511 Your session string (keep it private, it IS your account):\n\n"
            f"{session_string}\n\n"
            "Anyone holding this can act as you. Delete this message once saved."
        )

    # ------------------------------------------------------------------ #
    # Daily briefing
    # ------------------------------------------------------------------ #
    @dp.message(Command("briefing"))
    async def _briefing(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.agent.briefing import daily_briefing
        from app.db import repo
        from app.db.base import session_scope
        from app.llm import get_llm

        args = (command.args or "").strip().lower()

        if args in {"on", "enable"}:
            settings = get_settings()
            async with session_scope() as session:
                await repo.set_setting(session, "briefing_enabled", "true")
            settings.briefing_enabled = True
            await message.answer(
                f"\u2705 Daily briefing on.\n\n"
                f"Schedule: {settings.briefing_cron} (cron, UTC)\n"
                "Change it with:  /briefing cron 0 6 * * *"
            )
            return

        if args in {"off", "disable"}:
            settings = get_settings()
            async with session_scope() as session:
                await repo.set_setting(session, "briefing_enabled", "false")
            settings.briefing_enabled = False
            await message.answer("\U0001F515 Daily briefing off.")
            return

        if args.startswith("cron "):
            expression = (command.args or "")[5:].strip()
            try:
                from app.scheduler import next_cron

                next_cron(expression)
            except Exception as exc:  # noqa: BLE001
                await message.answer(f"\u274C Invalid cron expression: {str(exc)[:200]}")
                return
            settings = get_settings()
            async with session_scope() as session:
                await repo.set_setting(session, "briefing_cron", expression)
            settings.briefing_cron = expression
            await message.answer(f"\u2705 Briefing schedule: {expression}")
            return

        # No argument: send one right now.
        hours = get_settings().briefing_period_hours
        try:
            text = await daily_briefing(
                chat_id=message.chat.id, period_hours=hours, llm=get_llm()
            )
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"\u274C Could not build the briefing: {str(exc)[:300]}")
            return
        await message.answer(text[:4000])

    # ------------------------------------------------------------------ #
    # Voice notes: speaking is faster than typing
    # ------------------------------------------------------------------ #
    @dp.message(F.voice | F.audio)
    async def _voice(message: Message) -> None:
        if not await guard(message):
            return

        settings = get_settings()
        if not settings.voice_enabled:
            await message.answer("\U0001F507 Voice messages are turned off.")
            return

        from app.integrations.transcription import TranscriptionError, get_transcriber
        from app.security import safe_path

        media = message.voice or message.audio
        if media is None:
            return
        if media.duration and media.duration > 600:
            await message.answer("\u26A0\uFE0F That voice note is over 10 minutes; send a shorter one.")
            return

        notice = await message.answer("\U0001F3A7 Listening...")

        target = safe_path(f"temp/voice_{media.file_unique_id}.ogg")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            file = await message.bot.get_file(media.file_id)
            await message.bot.download_file(file.file_path, destination=str(target))
        except Exception as exc:  # noqa: BLE001
            await notice.edit_text(f"\u274C Could not download the audio: {str(exc)[:200]}")
            return

        try:
            transcript = await get_transcriber().transcribe(target)
        except TranscriptionError as exc:
            await notice.edit_text(f"\u274C {str(exc)[:400]}")
            return
        except Exception as exc:  # noqa: BLE001
            await notice.edit_text(f"\u274C Transcription failed: {str(exc)[:300]}")
            return
        finally:
            with contextlib.suppress(Exception):
                target.unlink()

        text = (transcript.text or "").strip()
        if not text:
            await notice.edit_text("\U0001F914 I could not hear anything in that.")
            return

        await notice.edit_text(f"\U0001F3A4 \"{text[:300]}\"")

        # Route it exactly as if it had been typed, so voice gets chat, tasks
        # and follow-ups without a separate code path.
        from app.agent.conversation import handle_message

        try:
            reply = await handle_message(message.chat.id, message.from_user.id, text)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"\u274C {str(exc)[:300]}")
            return
        if reply.text:
            await message.answer(reply.text[:4000])


def _servers_for(domain: str) -> tuple[str, str]:
    """Best-guess IMAP/SMTP hosts so the owner does not have to know them."""
    known = {
        "gmail.com": ("imap.gmail.com", "smtp.gmail.com"),
        "googlemail.com": ("imap.gmail.com", "smtp.gmail.com"),
        "outlook.com": ("outlook.office365.com", "smtp.office365.com"),
        "hotmail.com": ("outlook.office365.com", "smtp.office365.com"),
        "live.com": ("outlook.office365.com", "smtp.office365.com"),
        "yahoo.com": ("imap.mail.yahoo.com", "smtp.mail.yahoo.com"),
        "icloud.com": ("imap.mail.me.com", "smtp.mail.me.com"),
        "zoho.com": ("imap.zoho.com", "smtp.zoho.com"),
        "yandex.com": ("imap.yandex.com", "smtp.yandex.com"),
    }
    if domain in known:
        return known[domain]
    # A sensible convention for self-hosted domains; the owner can correct it.
    return (f"imap.{domain}", f"smtp.{domain}")
