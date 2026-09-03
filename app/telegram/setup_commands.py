"""Telegram setup commands.

Everything the owner needs to connect accounts and configure models, without
touching the VPS:

    /connect            what is connected, what is missing
    /setkey <p> <key>   store an API key (Claude, ChatGPT, gateway, custom)
    /delkey <p>         remove a stored key
    /addllm ...         register any OpenAI-compatible endpoint
    /rmllm <name>       remove one
    /wa connect|status|logout
    /teams connect <tenant> <client> <secret> [chat]

Security: messages containing secrets are deleted from the chat immediately
after processing, and secret values are never written to logs.
"""

from __future__ import annotations

import contextlib
from typing import Any

from aiogram import F
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

from app.config import get_settings
from app.integrations import BridgeError, teams_bridge, whatsapp_bridge
from app.llm import LLMError, get_manager
from app.logging_conf import get_logger
from app.security.vault import VaultError, get_vault, mask

log = get_logger(__name__)

PROVIDER_ALIASES = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "chatgpt": "openai",
    "gpt": "openai",
    "openai": "openai",
    "omniroute": "omniroute",
    "gateway": "omniroute",
}

SETUP_HELP = """\U0001F511 Connect accounts

AI models (easiest way):
  /llm                           see every provider
  /llm claude                    get the link + steps
  /paste <key>                   paste what you copied - done

  /setkey claude sk-ant-...      (direct, if you already have the key)
  /delkey claude                 remove a key
  /addllm <name> <base_url> <model> [api_key]
        e.g. /addllm openrouter https://openrouter.ai/api/v1 \\
             anthropic/claude-sonnet-4.5 sk-or-xxx
  /rmllm <name>                  remove a custom provider
  /models  /provider  /model     choose which one to use

WhatsApp:
  /wa connect                    show the QR to link your account
  /wa status                     check the link
  /wa logout                     unlink

Teams  (needs a WORK or SCHOOL Microsoft account - personal ones cannot
        use the Teams API; that is Microsoft's restriction):
  /teams login                   sign in with a code + link (recommended)
  /teams login <client_id>       same, using your own app registration
  /teams login common            allow personal accounts to attempt sign-in
  /teams token <access_token>    paste a Graph token (quick, expires ~1h)
  /teams connect <tenant> <client> <secret>   app-only mode
  /teams status

Your Telegram account (so I can act as you + manage your bots):
  /tglogin <api_id> <api_hash> <phone>   from my.telegram.org
  /tgcode <code>                          the code Telegram sends
  /tg2fa <password>                       only if you use 2FA
  /tgstatus  /tglogout  /bots

Memory (I learn on my own; these are just controls):
  /autonomy [balanced|high|paranoid]   how much I do without asking
  /forgetrule <signature>        make me ask about something again
  /learned                       what I picked up
  /teach <key> <fact>            tell me something directly
  /forget <key>                  drop it

Your key messages are deleted from this chat automatically."""


async def _delete_secret_message(message: Message) -> None:
    """Remove a message that contained a credential."""
    with contextlib.suppress(Exception):
        await message.delete()


def _looks_like_guid(value: str) -> bool:
    parts = value.split("-")
    return len(parts) == 5 and len(value) == 36 and all(
        all(c in "0123456789abcdefABCDEF" for c in part) for part in parts
    )


def register_setup_handlers(bot: Any) -> None:
    """Attach the setup commands to an :class:`AgentBot` dispatcher."""
    dp = bot.dp
    guard = bot._guard

    # ------------------------------------------------------------------ #
    # Overview
    # ------------------------------------------------------------------ #
    @dp.message(Command("connect"))
    async def _connect(message: Message) -> None:
        if not await guard(message):
            return
        settings = get_settings()
        manager = get_manager()
        vault = get_vault()

        lines = ["\U0001F517 Connections", ""]

        lines.append("AI models:")
        for provider in manager.configured_providers():
            active = " \u2190 active" if provider.key == manager.active_key() else ""
            state = "\u2705" if provider.configured else "\u26A0\uFE0F needs key"
            lines.append(f"  {state} {provider.key}: {provider.model}{active}")

        lines += ["", "Stored keys:"]
        stored = [n for n in vault.names() if n.endswith("_api_key")]
        if stored:
            for name in stored:
                lines.append(f"  \U0001F511 {name.replace('_api_key', '')}: "
                             f"{mask(vault.get(name))}")
        else:
            lines.append("  (none - the OmniRoute gateway needs no key)")

        lines += ["", "WhatsApp:"]
        if not settings.whatsapp_enabled:
            lines.append("  \u2796 disabled (WHATSAPP_ENABLED=false)")
        else:
            try:
                status = await whatsapp_bridge().status()
                if status.get("logged_in"):
                    lines.append(f"  \u2705 linked as {status.get('jid', 'unknown')}")
                else:
                    lines.append("  \u26A0\uFE0F not linked - use /wa connect")
            except BridgeError as exc:
                lines.append(f"  \u274C bridge down: {str(exc)[:60]}")

        lines += ["", "Teams:"]
        if not settings.teams_enabled:
            lines.append("  \u2796 disabled (TEAMS_ENABLED=false)")
        else:
            try:
                status = await teams_bridge().status()
                if status.get("authenticated"):
                    lines.append(f"  \u2705 tenant {str(status.get('tenant'))[:20]}")
                elif status.get("configured"):
                    lines.append(f"  \u274C auth failing: {str(status.get('error'))[:60]}")
                else:
                    lines.append("  \u26A0\uFE0F not configured - use /teams connect")
            except BridgeError as exc:
                lines.append(f"  \u274C bridge down: {str(exc)[:60]}")

        lines += ["", "Setup help: /setup"]
        await message.answer("\n".join(lines)[:4000])

    @dp.message(Command("setup"))
    async def _setup(message: Message) -> None:
        if not await guard(message):
            return
        await message.answer(SETUP_HELP)

    # ------------------------------------------------------------------ #
    # API keys
    # ------------------------------------------------------------------ #
    @dp.message(Command("setkey"))
    async def _setkey(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        parts = (command.args or "").split(maxsplit=1)
        if len(parts) < 2:
            await message.answer(
                "Usage: /setkey <provider> <api_key>\n"
                "Providers: claude | chatgpt | omniroute | <custom name>"
            )
            return

        raw_provider, key = parts[0].lower(), parts[1].strip()
        provider = PROVIDER_ALIASES.get(raw_provider, raw_provider)
        manager = get_manager()
        known = {p.key for p in manager.configured_providers()} | {"anthropic", "openai", "omniroute"}
        if provider not in known:
            await _delete_secret_message(message)
            await message.answer(
                f"\u274C Unknown provider '{raw_provider}'.\n"
                "Use claude, chatgpt, omniroute, or add one first with /addllm."
            )
            return

        try:
            await get_vault().set(f"{provider}_api_key", key)
        except VaultError as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}")
            return

        # Rebuild the client so the new key is used immediately.
        manager._clients.pop(provider, None)
        await _delete_secret_message(message)

        health = await manager.client(provider).health()
        if health.get("ok"):
            await message.answer(
                f"\u2705 {provider} key saved and verified ({mask(key)}).\n"
                f"Switch with: /provider {raw_provider}"
            )
        else:
            await message.answer(
                f"\u26A0\uFE0F {provider} key saved ({mask(key)}) but the check failed:\n"
                f"{str(health.get('error'))[:200]}"
            )

    @dp.message(Command("delkey"))
    async def _delkey(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        raw = (command.args or "").strip().lower()
        if not raw:
            await message.answer("Usage: /delkey <provider>")
            return
        provider = PROVIDER_ALIASES.get(raw, raw)
        removed = await get_vault().delete(f"{provider}_api_key")
        get_manager()._clients.pop(provider, None)
        await message.answer(
            f"\U0001F5D1 Removed the {provider} key." if removed
            else f"No stored key for {provider}."
        )

    # ------------------------------------------------------------------ #
    # Guided LLM connection: /llm -> pick -> open URL -> /paste <key>
    # ------------------------------------------------------------------ #
    @dp.message(Command("llm"))
    async def _llm(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.llm.presets import catalogue, resolve

        manager = get_manager()
        choice = (command.args or "").strip().lower()

        if not choice:
            active = manager.active_key()
            ready = {p.key for p in manager.configured_providers() if p.configured}
            lines = ["\U0001F9E0 AI models you can connect", ""]
            for preset in catalogue():
                if preset.key == active:
                    mark = "\u2705"
                elif preset.key in ready:
                    mark = "\U0001F7E2"
                elif preset.auth == "none":
                    mark = "\U0001F7E2"
                else:
                    mark = "\u26AA"
                tag = " (free)" if preset.free else ""
                lines.append(f"{mark} {preset.key}{tag} - {preset.label}")
            lines += [
                "",
                "\u2705 active   \U0001F7E2 ready   \u26AA not connected",
                "",
                "Connect one:  /llm claude",
                "Switch:       /provider groq",
            ]
            await message.answer("\n".join(lines)[:4000])
            return

        preset = resolve(choice)
        if preset is None:
            await message.answer(
                f"Unknown provider '{choice}'. Send /llm to see the list."
            )
            return

        if preset.auth == "none":
            try:
                await manager.set_active(preset.key)
            except LLMError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer(
                f"\u2705 {preset.label} needs no key - now active.\n"
                f"Model: {manager.active_model()}"
            )
            return

        bot._pending_key = preset.key
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(preset.steps, 1)) or \
            "1. Open the link\n2. Create an API key\n3. Send:  /paste <key>"
        note = f"\n\n\u2139\uFE0F {preset.notes}" if preset.notes else ""
        await message.answer(
            f"\U0001F517 Connect {preset.label}\n\n"
            f"{preset.signup_url}\n\n"
            f"{steps}{note}\n\n"
            f"Default model: {preset.default_model}\n"
            "Your message with the key is deleted automatically."
        )

    @dp.message(Command("paste"))
    async def _paste(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        value = (command.args or "").strip()
        pending = getattr(bot, "_pending_key", "")

        if not pending:
            await _delete_secret_message(message)
            await message.answer(
                "Nothing is waiting for a key. Start with /llm <provider>."
            )
            return
        if not value:
            await message.answer(f"Usage: /paste <your {pending} key>")
            return

        try:
            result = await get_manager().connect_preset(pending, value)
        except (LLMError, VaultError) as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}\n\nTry /paste again with the full key.")
            return

        await _delete_secret_message(message)
        bot._pending_key = ""

        if result["verified"]:
            try:
                await get_manager().set_active(result["provider"])
            except LLMError:
                pass
            await message.answer(
                f"\u2705 {result['label']} connected and verified.\n"
                f"Model: {get_manager().active_model()}\n\n"
                "It is now the active model. Send me any task."
            )
        else:
            await message.answer(
                f"\u26A0\uFE0F Key saved for {result['label']}, but the check failed:\n"
                f"{result['error']}\n\n"
                f"It is stored - switch with /provider {result['provider']} once fixed."
            )

    # ------------------------------------------------------------------ #
    # Custom providers
    # ------------------------------------------------------------------ #
    @dp.message(Command("addllm"))
    async def _addllm(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        parts = (command.args or "").split()
        if len(parts) < 3:
            await message.answer(
                "Usage: /addllm <name> <base_url> <model> [api_key]\n\n"
                "Example:\n"
                "/addllm openrouter https://openrouter.ai/api/v1 "
                "anthropic/claude-sonnet-4.5 sk-or-xxx"
            )
            return

        name, base_url, model = parts[0], parts[1], parts[2]
        api_key = parts[3] if len(parts) > 3 else ""
        try:
            await get_manager().add_custom_provider(name, base_url, model, api_key)
        except (LLMError, VaultError) as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}")
            return

        await _delete_secret_message(message)
        health = await get_manager().client(name.lower()).health()
        state = "verified" if health.get("ok") else f"unverified ({str(health.get('error'))[:80]})"
        await message.answer(
            f"\u2705 Added provider '{name.lower()}' - {state}\n"
            f"Model: {model}\n\nUse it with: /provider {name.lower()}"
        )

    @dp.message(Command("rmllm"))
    async def _rmllm(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        name = (command.args or "").strip().lower()
        if not name:
            await message.answer("Usage: /rmllm <name>")
            return
        removed = await get_manager().remove_custom_provider(name)
        await message.answer(
            f"\U0001F5D1 Removed provider '{name}'." if removed
            else f"No custom provider named '{name}'."
        )

    # ------------------------------------------------------------------ #
    # Autonomy: how much the agent does without asking
    # ------------------------------------------------------------------ #
    @dp.message(Command("autonomy"))
    async def _autonomy(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.config import get_settings
        from app.db import repo
        from app.db.base import session_scope

        level = (command.args or "").strip().lower()
        valid = {"balanced", "high", "paranoid"}

        if not level:
            current = get_settings().autonomy_level
            async with session_scope() as session:
                patterns = await repo.list_approval_patterns(session, limit=10)
            lines = [
                f"\U0001F9E0 Autonomy: {current}",
                "",
                "balanced  ask only about irreversible, unrequested actions",
                "high      act without confirmation (secrets stay protected)",
                "paranoid  confirm every risky action",
                "",
                "Change with:  /autonomy high",
            ]
            if patterns:
                lines += ["", "Learned from your decisions:"]
                for row in patterns:
                    mark = "\u2705" if row.approved_count >= 3 else "\u2022"
                    lines.append(
                        f"{mark} {row.signature}  (+{row.approved_count}/-{row.rejected_count})"
                    )
                lines += ["", "Forget one with:  /forgetrule <signature>"]
            await message.answer("\n".join(lines))
            return

        if level not in valid:
            await message.answer(f"Unknown level. Choose: {', '.join(sorted(valid))}")
            return

        async with session_scope() as session:
            await repo.set_setting(session, "autonomy_level", level)
        get_settings().autonomy_level = level
        explain = {
            "balanced": "I will ask only before irreversible actions you did not request.",
            "high": "I will act on my own. Credentials and task state stay protected.",
            "paranoid": "I will confirm every risky action with you first.",
        }[level]
        await message.answer(f"\u2705 Autonomy: {level}\n\n{explain}")

    @dp.message(Command("forgetrule"))
    async def _forgetrule(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.db import repo
        from app.db.base import session_scope

        sig = (command.args or "").strip()
        if not sig:
            usage = "Usage: /forgetrule <signature>\nSee them with /autonomy"
            await message.answer(usage)
            return
        async with session_scope() as session:
            removed = await repo.forget_approval_pattern(session, sig)
        confirmed = f"\u2705 Forgotten: {sig}\nI will ask about it again."
        await message.answer(confirmed if removed else f"No such rule: {sig}")

    # ------------------------------------------------------------------ #
    # What the agent has taught itself
    # ------------------------------------------------------------------ #
    @dp.message(Command("learned"))
    async def _learned(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.db import repo
        from app.db.base import session_scope

        query = (command.args or "").strip()
        async with session_scope() as session:
            rows = (
                await repo.memory_search(session, query, limit=25)
                if query
                else await repo.memory_recent(session, limit=25)
            )
        if not rows:
            await message.answer(
                "Nothing learned yet. I pick things up automatically as I do tasks."
            )
            return

        icons = {"preference": "\u2B50", "gotcha": "\u26A0\uFE0F",
                 "workflow": "\U0001F501", "fact": "\U0001F4A1"}
        bullet = "\u2022"
        lines = ["\U0001F9E0 What I have learned:", ""]
        for row in rows:
            lines.append(f"{icons.get(row.kind, bullet)} {row.key}")
            lines.append(f"    {row.value[:180]}")
        lines += ["", "Remove one with: /forget <key>"]
        await message.answer("\n".join(lines)[:4000])

    @dp.message(Command("forget"))
    async def _forget(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.db import repo
        from app.db.base import session_scope

        key = (command.args or "").strip()
        if not key:
            await message.answer("Usage: /forget <key>   (see /learned)")
            return
        async with session_scope() as session:
            removed = await repo.memory_delete(session, key)
        await message.answer(
            f"\U0001F5D1 Forgot '{key}'." if removed else f"I have nothing stored as '{key}'."
        )

    @dp.message(Command("teach"))
    async def _teach(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.db import repo
        from app.db.base import session_scope
        from app.tools.memory_tools import looks_like_secret

        raw = (command.args or "").strip()
        if not raw:
            await message.answer(
                "Teach me something durable:\n"
                "/teach <key> <what to remember>\n\n"
                "Example: /teach report_format owner wants PDF, not DOCX"
            )
            return
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            await message.answer("Usage: /teach <key> <what to remember>")
            return
        key, value = parts[0], parts[1]
        if looks_like_secret(raw):
            await _delete_secret_message(message)
            await message.answer(
                "\u274C That looks like a credential. Use /setkey for secrets."
            )
            return
        async with session_scope() as session:
            await repo.memory_store(
                session, key=key, value=value, kind="preference", tags=["taught"]
            )
        await message.answer(f"\u2705 Learned: {key}")

    # ------------------------------------------------------------------ #
    # WhatsApp account
    # ------------------------------------------------------------------ #
    @dp.message(Command("wa"))
    async def _wa(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        args = (command.args or "").strip()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "status"

        if not get_settings().whatsapp_enabled:
            await message.answer("WhatsApp is disabled. Set WHATSAPP_ENABLED=true and restart.")
            return

        bridge = whatsapp_bridge()

        if action in {"connect", "login", "qr"}:
            try:
                status = await bridge.status()
                if status.get("logged_in"):
                    await message.answer(
                        f"\u2705 Already linked as {status.get('jid', 'unknown')}"
                    )
                    return
                png = await bridge.login_qr_png()
            except BridgeError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer_photo(
                BufferedInputFile(png, filename="whatsapp-qr.png"),
                caption=(
                    "\U0001F4F1 Link WhatsApp\n\n"
                    "1. WhatsApp \u2192 Settings \u2192 Linked devices\n"
                    "2. Tap 'Link a device'\n"
                    "3. Scan this code (valid ~60s)\n\n"
                    "Then check with /wa status"
                ),
            )
            return

        if action in {"logout", "unlink", "disconnect"}:
            try:
                await bridge.logout()
            except BridgeError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer("\U0001F513 WhatsApp unlinked.")
            return

        if action == "send":
            send_parts = (parts[1] if len(parts) > 1 else "").split(maxsplit=1)
            if len(send_parts) < 2:
                await message.answer("Usage: /wa send <number> <text>")
                return
            try:
                result = await bridge.send_text(send_parts[0], send_parts[1])
            except BridgeError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer(f"\u2705 Sent (id {result.get('message_id', '?')})")
            return

        try:
            status = await bridge.status()
        except BridgeError as exc:
            await message.answer(f"\u274C WhatsApp bridge unreachable: {exc}")
            return
        if status.get("logged_in"):
            await message.answer(f"\u2705 Linked as {status.get('jid', 'unknown')}")
        else:
            await message.answer("\u26A0\uFE0F Not linked. Use /wa connect")

    # ------------------------------------------------------------------ #
    # Telegram ACCOUNT (userbot) - lets the agent act as the owner and
    # manage the owner's other bots through BotFather.
    # ------------------------------------------------------------------ #
    @dp.message(Command("tglogin"))
    async def _tglogin(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.integrations import UserbotError, get_userbot

        parts = (command.args or "").split()
        if len(parts) < 3:
            await message.answer(
                "Link your Telegram ACCOUNT (so I can read your chats and manage your bots):\n\n"
                "1. Open https://my.telegram.org \u2192 API development tools\n"
                "2. Create an app, copy api_id and api_hash\n"
                "3. Send:\n"
                "   /tglogin <api_id> <api_hash> <phone>\n\n"
                "Example:\n/tglogin 1234567 abcdef0123456789abcdef0123456789 +8801712345678"
            )
            return

        api_id, api_hash, phone = parts[0], parts[1], parts[2]
        if not api_id.isdigit():
            await _delete_secret_message(message)
            await message.answer("\u274C api_id must be a number.")
            return

        userbot = get_userbot()
        if not userbot.available():
            await message.answer(
                "\u274C The 'telethon' package is missing from this build."
            )
            return
        try:
            await userbot.start_login(int(api_id), api_hash, phone)
        except UserbotError as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}")
            return

        await _delete_secret_message(message)
        await message.answer(
            "\U0001F4F2 Telegram sent a login code to your account.\n\n"
            "Send it as:  /tgcode 12345\n"
            "(If you have 2FA, I will ask for the password next.)"
        )

    @dp.message(Command("tgcode"))
    async def _tgcode(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.integrations import TwoFactorRequired, UserbotError, get_userbot

        code = (command.args or "").strip()
        if not code:
            await message.answer("Usage: /tgcode <code>")
            return
        try:
            result = await get_userbot().submit_code(code)
        except TwoFactorRequired:
            await _delete_secret_message(message)
            await message.answer(
                "\U0001F510 This account has 2FA.\nSend:  /tg2fa <your password>"
            )
            return
        except UserbotError as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}")
            return

        await _delete_secret_message(message)
        await message.answer(
            f"\u2705 Telegram account linked: {result.get('name') or result.get('user_id')}\n"
            f"@{result.get('username') or 'no username'}\n\n"
            "I can now read your chats, message people as you, and manage your bots.\n"
            "Try: /bots"
        )

    @dp.message(Command("tg2fa"))
    async def _tg2fa(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        from app.integrations import UserbotError, get_userbot

        password = (command.args or "").strip()
        if not password:
            await message.answer("Usage: /tg2fa <password>")
            return
        try:
            result = await get_userbot().submit_password(password)
        except UserbotError as exc:
            await _delete_secret_message(message)
            await message.answer(f"\u274C {exc}")
            return
        await _delete_secret_message(message)
        await message.answer(
            f"\u2705 Telegram account linked: {result.get('name') or result.get('user_id')}"
        )

    @dp.message(Command("tgstatus"))
    async def _tgstatus(message: Message) -> None:
        if not await guard(message):
            return
        from app.integrations import get_userbot

        status = await get_userbot().status()
        if not status.get("available"):
            await message.answer("\u274C telethon is not installed in this build.")
            return
        if status.get("linked"):
            await message.answer(
                f"\u2705 Linked as {status.get('name')} "
                f"(@{status.get('username') or 'none'}, id {status.get('user_id')})"
            )
        else:
            reason = status.get("error", "")
            await message.answer(
                "\u26A0\uFE0F Telegram account not linked. Use /tglogin"
                + (f"\n{reason[:200]}" if reason else "")
            )

    @dp.message(Command("tglogout"))
    async def _tglogout(message: Message) -> None:
        if not await guard(message):
            return
        from app.integrations import get_userbot

        await get_userbot().logout()
        await message.answer("\U0001F513 Telegram account unlinked.")

    @dp.message(Command("bots"))
    async def _bots(message: Message) -> None:
        if not await guard(message):
            return
        from app.integrations import UserbotError, get_userbot

        await message.answer("\u23F3 Asking BotFather...")
        try:
            replies = await get_userbot().list_bots()
        except UserbotError as exc:
            await message.answer(f"\u274C {exc}")
            return
        text = "\n\n".join(replies)[:3500] or "BotFather did not reply."
        await message.answer(f"\U0001F916 Your bots:\n\n{text}")

    # ------------------------------------------------------------------ #
    # Teams account
    # ------------------------------------------------------------------ #
    @dp.message(Command("teams"))
    async def _teams(message: Message, command: CommandObject) -> None:
        if not await guard(message):
            return
        args = (command.args or "").strip()
        parts = args.split()
        action = parts[0].lower() if parts else "status"

        if not get_settings().teams_enabled:
            await message.answer("Teams is disabled. Set TEAMS_ENABLED=true and restart.")
            return

        bridge = teams_bridge()

        if action in {"login", "signin", "device"}:
            # /teams login [client_id] [tenant]  - order-independent.
            # A GUID is a client id; anything else is a tenant.
            client_id, tenant = "", ""
            tenant_aliases = {
                "common": "common",          # work/school AND personal
                "personal": "consumers",
                "consumers": "consumers",
                "work": "organizations",
                "school": "organizations",
                "organizations": "organizations",
            }
            for token in parts[1:]:
                low = token.lower()
                if _looks_like_guid(token):
                    client_id = token
                elif low in tenant_aliases:
                    tenant = tenant_aliases[low]
                elif "." in token:               # e.g. contoso.onmicrosoft.com
                    tenant = token
            try:
                started = await bridge.start_device_login(client_id, tenant)
            except BridgeError as exc:
                detail = str(exc)
                if "65002" in detail or "preauthorization" in detail.lower():
                    await message.answer(
                        "\u274C Your tenant will not let this app sign in.\n\n"
                        "Fix it with your own app registration (5 minutes):\n"
                        "1. portal.azure.com \u2192 App registrations \u2192 New\n"
                        "2. Accounts: 'Any organizational directory'\n"
                        "3. Authentication \u2192 Add platform \u2192 Mobile/desktop\n"
                        "   \u2192 tick 'Allow public client flows' = Yes\n"
                        "4. Copy the Application (client) ID, then run:\n"
                        "   /teams login <client_id>\n\n"
                        "Or paste a token instead:  /teams token <access_token>"
                    )
                    return
                await message.answer(f"\u274C Could not start sign-in: {detail[:250]}")
                return

            code = started.get("user_code", "?")
            url = started.get("verification_url", "https://microsoft.com/devicelogin")
            interval = max(3, int(started.get("interval", 5)))
            expires = int(started.get("expires_in", 900))

            await message.answer(
                "\U0001F510 Sign in to Microsoft Teams\n\n"
                f"1. Open:  {url}\n"
                f"2. Enter this code:\n\n      {code}\n\n"
                "3. Sign in with your WORK or SCHOOL account\n\n"
                "\u26A0\uFE0F A personal account (outlook.com, hotmail, live) will "
                "not work - Microsoft does not expose Teams to personal accounts.\n\n"
                "I am waiting - I will tell you as soon as it is done."
            )

            # Poll in the background so the bot stays responsive.
            async def _wait() -> None:
                import asyncio

                deadline = min(expires, 900)
                waited = 0
                while waited < deadline:
                    await asyncio.sleep(interval)
                    waited += interval
                    try:
                        state = await bridge.poll_device_login()
                    except BridgeError as exc:
                        await message.answer(f"\u274C Sign-in check failed: {str(exc)[:200]}")
                        return
                    if state.get("done"):
                        account = state.get("account") or "your account"
                        await message.answer(
                            f"\u2705 Teams connected as {account}\n\n"
                            "I can now read and send your Teams messages.\n"
                            "Try:  /teams status"
                        )
                        return
                    if not state.get("pending", True):
                        await message.answer(
                            f"\u274C Sign-in stopped: {str(state.get('error'))[:200]}"
                        )
                        return
                await message.answer(
                    "\u23F0 The sign-in code expired. Run /teams login again."
                )

            import asyncio

            asyncio.create_task(_wait())
            return

        if action == "connect":
            if len(parts) < 4:
                await message.answer(
                    "Usage:\n/teams connect <tenant_id> <client_id> <client_secret> "
                    "[default_chat]\n\n"
                    "Get these from Azure Portal \u2192 App registrations.\n"
                    "Needs application permissions with admin consent:\n"
                    "  Chat.ReadWrite.All, ChannelMessage.Send, ChannelMessage.Read.All"
                )
                return
            tenant, client_id, secret = parts[1], parts[2], parts[3]
            default_chat = parts[4] if len(parts) > 4 else ""
            try:
                await bridge.configure(tenant, client_id, secret, default_chat)
            except BridgeError as exc:
                await _delete_secret_message(message)
                await message.answer(f"\u274C Teams rejected the credentials:\n{str(exc)[:300]}")
                return

            vault = get_vault()
            with contextlib.suppress(VaultError):
                await vault.set("teams_tenant_id", tenant)
                await vault.set("teams_client_id", client_id)
                await vault.set("teams_client_secret", secret)

            await _delete_secret_message(message)
            await message.answer(
                f"\u2705 Teams connected (tenant {tenant[:8]}...).\n"
                "Test it with: /teams send <chat_id> hello"
            )
            return

        if action == "token":
            if len(parts) < 2:
                await message.answer(
                    "\U0001F511 Paste a Microsoft Graph access token\n\n"
                    "1. Open  https://developer.microsoft.com/graph/graph-explorer\n"
                    "2. Sign in with your work account\n"
                    "3. 'Access token' tab \u2192 copy the token\n"
                    "4. Send:  /teams token <paste it here>\n\n"
                    "Note: these expire in about an hour. For a permanent link "
                    "use /teams login with your own client id."
                )
                return
            token = args.split(maxsplit=1)[1].strip()
            try:
                result = await bridge.login_with_token(token)
            except BridgeError as exc:
                await _delete_secret_message(message)
                await message.answer(f"\u274C {str(exc)[:250]}")
                return
            await _delete_secret_message(message)
            await message.answer(
                f"\u2705 Teams connected as {result.get('account', 'your account')}\n\n"
                "\u26A0\uFE0F This token expires in about an hour. "
                "Re-paste when it does, or set up /teams login for a lasting link."
            )
            return

        if action in {"disconnect", "logout"}:
            try:
                await bridge.disconnect()
            except BridgeError as exc:
                await message.answer(f"\u274C {exc}")
                return
            for key in ("teams_tenant_id", "teams_client_id", "teams_client_secret"):
                await get_vault().delete(key)
            await message.answer("\U0001F513 Teams disconnected.")
            return

        if action == "send":
            if len(parts) < 3:
                await message.answer("Usage: /teams send <chat_id> <text>")
                return
            chat = parts[1]
            text = args.split(maxsplit=2)[2]
            try:
                result = await bridge.send_text(chat, text)
            except BridgeError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer(f"\u2705 Sent (id {result.get('message_id', '?')})")
            return

        try:
            status = await bridge.status()
        except BridgeError as exc:
            await message.answer(f"\u274C Teams bridge unreachable: {exc}")
            return
        if status.get("authenticated"):
            mode = status.get("mode", "")
            who = status.get("account") or f"tenant {status.get('tenant')}"
            label = "signed in as" if mode == "delegated" else "app mode,"
            await message.answer(f"\u2705 Teams connected - {label} {who}")
        elif status.get("configured"):
            await message.answer(f"\u274C Configured but auth failed:\n{status.get('error')}")
        else:
            await message.answer(
                "\u26A0\uFE0F Teams not connected.\n\n"
                "\u26A0\uFE0F Teams needs a WORK or SCHOOL Microsoft account. "
                "Personal accounts (outlook.com, hotmail, live) cannot use the "
                "Teams API at all - that is Microsoft's rule, not mine.\n\n"
                "1. /teams login\n"
                "   sign in with a code (best - stays connected)\n\n"
                "2. /teams token <access_token>\n"
                "   paste a token from Graph Explorer (quick, ~1 hour)\n\n"
                "3. /teams connect <tenant> <client> <secret>\n"
                "   app-only mode with an Azure app registration\n\n"
                "Tenant blocked the sign-in? Register your own app and run:\n"
                "  /teams login <client_id>"
            )
