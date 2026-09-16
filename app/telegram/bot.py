"""Telegram bot (aiogram v3, long polling).

Only allowlisted numeric user IDs can interact - every other update is dropped
before any handler runs.  Natural language creates a background Task; commands
inspect and control state.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message, Update

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, ApprovalStatus, TaskStatus
from app.logging_conf import get_logger
from app.telegram import otp_panel
from app.telegram.notifier import Notifier

log = get_logger(__name__)

HELP_TEXT = """\U0001F916 Personal AI Agent

Just send me an instruction in plain language, e.g.
"Download https://example.com/report.pdf and send it to me when it's ready."

Just talk to me normally. I work out whether you are chatting or giving me a
job, and follow-ups stay on the same job.

Commands:
/start   - check that the agent is alive
/help    - this message
/new     - start a fresh conversation
/history - list/switch past conversations
/mode    - auto | chat | task
/session - what this conversation is about
/status  - system status (workers, db, llm, resources)
/tasks   - recent tasks
/task <id>   - details of one task
/cancel <id> - cancel a task
/reply <id> <text> - answer an agent question
/approve <id> / /reject <id> - decide a HIGH_RISK request
/jobs    - scheduled jobs
/memory <query> - search long-term memory
/otpchat - open the dedicated OTP-bot thread (send number files there)
/otpbot  - status of the OTP-number automation (just say "start"/"stop" to run it)

AI model:
/models  - list providers and models
/model <name>    - switch model (e.g. /model gpt-4o)
/provider <name> - switch: omniroute | claude | chatgpt | local

Accounts & setup:
/connect - what is connected right now
/setup   - how to connect everything
/llm     - list AI models; /llm claude to connect one
/paste <key> - paste the key you just copied
/addllm <name> <url> <model> [key] - add any OpenAI-compatible LLM

Messaging:
/wa connect | phone <num> | status | logout | send <num> <text>
/teams connect <tenant> <client> <secret> | status | send <chat> <text>
/inbox   - recent WhatsApp/Teams messages

Your Telegram account (acts as YOU, not the bot - message any chat, run other bots):
/tglogin <api_id> <api_hash> <phone> - link by phone code (from my.telegram.org)
/tgstring <session string>          - link by pasting an existing Telethon or
                                       Pyrogram session string instead (skips
                                       the code - use if /tglogin can't deliver one)
/tgstatus - is it linked   /tglogout - unlink

Files: just send me a document (e.g. a .txt of numbers) and tell me what to do
with it, e.g. "add these numbers to @SomeBot" - I will send it through your
own Telegram account and can repeat that on a schedule (/jobs).

Contacts & skills:
/members [query] - people seen on Telegram/WhatsApp
/syncmembers - pull in contacts now
/skills  - what the agent has taught itself (grows on its own as I learn)
"""


class AgentBot:
    def __init__(self, notifier: Notifier | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.notifier = notifier or Notifier()
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self._task: asyncio.Task | None = None
        self._rate: dict[int, list[float]] = {}
        # provider name waiting for /paste <key>
        self._pending_key: str = ""
        self._register()

    # ------------------------------------------------------------------ #
    # Access control
    # ------------------------------------------------------------------ #
    def _authorized(self, message: Message) -> bool:
        user = message.from_user
        if user is None:
            return False
        return user.id in self.settings.allowed_user_ids

    def _rate_ok(self, user_id: int) -> bool:
        now = time.monotonic()
        window = [t for t in self._rate.get(user_id, []) if now - t < 60]
        if len(window) >= self.settings.telegram_rate_limit_per_min:
            self._rate[user_id] = window
            return False
        window.append(now)
        self._rate[user_id] = window
        return True

    async def _send_next_prompt(self, query: Any, reply: str) -> None:
        """Send a post-tagging reply, with buttons when it is another question.

        _continue_after_tagging either asks about the next country or reports
        that everything started. Only the question needs a keyboard, and it
        has to carry the id of whichever entry is now pending - re-reading it
        here keeps the buttons and the prompt describing the same file.
        """
        from app.automation import otp_bot

        pending = await otp_bot.get_awaiting_tag_entry()
        markup = otp_panel.service_keyboard(pending["id"]) if pending else None
        with contextlib.suppress(Exception):
            await query.message.answer(reply, reply_markup=markup)

    async def _guard(self, message: Message) -> bool:
        if not self._authorized(message):
            log.warning(
                "unauthorized_telegram_user",
                extra={"user_id": getattr(message.from_user, "id", None)},
            )
            with contextlib.suppress(Exception):
                await message.answer("\U0001F6AB Not authorized.")
            return False
        if not self._rate_ok(message.from_user.id):
            with contextlib.suppress(Exception):
                await message.answer("\u23F3 Slow down a moment - rate limit reached.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    def _register(self) -> None:
        dp = self.dp

        @dp.message(Command("start"))
        async def _start(message: Message) -> None:
            if not await self._guard(message):
                return
            await message.answer(
                "\u2705 Agent online.\n\n"
                f"Model: {self.settings.llm_model}\n"
                f"Workspace: {self.settings.workspace_dir}\n\n"
                "Send an instruction, or /help."
            )

        @dp.message(Command("help"))
        async def _help(message: Message) -> None:
            if not await self._guard(message):
                return
            await message.answer(HELP_TEXT)

        @dp.message(Command("status"))
        async def _status(message: Message) -> None:
            if not await self._guard(message):
                return
            from app.monitoring import health_snapshot

            snapshot = await health_snapshot()
            db = snapshot["checks"]["database"]
            llm = snapshot["checks"]["llm"]
            res = snapshot["resources"]
            stats = snapshot["stats"]
            counts = ", ".join(f"{k}: {v}" for k, v in sorted(stats["tasks_by_status"].items())) or "none"
            overall = "\u2705 healthy" if snapshot["ok"] else "\u26A0\uFE0F degraded"
            await message.answer(
                f"\U0001F4CA Status: {overall}\n\n"
                f"Uptime: {snapshot['uptime_s'] // 60} min\n"
                f"Database: {'ok' if db['ok'] else 'FAIL'}\n"
                f"LLM: {'ok' if llm['ok'] else 'FAIL'} ({llm.get('model', '?')})\n"
                f"Workers: {snapshot['workers']['running']}\n\n"
                f"Tasks -> {counts}\n"
                f"Active: {stats['active_tasks']} | Approvals: {stats['pending_approvals']} | "
                f"Jobs: {stats['enabled_jobs']}\n\n"
                f"CPU {res['cpu_percent']}% | RAM {res['memory_percent']}% | Disk {res['disk_percent']}%"
            )

        @dp.message(Command("tasks"))
        async def _tasks(message: Message) -> None:
            if not await self._guard(message):
                return
            async with session_scope() as session:
                rows = await repo.list_tasks(session, limit=10)
            if not rows:
                await message.answer("No tasks yet.")
                return
            icons = {
                "PENDING": "\u23F3", "RUNNING": "\u25B6\uFE0F", "WAITING": "\u23F8",
                "WAITING_FOR_USER": "\u2753", "WAITING_FOR_EXTERNAL_EVENT": "\u23F8",
                "COMPLETED": "\u2705", "FAILED": "\u274C", "CANCELLED": "\U0001F6D1",
            }
            lines = ["\U0001F4CB Recent tasks:", ""]
            for task in rows:
                icon = icons.get(task.status, "\u2022")
                lines.append(f"{icon} {task.id} - {task.title[:48]}")
            lines += ["", "Details: /task <id>"]
            await message.answer("\n".join(lines))

        @dp.message(Command("task"))
        async def _task(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            task_id = (command.args or "").strip()
            if not task_id:
                await message.answer("Usage: /task <id>")
                return
            async with session_scope() as session:
                task = await repo.get_task(session, task_id) or await repo.find_task_by_prefix(
                    session, task_id
                )
                if task is None:
                    await message.answer(f"Task not found: {task_id}")
                    return
                calls = await repo.list_tool_calls(session, task.id, limit=10)
                payload = {
                    "id": task.id, "status": task.status, "title": task.title,
                    "step": task.current_step, "max_steps": task.max_steps,
                    "result": task.result, "error": task.error,
                    "files": list(task.output_files or []),
                }
            lines = [
                f"Task {payload['id']}",
                f"Status: {payload['status']}  (step {payload['step']}/{payload['max_steps']})",
                f"Request: {payload['title'][:300]}",
            ]
            if calls:
                lines += ["", "Tool calls:"]
                lines += [f"  {c.step}. {c.tool} -> {c.status}" for c in calls]
            if payload["result"]:
                lines += ["", f"Result: {payload['result'][:1200]}"]
            if payload["error"]:
                lines += ["", f"Error: {payload['error'][:800]}"]
            if payload["files"]:
                lines += ["", "Files: " + ", ".join(payload["files"][:10])]
            await message.answer("\n".join(lines)[:4000])

        @dp.message(Command("cancel"))
        async def _cancel(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            task_id = (command.args or "").strip()
            if not task_id:
                await message.answer("Usage: /cancel <id>")
                return
            async with session_scope() as session:
                task = await repo.get_task(session, task_id) or await repo.find_task_by_prefix(
                    session, task_id
                )
                if task is None:
                    await message.answer(f"Task not found: {task_id}")
                    return
                if task.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                    await message.answer(f"Task {task.id} already {task.status}.")
                    return
                await repo.set_task_status(
                    session, task.id, TaskStatus.CANCELLED, error="cancelled by owner"
                )
                await repo.log_event(session, "task_cancelled", task_id=task.id,
                                     data={"by": message.from_user.id})
            await message.answer(f"\U0001F6D1 Cancelled {task.id}.")

        @dp.message(Command("approve"))
        async def _approve(message: Message, command: CommandObject) -> None:
            await self._decide(message, command, approved=True)

        @dp.message(Command("reject"))
        async def _reject(message: Message, command: CommandObject) -> None:
            await self._decide(message, command, approved=False)

        @dp.message(Command("reply"))
        async def _reply(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            parts = (command.args or "").split(maxsplit=1)
            if len(parts) < 2:
                await message.answer("Usage: /reply <task_id> <your answer>")
                return
            task_id, answer = parts[0], parts[1]
            async with session_scope() as session:
                task = await repo.get_task(session, task_id) or await repo.find_task_by_prefix(
                    session, task_id
                )
                if task is None:
                    await message.answer(f"Task not found: {task_id}")
                    return
                context = dict(task.context or {})
                context.setdefault("user_replies", []).append(answer)

                answer_text = answer
                chat_row = await repo.ensure_session(session, message.chat.id)
                chat_ctx = dict(chat_row.context or {})
                pending_upload = chat_ctx.get("pending_upload")
                if pending_upload and pending_upload.get("path"):
                    answer_text = (
                        f"{answer}\n\n"
                        f"[Attached file: {pending_upload['path']} "
                        f"(uploaded as \"{pending_upload.get('name', '')}\")]"
                    )
                    chat_ctx.pop("pending_upload", None)
                    await repo.update_session(session, message.chat.id, context=chat_ctx)

                await repo.update_task(
                    session,
                    task.id,
                    context=context,
                    user_request=f"{task.user_request}\n\n[owner reply] {answer_text}",
                    status=TaskStatus.PENDING.value,
                    run_after=None,
                )
                await repo.log_event(session, "user_reply", task_id=task.id,
                                     data={"answer": answer[:500]})
            await message.answer(f"\u25B6\uFE0F Resuming {task.id} with your answer.")

        @dp.message(Command("jobs"))
        async def _jobs(message: Message) -> None:
            if not await self._guard(message):
                return
            async with session_scope() as session:
                jobs = await repo.list_jobs(session, limit=20)
            if not jobs:
                await message.answer("No scheduled jobs.")
                return
            lines = ["\U0001F4C5 Scheduled jobs:", ""]
            for job in jobs:
                when = job.next_run_at.strftime("%Y-%m-%d %H:%M UTC") if job.next_run_at else "-"
                state = "on" if job.enabled else "off"
                lines.append(f"{job.id} [{state}] {job.kind} @ {when} - {job.name[:40]}")
            await message.answer("\n".join(lines)[:4000])

        @dp.message(Command("memory"))
        async def _memory(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            query = (command.args or "").strip()
            async with session_scope() as session:
                rows = (
                    await repo.memory_search(session, query, limit=10)
                    if query
                    else await repo.memory_recent(session, limit=10)
                )
            if not rows:
                await message.answer("Nothing in memory.")
                return
            await message.answer(
                "\U0001F9E0 Memory:\n\n"
                + "\n".join(f"- {r.key}: {r.value[:200]}" for r in rows)[:3800]
            )

        @dp.message(Command("models"))
        async def _models(message: Message) -> None:
            if not await self._guard(message):
                return
            from app.llm import KNOWN_MODELS, get_manager

            manager = get_manager()
            active = manager.active_key()
            lines = ["\U0001F9E0 AI providers:", ""]
            for provider in manager.configured_providers():
                mark = "\u2705" if provider.key == active else ("\u2022" if provider.configured else "\u26A0\uFE0F")
                state = "active" if provider.key == active else (
                    "ready" if provider.configured else "no API key"
                )
                lines.append(f"{mark} {provider.key} - {provider.label}")
                lines.append(f"    model: {provider.model}  ({state})")
            suggestions = KNOWN_MODELS.get(active, [])
            if suggestions:
                lines += ["", f"Popular {active} models:"]
                lines += [f"  {name}" for name in suggestions]
            lines += ["", "Switch: /provider claude   |   /model gpt-4o"]
            await message.answer("\n".join(lines)[:4000])

        @dp.message(Command("provider"))
        async def _provider(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            from app.llm import LLMError, get_manager

            raw = (command.args or "").strip().lower()
            if not raw:
                await message.answer(
                    "Usage: /provider omniroute | claude | chatgpt | local"
                )
                return
            alias = {
                "claude": "anthropic", "anthropic": "anthropic",
                "chatgpt": "openai", "gpt": "openai", "openai": "openai",
                "local": "ollama", "ollama": "ollama", "offline": "echo",
                "omniroute": "omniroute", "gateway": "omniroute",
                "free": "omniroute", "omni": "omniroute",
            }
            parts = raw.split()
            key = alias.get(parts[0], parts[0])
            model = parts[1] if len(parts) > 1 else None
            try:
                info = await get_manager().set_active(key, model)
            except LLMError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer(
                f"\u2705 Switched to {info.key}\nModel: {info.model}\n\n"
                "New tasks will use this model."
            )

        @dp.message(Command("model"))
        async def _model(message: Message, command: CommandObject) -> None:
            if not await self._guard(message):
                return
            from app.llm import LLMError, get_manager

            model = (command.args or "").strip()
            manager = get_manager()
            if not model:
                await message.answer(
                    f"Current: {manager.active_key()} / {manager.active_model()}\n"
                    "Usage: /model <model name>   (see /models)"
                )
                return
            try:
                applied = await manager.set_model(model)
            except LLMError as exc:
                await message.answer(f"\u274C {exc}")
                return
            await message.answer(f"\u2705 Model set to {applied} on {manager.active_key()}")

        @dp.message(Command("inbox"))
        async def _inbox(message: Message) -> None:
            if not await self._guard(message):
                return
            async with session_scope() as session:
                rows = await repo.list_inbound(session, limit=15)
            if not rows:
                await message.answer("No WhatsApp/Teams messages received yet.")
                return
            lines = ["\U0001F4E5 Recent inbound:", ""]
            for row in rows:
                icon = "\U0001F4AC" if row.channel == "whatsapp" else "\U0001F465"
                who = row.sender_name or row.sender
                mark = "" if row.handled else " *"
                lines.append(f"{icon} {who[:24]}{mark}: {row.text[:70]}")
            lines += ["", "* = not yet handled"]
            await message.answer("\n".join(lines)[:4000])

        # Account/model setup commands live in their own module.
        from app.telegram.setup_commands import register_setup_handlers

        register_setup_handlers(self)

        # Websites, email, voice notes and the daily briefing.
        from app.telegram.capability_commands import register_capability_commands

        register_capability_commands(dp, self._guard)

        # Contacts/members sync and the self-updating skills memory.
        from app.telegram.member_commands import register_member_handlers

        register_member_handlers(dp, self._guard)

        @dp.message(Command("new"))
        async def _new(message: Message) -> None:
            # Start a fresh conversation thread. Old threads stay saved -
            # see /history to list and switch back to any of them.
            if not await self._guard(message):
                return
            async with session_scope() as session:
                await repo.ensure_session(session, message.chat.id)
                await repo.reset_session(session, message.chat.id)
            await message.answer(
                "\U0001F195 New chat started.\n"
                "(Your previous conversation is saved - see /history. "
                "Tasks and memory are untouched.)"
            )

        @dp.message(Command("history"))
        async def _history(message: Message, command: CommandObject) -> None:
            # List past conversation threads, or switch to one: /history <id>.
            if not await self._guard(message):
                return
            arg = (command.args or "").strip()
            async with session_scope() as session:
                if arg:
                    threads = await repo.list_threads(session, message.chat.id, limit=100)
                    match = next(
                        (t for t in threads if t["thread_id"].startswith(arg)), None
                    )
                    if match is None:
                        await message.answer(
                            f"\u274C No saved thread starts with '{arg}'. Send /history to list them."
                        )
                        return
                    await repo.switch_thread(session, message.chat.id, match["thread_id"])
                    await message.answer(
                        f"\U0001F4AC Switched to: {match['title']}\n"
                        f"({match['message_count']} messages)"
                    )
                    return

                threads = await repo.list_threads(session, message.chat.id, limit=15)

            if not threads:
                await message.answer("No conversation history yet.")
                return
            lines = ["\U0001F4DA Recent conversations:", ""]
            for t in threads:
                marker = "\u25B6\uFE0F " if t["is_current"] else "   "
                lines.append(
                    f"{marker}{t['thread_id'][:8]} - {t['title']} "
                    f"({t['message_count']} msgs)"
                )
            lines += ["", "/history <id> to switch back to one", "/new to start another"]
            await message.answer("\n".join(lines))

        @dp.message(Command("otpchat"))
        async def _otpchat(message: Message) -> None:
            """Open (creating it the first time) the dedicated OTP-bot thread.

            Answers "where do I send the number files?" once and for all -
            after this, this chat is in that thread, and every file sent goes
            straight into the queue.
            """
            if not await self._guard(message):
                return
            from app.automation import otp_bot

            await otp_bot.ensure_thread(message.chat.id)
            queue = await otp_bot.get_queue()
            active = await otp_bot.get_active_files()
            await message.answer(
                "\U0001F501 You are now in the OTP Bot thread.\n\n"
                "Send number files here - each one is queued automatically. "
                "Say \"start\" (or \"done\") when you've sent them all, \"stop\" to pause.\n\n"
                f"Queued now: {', '.join(f['name'] for f in queue) or '(none)'}\n"
                f"Currently running: {', '.join(f['name'] for f in active) or '(none)'}\n\n"
                "/history to switch back to another conversation."
            )

        @dp.message(Command("otpbot"))
        async def _otpbot(message: Message, command: CommandObject) -> None:
            # Read-only status/on/off for the deterministic OTP-number bot
            # automation. The actual queue-file/tag/start/stop workflow is
            # conversational - see app.automation.otp_bot's module docstring -
            # so the owner never has to remember command syntax for the
            # day-to-day flow, only this one for a quick health check.
            if not await self._guard(message):
                return
            from app.automation import otp_bot

            arg = (command.args or "").strip().lower()
            if arg in {"on", "enable", "enabled"}:
                cfg = await otp_bot.save_config({"enabled": True})
                await message.answer(
                    f"\u2705 Automation enabled - checking {cfg['target_bot']} every "
                    f"{cfg['interval_minutes']} min."
                )
                return
            if arg in {"off", "disable", "disabled"}:
                await otp_bot.stop_automation()
                await message.answer("\u23F8\uFE0F Automation disabled.")
                return
            if arg in {"run", "now"}:
                await message.answer("\U0001F504 Running one cycle now...")
                result = await otp_bot.run_cycle()
                if result.ok:
                    await message.answer(
                        f"\u2705 {result.action} (active quota was {result.active_quota})"
                        + (f"\n\n{result.add_reply[:500]}" if result.add_reply else "")
                    )
                else:
                    await message.answer(f"\u274C {result.error[:400]}")
                return

            cfg = await otp_bot.get_config()
            await message.answer(
                await otp_panel.status_text(),
                reply_markup=otp_panel.control_keyboard(bool(cfg["enabled"])),
            )

        @dp.callback_query(F.data.startswith(f"{otp_panel.PREFIX}:"))
        async def _otp_callback(query: CallbackQuery) -> None:
            """Every OTP panel button lands here.

            Buttons are shared state: the owner can tap one on an old message
            long after things moved on, so each branch re-reads current state
            rather than trusting what the keyboard was rendered from.
            """
            if not self._authorized(query):
                with contextlib.suppress(Exception):
                    await query.answer("Not authorized.", show_alert=True)
                return

            from app.automation import otp_bot, otp_schedule

            # "otp:<action>:<rest>" - rest may itself contain ':' (service
            # names are user-visible text), so split at most twice.
            try:
                _, action, rest = query.data.split(":", 2)
            except ValueError:
                await query.answer()
                return

            async def refresh_panel(note: str = "") -> None:
                cfg = await otp_bot.get_config()
                text = await otp_panel.status_text()
                if note:
                    text = f"{note}\n\n{text}"
                with contextlib.suppress(Exception):
                    # Telegram rejects an edit that changes nothing; that is
                    # a no-op for us, not an error worth surfacing.
                    await query.message.edit_text(
                        text,
                        reply_markup=otp_panel.control_keyboard(bool(cfg["enabled"])),
                    )

            if action == "svc":
                entry_id, service = rest.split(":", 1)
                entry = await otp_bot.set_queue_tag(entry_id, service)
                if entry is None:
                    await query.answer("That file is gone already.", show_alert=True)
                    return
                await query.answer(f"{service} set")
                await otp_bot.set_awaiting_tag_entry(None)
                reply = await otp_bot._continue_after_tagging(
                    f"\u2705 {entry.get('country') or entry['name']} -> {service}"
                )
                await self._send_next_prompt(query, reply)
                return

            if action == "skip":
                entry = await otp_bot.get_awaiting_tag_entry()
                await otp_bot.remove_from_queue(rest)
                await query.answer("Removed")
                label = (entry or {}).get("country") or "File"
                reply = await otp_bot._continue_after_tagging(f"\U0001F5D1 {label} bad deoa holo.")
                await self._send_next_prompt(query, reply)
                return

            if action == "int":
                minutes = int(rest)
                await otp_bot.save_config({"interval_minutes": minutes})
                await query.answer(f"Every {minutes} min")
                await refresh_panel(f"\u23F1 Checking every {minutes} minutes.")
                return

            if action == "clean":
                await otp_bot.set_cleanup_mode(rest)
                cfg = await otp_bot.get_config()
                if rest == "force" and not cfg.get("force_delete_uid"):
                    await query.answer()
                    with contextlib.suppress(Exception):
                        await query.message.answer(
                            "\u26A0\uFE0F Wipe mode needs your bot user id for /frcd.\n"
                            "Set it in the web UI (OTP Bot -> Your user id), "
                            "otherwise I'll fall back to /useddelete."
                        )
                    return
                await query.answer("Saved")
                await refresh_panel()
                return

            if action == "ask_int":
                cfg = await otp_bot.get_config()
                await query.answer()
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        "\u23F1 Koto min por por check korbo?",
                        reply_markup=otp_panel.interval_keyboard(cfg["interval_minutes"]),
                    )
                return

            if action == "ask_clean":
                cfg = await otp_bot.get_config()
                await query.answer()
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        "\U0001F9F9 Add korar age ki korbo?",
                        reply_markup=otp_panel.cleanup_keyboard(
                            bool(cfg.get("force_delete_before_add"))
                        ),
                    )
                return

            if action == "refreshc":
                await query.answer("Asking the bot...")
                outcome = await otp_bot.refresh_countries_from_bot()
                if not outcome["ok"]:
                    await refresh_panel(f"\u274C {outcome['error']}")
                    return
                names = ", ".join(outcome["countries"]) or "(kichu nai)"
                note = f"\U0001F30D Bot theke country list update kora holo: {names}"
                if outcome.get("new"):
                    note += f"\nNotun: {', '.join(outcome['new'])}"
                await refresh_panel(note)
                return

            if action == "ask_country":
                # Per-country settings: pick the country, then the interval.
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                markup = otp_panel.country_keyboard(entries, "pickc")
                await query.answer()
                if markup is None:
                    with contextlib.suppress(Exception):
                        await query.message.answer("Kono country nai - age file dao.")
                    return
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        "\U0001F30D Kon country-r setting bodlabo?", reply_markup=markup
                    )
                return

            if action == "pickc":
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                names = otp_panel.country_names(entries)
                index = int(rest)
                if index >= len(names):
                    await query.answer("That country is gone.", show_alert=True)
                    return
                country = names[index]
                cfg = await otp_bot.get_config()
                current = await otp_schedule.effective_config(country, cfg)
                await query.answer()
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        f"\u23F1 {country}: koto min por por check korbo?",
                        reply_markup=otp_panel.country_interval_keyboard(
                            index, current.get("interval_minutes")
                        ),
                    )
                return

            if action == "cint":
                index_raw, minutes_raw = rest.split(":", 1)
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                names = otp_panel.country_names(entries)
                index = int(index_raw)
                if index >= len(names):
                    await query.answer("That country is gone.", show_alert=True)
                    return
                country = names[index]
                minutes = int(minutes_raw)
                await otp_schedule.set_country_settings(
                    country, {"interval_minutes": minutes}
                )
                await otp_schedule.arm_country(country, minutes)
                await query.answer(f"{country}: every {minutes} min")
                await refresh_panel(f"\u23F1 {country} ekhon {minutes} min por por check hobe.")
                return

            if action == "ask_preset":
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                markup = otp_panel.country_keyboard(entries, "prec")
                await query.answer()
                if markup is None:
                    with contextlib.suppress(Exception):
                        await query.message.answer("Kono country nai - age file dao.")
                    return
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        "\U0001F4D0 Kon country-te preset apply korbo?", reply_markup=markup
                    )
                return

            if action == "prec":
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                names = otp_panel.country_names(entries)
                index = int(rest)
                if index >= len(names):
                    await query.answer("That country is gone.", show_alert=True)
                    return
                presets = await otp_schedule.get_presets()
                preset_names = sorted(presets)
                await query.answer()
                lines = [f"\U0001F4D0 {names[index]} - kon preset?", ""]
                for name in preset_names:
                    note = presets[name].get("_note", "")
                    every = presets[name].get("interval_minutes")
                    lines.append(f"\u2022 {name} ({every}m) - {note}" if note else f"\u2022 {name} ({every}m)")
                with contextlib.suppress(Exception):
                    await query.message.answer(
                        "\n".join(lines),
                        reply_markup=otp_panel.preset_keyboard(preset_names, index),
                    )
                return

            if action == "usepre":
                index_raw, preset_raw = rest.split(":", 1)
                entries = await otp_bot.get_active_files() + await otp_bot.get_queue()
                names = otp_panel.country_names(entries)
                index = int(index_raw)
                preset_names = sorted(await otp_schedule.get_presets())
                preset_index = int(preset_raw)
                if index >= len(names) or preset_index >= len(preset_names):
                    await query.answer("Gone already.", show_alert=True)
                    return
                country, preset = names[index], preset_names[preset_index]
                applied = await otp_schedule.apply_preset(preset, country)
                if applied is None:
                    await query.answer("Preset not found.", show_alert=True)
                    return
                await query.answer(f"{preset} applied")
                await refresh_panel(f"\U0001F4D0 {country} -> '{preset}' preset apply kora holo.")
                return

            if action == "start":
                await query.answer("Starting...")
                result = await otp_bot.start_automation()
                if result["ok"]:
                    await refresh_panel(otp_bot._format_start_success(result))
                elif result.get("missing_tags"):
                    entry = result["missing_tags"][0]
                    await otp_bot.set_awaiting_tag_entry(entry["id"])
                    with contextlib.suppress(Exception):
                        await query.message.answer(
                            otp_bot._tag_question(entry),
                            reply_markup=otp_panel.service_keyboard(entry["id"]),
                        )
                else:
                    await refresh_panel(f"\u274C {result['error']}")
                return

            if action == "stop":
                await otp_bot.stop_automation()
                await query.answer("Stopped")
                await refresh_panel("\u23F8 Bondho kora holo.")
                return

            if action == "run":
                await query.answer("Checking...")
                result = await otp_bot.run_cycle()
                note = (
                    f"\u2705 {result.action} (quota {result.active_quota})"
                    if result.ok
                    else f"\u274C {result.error}"
                )
                await refresh_panel(note)
                return

            if action == "clearq":
                count = await otp_bot.clear_queue()
                await query.answer(f"{count} removed" if count else "Already empty")
                await refresh_panel()
                return

            if action == "rmc":
                queue = await otp_bot.get_queue()
                active = await otp_bot.get_active_files()
                entry = next((e for e in queue + active if e["id"] == rest), None)
                if entry is None:
                    await query.answer("Already gone.", show_alert=True)
                    await refresh_panel()
                    return
                removed = await otp_bot.remove_by_country(entry.get("country") or "")
                await query.answer(f"{len(removed)} removed")
                await refresh_panel(f"\U0001F5D1 {entry.get('country')} bad deoa holo.")
                return

            if action == "status":
                await query.answer()
                await refresh_panel()
                return

            await query.answer()

        @dp.message(Command("mode"))
        async def _mode(message: Message, command: CommandObject) -> None:
            """auto (default) | chat (never create tasks) | task (always create)."""
            if not await self._guard(message):
                return
            choice = (command.args or "").strip().lower()
            if choice not in {"auto", "chat", "task"}:
                async with session_scope() as session:
                    row = await repo.ensure_session(session, message.chat.id)
                    current = row.mode
                await message.answer(
                    f"Current mode: {current}\n\n"
                    "/mode auto - I decide: chat vs background job (default)\n"
                    "/mode chat - just talk, never start jobs\n"
                    "/mode task - treat every message as a job"
                )
                return
            async with session_scope() as session:
                await repo.ensure_session(session, message.chat.id)
                await repo.update_session(session, message.chat.id, mode=choice)
            await message.answer(f"\u2705 Mode set to {choice}.")

        @dp.message(Command("session"))
        async def _session_info(message: Message) -> None:
            if not await self._guard(message):
                return
            async with session_scope() as session:
                row = await repo.ensure_session(session, message.chat.id)
                mode, turns, active_id = row.mode, row.turn_count, row.active_task_id
                started = row.started_at
                active = await repo.get_task(session, active_id) if active_id else None
                snapshot = (
                    {"id": active.id, "title": active.title, "status": active.status}
                    if active else None
                )
            lines = [
                "\U0001F4AC This conversation",
                "",
                f"Mode: {mode}",
                f"Turns: {turns}",
                f"Started: {started.strftime('%Y-%m-%d %H:%M UTC') if started else '-'}",
            ]
            if snapshot:
                lines += ["", f"Currently about: {snapshot['title'][:60]}",
                          f"  {snapshot['id']} - {snapshot['status'].lower()}"]
            else:
                lines += ["", "Not tied to any job right now."]
            lines += ["", "/new to start fresh   /mode to change behaviour"]
            await message.answer("\n".join(lines))

        @dp.message(F.document)
        async def _document(message: Message) -> None:
            """Save an uploaded file into the workspace so a task can use it."""
            if not await self._guard(message):
                return
            doc = message.document
            if doc is None:
                return
            if doc.file_size and doc.file_size > self.settings.max_file_bytes:
                await message.answer(
                    f"\u26A0\uFE0F That file is over the {self.settings.max_file_mb} MB limit."
                )
                return

            from app.security import safe_path

            name = doc.file_name or f"upload_{doc.file_unique_id}"
            # Keep the name but never let it escape uploads/ or collide silently.
            safe_name = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "file"
            target = safe_path(f"uploads/{safe_name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                stem, _, ext = safe_name.rpartition(".")
                stem = stem or safe_name
                target = safe_path(f"uploads/{stem}_{doc.file_unique_id}{'.' + ext if ext else ''}")

            try:
                file = await message.bot.get_file(doc.file_id)
                await message.bot.download_file(file.file_path, destination=str(target))
            except Exception as exc:  # noqa: BLE001
                await message.answer(f"\u274C Could not download that file: {str(exc)[:200]}")
                return

            from app.security import rel_path

            rel = rel_path(target)
            async with session_scope() as db_session:
                row = await repo.ensure_session(db_session, message.chat.id)
                ctx = dict(row.context or {})
                ctx["pending_upload"] = {"path": rel, "name": safe_name}
                await repo.update_session(db_session, message.chat.id, context=ctx)
                # Record it as a visible chat turn too - otherwise it only
                # lived in invisible session context and the web dashboard's
                # Chat panel (same shared conversation) never showed it.
                await repo.add_message(
                    db_session, chat_id=message.chat.id, role="user",
                    content=f"\U0001F4CE Uploaded: {safe_name}",
                    thread_id=row.current_thread_id,
                )

            from app.automation import otp_bot

            if await otp_bot.is_otp_thread(message.chat.id):
                analysis = await otp_bot.enqueue_file(rel, safe_name)
                countries = analysis["countries"]
                lines = [f"\U0001F4C1 Saved: {safe_name}", ""]
                lines.append(f"Detected {len(countries)} country/countries:")
                for country, count in countries.items():
                    lines.append(f"  \u2022 {country}: {count} numbers")
                queue = await otp_bot.get_queue()
                lines += [
                    "",
                    f"Queued ({len(queue)} entr{'y' if len(queue) == 1 else 'ies'} total). "
                    "Send more files, then tap Start when you're finished.",
                ]
                cfg = await otp_bot.get_config()
                await message.answer(
                    "\n".join(lines),
                    reply_markup=otp_panel.control_keyboard(bool(cfg["enabled"])),
                )
            else:
                await message.answer(
                    f"\U0001F4C1 Saved: {rel}\n\n"
                    "Tell me what to do with it - I'll attach it to your next instruction.\n"
                    "(For OTP-bot number files, use /otpchat to open the dedicated thread; "
                    "files sent there are queued automatically.)"
                )

        @dp.message(F.text & ~F.text.startswith("/"))
        async def _natural(message: Message) -> None:
            if not await self._guard(message):
                return
            text = (message.text or "").strip()
            if not text:
                return

            from app.agent.conversation import handle_message

            try:
                await self.bot.send_chat_action(message.chat.id, "typing")
            except Exception:  # noqa: BLE001 - cosmetic only
                pass

            try:
                reply = await handle_message(
                    message.chat.id, message.from_user.id, text
                )
            except Exception:  # noqa: BLE001 - never leave the owner without a reply
                log.exception("conversation_failed", extra={"chat_id": message.chat.id})
                await message.answer(
                    "\u274C Something broke handling that. It is logged - try again."
                )
                return

            await message.answer(reply.text[:4000])

    async def _decide(self, message: Message, command: CommandObject, *, approved: bool) -> None:
        if not await self._guard(message):
            return
        approval_id = (command.args or "").strip()
        if not approval_id:
            await message.answer(f"Usage: /{'approve' if approved else 'reject'} <approval_id>")
            return
        async with session_scope() as session:
            approval = await repo.decide_approval(
                session, approval_id, approved=approved, user_id=message.from_user.id
            )
            if approval is None:
                await message.answer(f"Approval not found: {approval_id}")
                return
            state = approval.status
            task_id = approval.task_id
            tool = approval.tool
            if state == ApprovalStatus.APPROVED.value:
                await repo.update_task(
                    session, task_id, status=TaskStatus.PENDING.value, run_after=None
                )
            elif state == ApprovalStatus.REJECTED.value:
                await repo.update_task(
                    session, task_id, status=TaskStatus.PENDING.value, run_after=None
                )
            await repo.log_event(session, "approval_decided", task_id=task_id,
                                 data={"status": state, "tool": tool})
            decided_args = dict(approval.args or {})

        # Remember the decision so this shape of action stops (or keeps)
        # interrupting the owner in future.
        from app.security import autonomy

        await autonomy.remember_decision(tool, decided_args, approved=approved)
        icons = {"APPROVED": "\u2705 Approved", "REJECTED": "\U0001F6AB Rejected",
                 "EXPIRED": "\u23F0 Expired", "PENDING": "\u23F3 Pending"}
        await message.answer(f"{icons.get(state, state)}: {tool} (task {task_id})")

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if not self.settings.telegram_enabled:
            log.warning("telegram_disabled",
                        extra={"reason": "missing token or allowlist"})
            return
        await self.bot.delete_webhook(drop_pending_updates=True)
        self._task = asyncio.create_task(
            self.dp.start_polling(self.bot, handle_signals=False, allowed_updates=["message"])
        )
        log.info("telegram_started", extra={"allowed_users": len(self.settings.allowed_user_ids)})

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self.dp.stop_polling()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        with contextlib.suppress(Exception):
            await self.bot.session.close()
        log.info("telegram_stopped")

    async def feed_update(self, update: Update) -> Any:
        """Used by tests / webhook mode."""
        return await self.dp.feed_update(self.bot, update)
