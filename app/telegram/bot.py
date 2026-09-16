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
from aiogram.types import Message, Update

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, ApprovalStatus, TaskStatus
from app.logging_conf import get_logger
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
            """Start a fresh conversation thread."""
            if not await self._guard(message):
                return
            async with session_scope() as session:
                await repo.ensure_session(session, message.chat.id)
                await repo.reset_session(session, message.chat.id)
            await message.answer(
                "\U0001F195 Fresh start. I have cleared this conversation's context.\n"
                "(Your tasks and memory are untouched.)"
            )

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

            await message.answer(
                f"\U0001F4C1 Saved: {rel}\n\n"
                "Tell me what to do with it, e.g. \"add these numbers to @PBDxbot\" "
                "- I will attach this file automatically to your next instruction."
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
