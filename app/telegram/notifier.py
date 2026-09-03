"""Outbound Telegram notifications.

Kept separate from the tools so system messages (task completed / failed /
approval requests) do not depend on the agent loop.  Every notification is
idempotency-guarded so a restart cannot double-notify.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger
from app.tools.telegram_tools import get_api

log = get_logger(__name__)

MAX_LEN = 4000


class Notifier:
    """Sends task lifecycle updates to the owner's Telegram chat."""

    def __init__(self, api: Any | None = None) -> None:
        self._api = api

    @property
    def api(self) -> Any:
        return self._api or get_api()

    # ------------------------------------------------------------------ #
    async def send(self, chat_id: int | None, text: str, *, dedupe_key: str | None = None) -> dict[str, Any]:
        settings = get_settings()
        target = chat_id or settings.owner_chat_id
        if not target:
            log.warning("notify_no_chat", extra={"event": "notify_skipped"})
            return {"sent": False, "reason": "no chat id configured"}
        if not settings.telegram_bot_token:
            log.warning("notify_no_token")
            return {"sent": False, "reason": "no bot token configured"}

        if dedupe_key:
            async with session_scope() as session:
                op = await repo.record_operation(
                    session, key=dedupe_key, kind="notify", task_id=None
                )
            if op is None:
                log.info("notify_duplicate_skipped", extra={"operation_key": dedupe_key})
                return {"sent": False, "reason": "already notified"}

        try:
            result = await self.api.call(
                "sendMessage",
                {"chat_id": int(target), "text": text[:MAX_LEN], "disable_web_page_preview": "true"},
            )
            return {"sent": True, "message_id": result.get("message_id")}
        except Exception as exc:  # noqa: BLE001 - notification must never crash a task
            log.error("notify_failed", extra={"error": str(exc)[:300]})
            return {"sent": False, "reason": str(exc)[:200]}

    async def send_file(self, chat_id: int | None, path: str, caption: str = "") -> dict[str, Any]:
        from app.security import safe_path

        settings = get_settings()
        target = chat_id or settings.owner_chat_id
        if not target:
            return {"sent": False, "reason": "no chat id configured"}
        file_path = safe_path(path, must_exist=True)
        with file_path.open("rb") as handle:
            result = await self.api.call(
                "sendDocument",
                {"chat_id": int(target), "caption": caption[:1024]},
                {"document": (file_path.name, handle)},
            )
        return {"sent": True, "message_id": result.get("message_id")}

    # ------------------------------------------------------------------ #
    async def task_started(self, task_id: str) -> None:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                return
            chat_id, title = task.chat_id, task.title
        await self.send(
            chat_id,
            f"\U0001F680 Task started\n\nID: {task_id}\nTask: {title[:200]}\n\nStatus:\n\u23F3 Running",
            dedupe_key=f"notify:started:{task_id}",
        )

    async def task_completed(self, task_id: str) -> None:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                return
            chat_id = task.chat_id
            title = task.title
            result = task.result or "Done."
            files = list(task.output_files or [])
            already = task.notified
            if not already:
                await repo.update_task(session, task_id, notified=True)
        if already:
            return

        lines = [
            "\u2705 Task completed",
            "",
            f"ID: {task_id}",
            f"Task: {title[:200]}",
            "",
            result[:2500],
        ]
        if files:
            lines += ["", "\U0001F4C1 Files:"] + [f"- {f}" for f in files[:10]]
        await self.send(chat_id, "\n".join(lines), dedupe_key=f"notify:done:{task_id}")

    async def task_failed(self, task_id: str) -> None:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                return
            chat_id, title = task.chat_id, task.title
            error, kind = task.error, task.failure_kind
            already = task.notified
            if not already:
                await repo.update_task(session, task_id, notified=True)
        if already:
            return
        await self.send(
            chat_id,
            "\u274C Task failed\n\n"
            f"ID: {task_id}\nTask: {title[:200]}\nType: {kind or 'UNKNOWN'}\n\n{error[:1500]}",
            dedupe_key=f"notify:failed:{task_id}",
        )

    async def task_question(self, task_id: str, question: str) -> None:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            chat_id = task.chat_id if task else None
        await self.send(
            chat_id,
            f"\u2753 Input needed\n\nTask: {task_id}\n\n{question[:2000]}\n\n"
            f"Reply with: /reply {task_id} <your answer>",
        )

    async def approval_request(
        self, *, chat_id: int | None, approval_id: str, task_id: str, tool: str, args: dict[str, Any]
    ) -> None:
        pretty = "\n".join(f"  {k}: {str(v)[:200]}" for k, v in list(args.items())[:8])
        await self.send(
            chat_id,
            "\U0001F510 Approval required\n\n"
            f"Task: {task_id}\nTool: {tool} (HIGH_RISK)\n\nArguments:\n{pretty}\n\n"
            f"Approve: /approve {approval_id}\nReject:  /reject {approval_id}",
            dedupe_key=f"notify:approval:{approval_id}",
        )
