"""Background task worker.

Claims tasks with a database lease so a crashed worker's task is picked up by
another one instead of being lost.  Telegram and the HTTP API are never blocked
by task execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import uuid
from typing import Any

from app.agent.engine import AgentEngine
from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import FailureKind, TaskStatus
from app.logging_conf import get_logger

log = get_logger(__name__)


def worker_identity(index: int) -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{index}-{uuid.uuid4().hex[:6]}"


class TaskWorker:
    """Runs ``concurrency`` parallel claim/execute loops."""

    def __init__(self, engine: AgentEngine | None = None, notifier: Any | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.notifier = notifier
        self.engine = engine or AgentEngine(notifier=notifier)
        self.concurrency = max(1, settings.worker_concurrency)
        self.poll_interval = settings.worker_poll_interval_s
        self.lease_s = max(120, settings.task_timeout_s)
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        recovered = await self.recover()
        log.info("worker_start", extra={"concurrency": self.concurrency, "recovered": recovered})
        for index in range(self.concurrency):
            self._tasks.append(asyncio.create_task(self._loop(worker_identity(index))))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        log.info("worker_stopped")

    async def recover(self) -> int:
        """Requeue tasks left RUNNING by a previous process."""
        async with session_scope() as session:
            count = await repo.recover_stale_running(session)
            if count:
                await repo.log_event(session, "worker_recovered_tasks", data={"count": count})
        await self.flush_notifications()
        return count

    async def flush_notifications(self) -> int:
        """Deliver messages for tasks that finished but were never announced.

        The status commit and the Telegram send cannot be one atomic act, so a
        crash in between leaves a task done-but-silent. Sweeping on startup
        closes that window: the owner always finds out, even if late.
        """
        if self.notifier is None:
            return 0
        async with session_scope() as session:
            pending = await repo.unnotified_terminal_tasks(session)

        sent = 0
        for task in pending:
            event = (
                "task_completed"
                if task.status == TaskStatus.COMPLETED.value
                else "task_failed"
            )
            handler = getattr(self.notifier, event, None)
            if handler is None:
                continue
            try:
                await handler(task.id)
                sent += 1
            except Exception as exc:  # noqa: BLE001 - one bad task must not stop the sweep
                log.error(
                    "notify_replay_failed",
                    extra={"task_id": task.id, "error": str(exc)[:200]},
                )
        if sent:
            log.info("notify_replayed", extra={"count": sent})
        return sent

    # ------------------------------------------------------------------ #
    async def _loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                task = await self._claim(worker_id)
                if task is None:
                    await asyncio.sleep(self.poll_interval)
                    continue
                await self._run_one(task.id, worker_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a worker loop must never die
                log.exception("worker_loop_error", extra={"worker": worker_id})
                await asyncio.sleep(min(30, self.poll_interval * 5))

    async def _claim(self, worker_id: str):
        async with session_scope() as session:
            return await repo.claim_next_task(session, worker_id, lease_s=self.lease_s)

    async def _run_one(self, task_id: str, worker_id: str) -> None:
        log.info("task_claimed", extra={"task_id": task_id, "worker": worker_id})
        heartbeat = asyncio.create_task(self._heartbeat(task_id, worker_id))
        try:
            status = await asyncio.wait_for(
                self.engine.run_task(task_id), timeout=self.settings.task_timeout_s
            )
            log.info("task_finished", extra={"task_id": task_id, "status": status})
        except asyncio.TimeoutError:
            async with session_scope() as session:
                await repo.set_task_status(
                    session,
                    task_id,
                    TaskStatus.FAILED,
                    error=f"task exceeded the global timeout ({self.settings.task_timeout_s}s)",
                    failure_kind=FailureKind.TEMPORARY.value,
                )
            log.error("task_timeout", extra={"task_id": task_id})
            if self.notifier is not None:
                await self.notifier.task_failed(task_id)
        except asyncio.CancelledError:
            async with session_scope() as session:
                await repo.update_task(
                    session, task_id, status=TaskStatus.PENDING.value, worker_id=None,
                    lease_expires_at=None,
                )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("task_crashed", extra={"task_id": task_id})
            async with session_scope() as session:
                await repo.set_task_status(
                    session,
                    task_id,
                    TaskStatus.FAILED,
                    error=f"unhandled error: {type(exc).__name__}: {exc}"[:2000],
                    failure_kind=FailureKind.UNKNOWN.value,
                )
            if self.notifier is not None:
                await self.notifier.task_failed(task_id)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, task_id: str, worker_id: str) -> None:
        interval = max(30, self.lease_s // 3)
        while True:
            await asyncio.sleep(interval)
            async with session_scope() as session:
                await repo.renew_lease(session, task_id, worker_id, lease_s=self.lease_s)
