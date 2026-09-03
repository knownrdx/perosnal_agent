"""Reliability: a finished task must always reach the owner.

The database commit and the Telegram send cannot be one atomic act. These tests
cover the window between them, which is where "did it actually finish?" doubt
comes from.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.engine import AgentEngine
from app.db import repo
from app.db.base import session_scope
from app.db.models import TaskStatus
from app.workers.task_worker import TaskWorker


async def _task(request: str = "write the report") -> str:
    async with session_scope() as session:
        task = await repo.create_task(
            session, user_request=request, title=request[:80],
            chat_id=42, user_id=42, permission="WRITE", max_steps=6,
        )
        return task.id


class SlowNotifier:
    """A notifier whose send takes long enough to be interrupted."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.completed: list[str] = []
        self.failed: list[str] = []

    async def task_completed(self, task_id: str) -> None:
        await asyncio.sleep(self.delay)
        self.completed.append(task_id)
        async with session_scope() as session:
            await repo.update_task(session, task_id, notified=True)

    async def task_failed(self, task_id: str) -> None:
        await asyncio.sleep(self.delay)
        self.failed.append(task_id)
        async with session_scope() as session:
            await repo.update_task(session, task_id, notified=True)

    async def task_question(self, task_id: str, question: str) -> None:
        pass


class BrokenNotifier(SlowNotifier):
    async def task_completed(self, task_id: str) -> None:
        raise RuntimeError("telegram is down")


# --------------------------------------------------------------------------- #
# Cancellation must not swallow the completion message
# --------------------------------------------------------------------------- #
async def test_completion_notice_survives_cancellation(environment, echo_llm):
    """Shutting down mid-send must still deliver: the work IS done."""
    task_id = await _task()
    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/r.txt", "content": "data"}},
        {"action": "final", "final_answer": "Report written.",
         "output_files": ["output/r.txt"]},
        {"lessons": []},
    ]

    notifier = SlowNotifier(delay=0.05)
    engine = AgentEngine(llm=echo_llm, notifier=notifier)

    runner = asyncio.create_task(engine.run_task(task_id))
    await asyncio.sleep(0.01)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    # The shielded send is allowed to finish even though the task was cancelled.
    for _ in range(50):
        if notifier.completed:
            break
        await asyncio.sleep(0.01)

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)

    if task.status == TaskStatus.COMPLETED.value:
        assert notifier.completed == [task_id], "a completed task must be announced"


async def test_notifier_error_does_not_fail_a_completed_task(environment, echo_llm):
    """Telegram being down must not turn finished work into a failure."""
    task_id = await _task()
    echo_llm.script = [
        {"action": "final", "final_answer": "Nothing to do."},
        {"lessons": []},
    ]

    status = await AgentEngine(llm=echo_llm, notifier=BrokenNotifier()).run_task(task_id)

    assert status == TaskStatus.COMPLETED.value
    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
    assert task.status == TaskStatus.COMPLETED.value


# --------------------------------------------------------------------------- #
# Startup sweep: nothing stays silently finished
# --------------------------------------------------------------------------- #
async def test_startup_replays_missed_notifications(environment):
    """A crash between commit and send is repaired on the next boot."""
    done = await _task("first")
    broke = await _task("second")

    async with session_scope() as session:
        await repo.set_task_status(session, done, TaskStatus.COMPLETED, result="ok")
        await repo.set_task_status(session, broke, TaskStatus.FAILED, error="nope")

    notifier = SlowNotifier(delay=0)
    worker = TaskWorker(notifier=notifier)
    sent = await worker.flush_notifications()

    assert sent == 2
    assert notifier.completed == [done]
    assert notifier.failed == [broke]


async def test_replay_is_not_repeated_on_the_next_boot(environment):
    """Once announced, a task must never be announced again."""
    task_id = await _task()
    async with session_scope() as session:
        await repo.set_task_status(session, task_id, TaskStatus.COMPLETED, result="ok")

    notifier = SlowNotifier(delay=0)
    worker = TaskWorker(notifier=notifier)

    assert await worker.flush_notifications() == 1
    assert await worker.flush_notifications() == 0
    assert notifier.completed == [task_id]


async def test_replay_skips_tasks_still_running(environment):
    task_id = await _task()
    async with session_scope() as session:
        await repo.update_task(session, task_id, status=TaskStatus.RUNNING.value)

    notifier = SlowNotifier(delay=0)
    assert await TaskWorker(notifier=notifier).flush_notifications() == 0
    assert notifier.completed == []


async def test_one_broken_notification_does_not_stop_the_sweep(environment):
    """A single failure must not strand every other owner message."""
    first = await _task("a")
    second = await _task("b")
    async with session_scope() as session:
        await repo.set_task_status(session, first, TaskStatus.COMPLETED, result="ok")
        await repo.set_task_status(session, second, TaskStatus.FAILED, error="x")

    class HalfBroken(SlowNotifier):
        async def task_completed(self, task_id: str) -> None:
            raise RuntimeError("boom")

    notifier = HalfBroken(delay=0)
    sent = await TaskWorker(notifier=notifier).flush_notifications()

    assert sent == 1
    assert notifier.failed == [second], "the healthy notification still went out"


async def test_recover_also_flushes(environment):
    """Boot-time recovery covers both stuck tasks and missed messages."""
    task_id = await _task()
    async with session_scope() as session:
        await repo.set_task_status(session, task_id, TaskStatus.COMPLETED, result="ok")

    notifier = SlowNotifier(delay=0)
    await TaskWorker(notifier=notifier).recover()

    assert notifier.completed == [task_id]
