"""Task state machine, worker recovery and the scheduler."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.db import repo
from app.db.base import session_scope
from app.db.models import JobKind, TaskStatus, utcnow
from app.scheduler.timeparse import TimeParseError, next_cron, parse_duration, parse_when


async def test_task_lifecycle_persisted(environment):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="do work", chat_id=42)
        task_id = task.id
    async with session_scope() as session:
        fetched = await repo.get_task(session, task_id)
        assert fetched.status == TaskStatus.PENDING.value
        await repo.set_task_status(session, task_id, TaskStatus.COMPLETED, result="done")
    async with session_scope() as session:
        fetched = await repo.get_task(session, task_id)
        assert fetched.status == TaskStatus.COMPLETED.value
        assert fetched.completed_at is not None


async def test_claim_is_exclusive(environment):
    async with session_scope() as session:
        await repo.create_task(session, user_request="only one worker gets this")

    async with session_scope() as session:
        first = await repo.claim_next_task(session, "worker-a")
    async with session_scope() as session:
        second = await repo.claim_next_task(session, "worker-b")

    assert first is not None
    assert second is None, "a claimed task must not be claimed twice"


async def test_expired_lease_is_reclaimed(environment):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="crashed worker task")
        task_id = task.id
    async with session_scope() as session:
        claimed = await repo.claim_next_task(session, "worker-a", lease_s=1)
        assert claimed is not None
    async with session_scope() as session:
        await repo.update_task(
            session, task_id, lease_expires_at=utcnow() - timedelta(seconds=10)
        )
    async with session_scope() as session:
        reclaimed = await repo.claim_next_task(session, "worker-b")
    assert reclaimed is not None and reclaimed.id == task_id


async def test_recover_stale_running_on_startup(environment):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="was running at crash")
        await repo.update_task(
            session, task.id, status=TaskStatus.RUNNING.value, worker_id="dead-worker"
        )
        task_id = task.id
    async with session_scope() as session:
        count = await repo.recover_stale_running(session)
    assert count == 1
    async with session_scope() as session:
        recovered = await repo.get_task(session, task_id)
    assert recovered.status == TaskStatus.PENDING.value and recovered.worker_id is None


async def test_run_after_delays_claim(environment):
    async with session_scope() as session:
        await repo.create_task(
            session, user_request="future task", run_after=utcnow() + timedelta(hours=1)
        )
    async with session_scope() as session:
        claimed = await repo.claim_next_task(session, "worker-a")
    assert claimed is None


async def test_scheduler_fires_due_job_and_reschedules(environment):
    from app.workers.scheduler_worker import SchedulerRunner

    async with session_scope() as session:
        job = await repo.create_job(
            session,
            name="daily check",
            kind=JobKind.INTERVAL.value,
            instruction="check the thing",
            chat_id=42,
            user_id=42,
            next_run_at=utcnow() - timedelta(seconds=5),
            interval_s=3600,
        )
        job_id = job.id

    fired = await SchedulerRunner().tick()
    assert fired == 1

    async with session_scope() as session:
        tasks = await repo.list_tasks(session, limit=10)
        refreshed = await repo.get_job(session, job_id)
    assert any(t.user_request == "check the thing" for t in tasks)
    assert refreshed.runs == 1 and refreshed.next_run_at > utcnow()


async def test_once_job_disables_after_firing(environment):
    from app.workers.scheduler_worker import SchedulerRunner

    async with session_scope() as session:
        job = await repo.create_job(
            session, name="one shot", kind=JobKind.ONCE.value, instruction="ping",
            chat_id=42, user_id=42, next_run_at=utcnow() - timedelta(seconds=1), max_runs=1,
        )
        job_id = job.id

    assert await SchedulerRunner().tick() == 1
    assert await SchedulerRunner().tick() == 0

    async with session_scope() as session:
        refreshed = await repo.get_job(session, job_id)
    assert refreshed.enabled is False


def test_parse_duration():
    assert parse_duration("2h") == 7200
    assert parse_duration("in 30 minutes") == 1800
    assert parse_duration("45s") == 45
    with pytest.raises(TimeParseError):
        parse_duration("whenever")


def test_parse_when_relative_and_absolute():
    base = utcnow()
    assert parse_when("in 2 hours", base=base) == base + timedelta(hours=2)
    tomorrow = parse_when("tomorrow 09:30", base=base)
    assert tomorrow > base and tomorrow.hour == 9 and tomorrow.minute == 30
    iso = parse_when("2030-01-01T10:00:00Z")
    assert iso.year == 2030
    with pytest.raises(TimeParseError):
        parse_when("sometime soon")


def test_cron_next_run():
    base = utcnow()
    nxt = next_cron("0 9 * * *", base=base)
    assert nxt > base and nxt.minute == 0
    with pytest.raises(TimeParseError):
        next_cron("not a cron")


async def test_worker_processes_queued_task(environment, echo_llm):
    """The worker loop claims and completes a task without blocking."""
    import asyncio

    from app.agent.engine import AgentEngine
    from app.workers.task_worker import TaskWorker

    async with session_scope() as session:
        task = await repo.create_task(session, user_request="write a file", chat_id=42)
        task_id = task.id

    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/worker.txt", "content": "from worker"}},
        {"action": "final", "final_answer": "Worker done.", "output_files": ["output/worker.txt"]},
    ]

    worker = TaskWorker(engine=AgentEngine(llm=echo_llm))
    worker.poll_interval = 0.05
    await worker.start()
    try:
        for _ in range(100):
            async with session_scope() as session:
                current = await repo.get_task(session, task_id)
            if current.status in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}:
                break
            await asyncio.sleep(0.1)
    finally:
        await worker.stop()

    assert current.status == TaskStatus.COMPLETED.value
    assert (environment.workspace / "output" / "worker.txt").exists()
