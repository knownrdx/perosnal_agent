"""Task control + scheduling tools (the agent managing its own work)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, JobKind, TaskStatus, utcnow
from app.scheduler.timeparse import TimeParseError, next_cron, parse_duration, parse_when
from app.security import Permission
from app.tools.base import Arg, InvalidInput, ToolContext
from app.tools.registry import tool


@tool(
    "task_create",
    description=(
        "Create a NEW background task that the worker will run independently. "
        "Use for work the owner asked to happen later or in parallel, not for your current step."
    ),
    permission=Permission.WRITE,
    args={
        "instruction": Arg("string", True, "Full natural-language instruction for the new task"),
        "delay": Arg("string", False, "Optional delay, e.g. 'in 30 minutes'", default=""),
    },
    timeout_s=30,
    max_retries=0,
)
async def task_create(
    instruction: str, delay: str = "", ctx: ToolContext | None = None
) -> dict[str, Any]:
    settings = get_settings()
    run_after = None
    if delay:
        try:
            run_after = utcnow() + timedelta(seconds=parse_duration(delay))
        except TimeParseError as exc:
            raise InvalidInput(str(exc)) from exc

    async with session_scope() as session:
        task = await repo.create_task(
            session,
            user_request=instruction,
            chat_id=ctx.chat_id if ctx else None,
            user_id=ctx.user_id if ctx else None,
            permission=(ctx.permission.value if ctx else Permission.WRITE.value),
            max_steps=settings.max_task_steps,
            max_retries=settings.max_task_retries,
            parent_task_id=ctx.task_id if ctx else None,
            run_after=run_after,
        )
        return {
            "created": True,
            "task_id": task.id,
            "status": task.status,
            "run_after": run_after.isoformat() if run_after else None,
        }


@tool(
    "task_status",
    description="Look up the status of a task by id, or list currently active tasks.",
    permission=Permission.READ,
    args={"task_id": Arg("string", False, "Task id; omit to list active tasks", default="")},
    timeout_s=30,
    max_retries=0,
)
async def task_status(task_id: str = "") -> dict[str, Any]:
    async with session_scope() as session:
        if task_id:
            task = await repo.get_task(session, task_id) or await repo.find_task_by_prefix(
                session, task_id
            )
            if task is None:
                return {"found": False, "task_id": task_id}
            return {
                "found": True,
                **task.short(),
                "result": task.result[:1000],
                "error": task.error[:500],
                "output_files": task.output_files,
            }
        rows = await repo.list_tasks(
            session, statuses=[s.value for s in ACTIVE_STATUSES], limit=20
        )
        return {"active": [t.short() for t in rows], "count": len(rows)}


@tool(
    "task_cancel",
    description="Cancel a running or pending task.",
    permission=Permission.HIGH_RISK,
    args={"task_id": Arg("string", True, "Task id to cancel")},
    timeout_s=30,
    max_retries=0,
    side_effect=True,
)
async def task_cancel(task_id: str, ctx: ToolContext | None = None) -> dict[str, Any]:
    async with session_scope() as session:
        task = await repo.get_task(session, task_id) or await repo.find_task_by_prefix(
            session, task_id
        )
        if task is None:
            raise InvalidInput(f"task not found: {task_id}")
        if ctx and task.id == ctx.task_id:
            raise InvalidInput("a task cannot cancel itself; finish with action=final instead")
        if task.status in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
            return {"cancelled": False, "task_id": task.id, "status": task.status,
                    "reason": "already finished"}
        await repo.set_task_status(session, task.id, TaskStatus.CANCELLED, error="cancelled by agent")
        await repo.log_event(session, "task_cancelled", task_id=task.id, data={"by": "agent"})
        return {"cancelled": True, "task_id": task.id}


@tool(
    "scheduler_create",
    description=(
        "Schedule an instruction to run later or repeatedly. "
        "kind=once with when='tomorrow 09:00' / 'in 2 hours' / ISO timestamp; "
        "kind=interval with every='30m'; kind=cron with cron='0 9 * * *'."
    ),
    permission=Permission.WRITE,
    args={
        "instruction": Arg("string", True, "What the agent should do when it fires"),
        "kind": Arg("string", False, "once | interval | cron", default="once",
                    choices=["once", "interval", "cron"]),
        "when": Arg("string", False, "Time expression for kind=once", default=""),
        "every": Arg("string", False, "Interval for kind=interval, e.g. '2h'", default=""),
        "cron": Arg("string", False, "Cron expression for kind=cron", default=""),
        "name": Arg("string", False, "Short job name", default=""),
        "max_runs": Arg("integer", False, "Stop after N runs", default=None),
    },
    timeout_s=30,
    max_retries=0,
)
async def scheduler_create(
    instruction: str,
    kind: str = "once",
    when: str = "",
    every: str = "",
    cron: str = "",
    name: str = "",
    max_runs: int | None = None,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    kind = kind.lower()
    interval_s: int | None = None
    try:
        if kind == "once":
            if not when:
                raise InvalidInput("kind=once requires 'when'")
            next_run = parse_when(when)
            job_kind = JobKind.ONCE
            max_runs = 1
        elif kind == "interval":
            if not every:
                raise InvalidInput("kind=interval requires 'every'")
            interval_s = parse_duration(every)
            if interval_s < 60:
                raise InvalidInput("minimum interval is 60 seconds")
            next_run = utcnow() + timedelta(seconds=interval_s)
            job_kind = JobKind.INTERVAL
        elif kind == "cron":
            if not cron:
                raise InvalidInput("kind=cron requires 'cron'")
            next_run = next_cron(cron)
            job_kind = JobKind.CRON
        else:
            raise InvalidInput("kind must be once, interval or cron")
    except TimeParseError as exc:
        raise InvalidInput(str(exc)) from exc

    async with session_scope() as session:
        job = await repo.create_job(
            session,
            name=name or instruction[:60],
            kind=job_kind.value,
            instruction=instruction,
            chat_id=ctx.chat_id if ctx else None,
            user_id=ctx.user_id if ctx else None,
            next_run_at=next_run,
            cron_expr=cron,
            interval_s=interval_s,
            max_runs=max_runs,
        )
        return {
            "created": True,
            "job_id": job.id,
            "kind": job.kind,
            "next_run_at": next_run.isoformat(),
        }


@tool(
    "scheduler_list",
    description="List scheduled jobs.",
    permission=Permission.READ,
    args={},
    timeout_s=30,
    max_retries=0,
)
async def scheduler_list() -> dict[str, Any]:
    async with session_scope() as session:
        jobs = await repo.list_jobs(session, limit=50)
        return {
            "count": len(jobs),
            "jobs": [
                {
                    "job_id": job.id,
                    "name": job.name,
                    "kind": job.kind,
                    "enabled": job.enabled,
                    "next_run_at": job.next_run_at.isoformat() if job.next_run_at else None,
                    "runs": job.runs,
                }
                for job in jobs
            ],
        }


@tool(
    "scheduler_cancel",
    description="Disable and remove a scheduled job.",
    permission=Permission.WRITE,
    args={"job_id": Arg("string", True, "Job id to remove")},
    timeout_s=30,
    max_retries=0,
    side_effect=True,
)
async def scheduler_cancel(job_id: str) -> dict[str, Any]:
    async with session_scope() as session:
        job = await repo.get_job(session, job_id)
        if job is None:
            raise InvalidInput(f"job not found: {job_id}")
        await repo.delete_job(session, job.id)
        return {"removed": True, "job_id": job.id}
