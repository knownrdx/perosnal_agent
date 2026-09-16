"""Repository layer.

All database access goes through these functions so the engine, workers, bot
and API share exactly one definition of "how state changes".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ACTIVE_STATUSES,
    AppSetting,
    ChatSession,
    Approval,
    ApprovalPattern,
    ApprovalStatus,
    Contact,
    Conversation,
    Credential,
    InboundMessage,
    Event,
    MemoryEntry,
    Operation,
    ScheduledJob,
    Task,
    TaskStatus,
    ToolCall,
    ToolCallStatus,
    new_id,
    utcnow,
)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
async def create_task(
    session: AsyncSession,
    *,
    user_request: str,
    title: str = "",
    chat_id: int | None = None,
    user_id: int | None = None,
    permission: str = "WRITE",
    max_steps: int = 14,
    max_retries: int = 2,
    context: dict[str, Any] | None = None,
    parent_task_id: str | None = None,
    scheduled_job_id: str | None = None,
    run_after: datetime | None = None,
) -> Task:
    task = Task(
        id=new_id(),
        user_request=user_request,
        title=(title or user_request)[:200],
        status=TaskStatus.PENDING.value,
        chat_id=chat_id,
        user_id=user_id,
        permission=permission,
        max_steps=max_steps,
        max_retries=max_retries,
        context=context or {},
        parent_task_id=parent_task_id,
        scheduled_job_id=scheduled_job_id,
        run_after=run_after,
        output_files=[],
    )
    session.add(task)
    await session.flush()
    return task


async def get_task(session: AsyncSession, task_id: str) -> Task | None:
    return await session.get(Task, task_id)


async def find_task_by_prefix(session: AsyncSession, prefix: str) -> Task | None:
    stmt = select(Task).where(Task.id.like(f"{prefix}%")).order_by(Task.created_at.desc()).limit(2)
    rows = (await session.scalars(stmt)).all()
    return rows[0] if len(rows) == 1 else None


async def list_tasks(
    session: AsyncSession,
    *,
    statuses: Sequence[str] | None = None,
    chat_id: int | None = None,
    limit: int = 20,
) -> list[Task]:
    stmt = select(Task).order_by(Task.created_at.desc()).limit(limit)
    if statuses:
        stmt = stmt.where(Task.status.in_(list(statuses)))
    if chat_id is not None:
        stmt = stmt.where(Task.chat_id == chat_id)
    return list((await session.scalars(stmt)).all())


async def count_tasks_by_status(session: AsyncSession) -> dict[str, int]:
    stmt = select(Task.status, func.count()).group_by(Task.status)
    return {status: count for status, count in (await session.execute(stmt)).all()}


async def claim_next_task(session: AsyncSession, worker_id: str, lease_s: int = 900) -> Task | None:
    """Atomically claim one runnable task for this worker.

    Runnable = PENDING/WAITING whose ``run_after`` has passed, or a RUNNING task
    whose worker lease expired (previous worker crashed).
    """
    now = utcnow()
    stmt = (
        select(Task)
        .where(
            or_(
                Task.status.in_([TaskStatus.PENDING.value, TaskStatus.WAITING.value]),
                (Task.status == TaskStatus.RUNNING.value) & (Task.lease_expires_at < now),
            ),
            or_(Task.run_after.is_(None), Task.run_after <= now),
        )
        .order_by(Task.created_at.asc())
        .limit(5)
    )
    candidates = list((await session.scalars(stmt)).all())
    for task in candidates:
        # Optimistic lock: only claim if the row still looks the way we read it.
        result = await session.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.status == task.status,
                or_(Task.worker_id.is_(None), Task.worker_id == task.worker_id),
            )
            .values(
                status=TaskStatus.RUNNING.value,
                worker_id=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_s),
                started_at=task.started_at or now,
                updated_at=now,
            )
        )
        if result.rowcount:
            await session.commit()
            await session.refresh(task)
            return task
    return None


async def renew_lease(session: AsyncSession, task_id: str, worker_id: str, lease_s: int = 900) -> None:
    await session.execute(
        update(Task)
        .where(Task.id == task_id, Task.worker_id == worker_id)
        .values(lease_expires_at=utcnow() + timedelta(seconds=lease_s), updated_at=utcnow())
    )


async def update_task(session: AsyncSession, task_id: str, **values: Any) -> None:
    values.setdefault("updated_at", utcnow())
    await session.execute(update(Task).where(Task.id == task_id).values(**values))


async def unnotified_terminal_tasks(
    session: AsyncSession, limit: int = 50
) -> list[Task]:
    """Tasks that finished but whose owner was never told.

    A crash between committing the status and sending the message would
    otherwise leave the owner permanently unaware that work completed.
    """
    stmt = (
        select(Task)
        .where(
            Task.status.in_([TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]),
            Task.notified.is_(False),
        )
        .order_by(Task.updated_at.asc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def update_task_context(
    session: AsyncSession, task_id: str, patch: dict[str, Any]
) -> None:
    """Merge keys into a task's context without dropping what is already there."""
    task = await session.get(Task, task_id)
    if task is None:
        return
    merged = dict(task.context or {})
    merged.update(patch)
    task.context = merged
    task.updated_at = utcnow()
    await session.flush()


async def set_task_status(
    session: AsyncSession,
    task_id: str,
    status: TaskStatus,
    *,
    result: str | None = None,
    error: str | None = None,
    failure_kind: str | None = None,
) -> None:
    values: dict[str, Any] = {"status": status.value, "updated_at": utcnow()}
    if status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        values["completed_at"] = utcnow()
        values["worker_id"] = None
        values["lease_expires_at"] = None
    if result is not None:
        values["result"] = result
    if error is not None:
        values["error"] = error
    if failure_kind is not None:
        values["failure_kind"] = failure_kind
    await session.execute(update(Task).where(Task.id == task_id).values(**values))


async def add_output_file(session: AsyncSession, task_id: str, path: str) -> None:
    task = await session.get(Task, task_id)
    if task is None:
        return
    files = list(task.output_files or [])
    if path not in files:
        files.append(path)
        task.output_files = files
        task.updated_at = utcnow()
        session.add(task)


async def requeue_task(session: AsyncSession, task_id: str, delay_s: int = 30) -> None:
    await session.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=TaskStatus.PENDING.value,
            worker_id=None,
            lease_expires_at=None,
            run_after=utcnow() + timedelta(seconds=delay_s),
            retry_count=Task.retry_count + 1,
            updated_at=utcnow(),
        )
    )


async def recover_stale_running(session: AsyncSession) -> int:
    """On startup: hand crashed RUNNING tasks back to the queue."""
    result = await session.execute(
        update(Task)
        .where(Task.status == TaskStatus.RUNNING.value)
        .values(
            status=TaskStatus.PENDING.value,
            worker_id=None,
            lease_expires_at=None,
            updated_at=utcnow(),
        )
    )
    return result.rowcount or 0


# --------------------------------------------------------------------------- #
# Tool calls + events
# --------------------------------------------------------------------------- #
async def record_tool_call(
    session: AsyncSession,
    *,
    task_id: str,
    step: int,
    tool: str,
    args: dict[str, Any],
    status: ToolCallStatus,
    result: dict[str, Any] | None = None,
    error: str = "",
    failure_kind: str = "",
    attempts: int = 1,
    duration_ms: float = 0.0,
    operation_key: str | None = None,
) -> ToolCall:
    call = ToolCall(
        id=new_id(),
        task_id=task_id,
        step=step,
        tool=tool,
        args=args,
        status=status.value,
        result=result or {},
        error=error[:4000],
        failure_kind=failure_kind,
        attempts=attempts,
        duration_ms=duration_ms,
        operation_key=operation_key,
    )
    session.add(call)
    await session.flush()
    return call


async def list_tool_calls(session: AsyncSession, task_id: str, limit: int = 50) -> list[ToolCall]:
    stmt = (
        select(ToolCall)
        .where(ToolCall.task_id == task_id)
        .order_by(ToolCall.created_at.asc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def log_event(
    session: AsyncSession,
    event: str,
    *,
    task_id: str | None = None,
    level: str = "INFO",
    data: dict[str, Any] | None = None,
) -> None:
    session.add(Event(task_id=task_id, level=level, event=event, data=data or {}))


async def recent_events(session: AsyncSession, limit: int = 50) -> list[Event]:
    stmt = select(Event).order_by(Event.created_at.desc()).limit(limit)
    return list((await session.scalars(stmt)).all())


# --------------------------------------------------------------------------- #
# Approvals
# --------------------------------------------------------------------------- #
async def create_approval(
    session: AsyncSession,
    *,
    task_id: str,
    tool: str,
    args: dict[str, Any],
    reason: str,
    ttl_minutes: int = 720,
) -> Approval:
    approval = Approval(
        id=new_id(),
        task_id=task_id,
        tool=tool,
        args=args,
        reason=reason,
        expires_at=utcnow() + timedelta(minutes=ttl_minutes),
    )
    session.add(approval)
    await session.flush()
    return approval


async def get_approval(session: AsyncSession, approval_id: str) -> Approval | None:
    approval = await session.get(Approval, approval_id)
    if approval is not None:
        return approval
    stmt = select(Approval).where(Approval.id.like(f"{approval_id}%")).limit(2)
    rows = (await session.scalars(stmt)).all()
    return rows[0] if len(rows) == 1 else None


async def pending_approval_for_task(session: AsyncSession, task_id: str) -> Approval | None:
    stmt = (
        select(Approval)
        .where(Approval.task_id == task_id, Approval.status == ApprovalStatus.PENDING.value)
        .order_by(Approval.created_at.desc())
        .limit(1)
    )
    return (await session.scalars(stmt)).first()


async def decide_approval(
    session: AsyncSession, approval_id: str, *, approved: bool, user_id: int | None
) -> Approval | None:
    approval = await get_approval(session, approval_id)
    if approval is None or approval.status != ApprovalStatus.PENDING.value:
        return approval
    expires = _aware(approval.expires_at)
    if expires and expires < utcnow():
        approval.status = ApprovalStatus.EXPIRED.value
    else:
        approval.status = (
            ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REJECTED.value
        )
    approval.decided_by = user_id
    approval.decided_at = utcnow()
    session.add(approval)
    await session.flush()
    return approval


async def list_pending_approvals(session: AsyncSession, limit: int = 20) -> list[Approval]:
    stmt = (
        select(Approval)
        .where(Approval.status == ApprovalStatus.PENDING.value)
        .order_by(Approval.created_at.desc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
async def memory_store(
    session: AsyncSession,
    *,
    key: str,
    value: str,
    kind: str = "fact",
    tags: list[str] | None = None,
    source_task_id: str | None = None,
) -> MemoryEntry:
    existing = (await session.scalars(select(MemoryEntry).where(MemoryEntry.key == key))).first()
    if existing:
        existing.value = value
        existing.kind = kind
        existing.tags = tags or existing.tags
        existing.updated_at = utcnow()
        session.add(existing)
        await session.flush()
        return existing
    entry = MemoryEntry(
        id=new_id(),
        key=key,
        value=value,
        kind=kind,
        tags=tags or [],
        source_task_id=source_task_id,
    )
    session.add(entry)
    await session.flush()
    return entry


async def memory_search(session: AsyncSession, query: str, limit: int = 10) -> list[MemoryEntry]:
    like = f"%{query.lower()}%"
    stmt = (
        select(MemoryEntry)
        .where(
            or_(
                func.lower(MemoryEntry.key).like(like),
                func.lower(MemoryEntry.value).like(like),
            )
        )
        .order_by(MemoryEntry.updated_at.desc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def memory_recent(
    session: AsyncSession, limit: int = 10, *, kinds: Sequence[str] | None = None
) -> list[MemoryEntry]:
    stmt = (
        select(MemoryEntry)
        .order_by(MemoryEntry.updated_at.desc(), MemoryEntry.id.desc())
        .limit(limit)
    )
    if kinds:
        stmt = stmt.where(MemoryEntry.kind.in_(list(kinds)))
    return list((await session.scalars(stmt)).all())


async def memory_delete(session: AsyncSession, key: str) -> bool:
    result = await session.execute(delete(MemoryEntry).where(MemoryEntry.key == key))
    return bool(result.rowcount)


# --------------------------------------------------------------------------- #
# Conversation (short-term memory)
# --------------------------------------------------------------------------- #
async def add_message(
    session: AsyncSession, *, chat_id: int, role: str, content: str, task_id: str | None = None
) -> None:
    session.add(
        Conversation(chat_id=chat_id, role=role, content=content[:8000], task_id=task_id)
    )


async def recent_messages(session: AsyncSession, chat_id: int, limit: int = 10) -> list[Conversation]:
    """Most recent turns, oldest first.

    Ordered by id as well as timestamp: two turns written in the same instant
    would otherwise come back in an arbitrary order, which would show the LLM a
    conversation where the answer precedes the question.
    """
    stmt = (
        select(Conversation)
        .where(Conversation.chat_id == chat_id)
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .limit(limit)
    )
    rows = list((await session.scalars(stmt)).all())
    return list(reversed(rows))


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
async def create_job(
    session: AsyncSession,
    *,
    name: str,
    kind: str,
    instruction: str,
    chat_id: int | None,
    user_id: int | None,
    next_run_at: datetime | None,
    cron_expr: str = "",
    interval_s: int | None = None,
    max_runs: int | None = None,
) -> ScheduledJob:
    job = ScheduledJob(
        id=new_id(),
        name=name[:200],
        kind=kind,
        instruction=instruction,
        chat_id=chat_id,
        user_id=user_id,
        cron_expr=cron_expr,
        interval_s=interval_s,
        next_run_at=next_run_at,
        max_runs=max_runs,
    )
    session.add(job)
    await session.flush()
    return job


async def due_jobs(session: AsyncSession, limit: int = 10) -> list[ScheduledJob]:
    stmt = (
        select(ScheduledJob)
        .where(
            ScheduledJob.enabled.is_(True),
            ScheduledJob.next_run_at.is_not(None),
            ScheduledJob.next_run_at <= utcnow(),
        )
        .order_by(ScheduledJob.next_run_at.asc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def list_jobs(session: AsyncSession, limit: int = 50) -> list[ScheduledJob]:
    stmt = select(ScheduledJob).order_by(ScheduledJob.created_at.desc()).limit(limit)
    return list((await session.scalars(stmt)).all())


async def get_job(session: AsyncSession, job_id: str) -> ScheduledJob | None:
    job = await session.get(ScheduledJob, job_id)
    if job is not None:
        return job
    stmt = select(ScheduledJob).where(ScheduledJob.id.like(f"{job_id}%")).limit(2)
    rows = (await session.scalars(stmt)).all()
    return rows[0] if len(rows) == 1 else None


async def update_job(session: AsyncSession, job_id: str, **values: Any) -> None:
    await session.execute(update(ScheduledJob).where(ScheduledJob.id == job_id).values(**values))


async def delete_job(session: AsyncSession, job_id: str) -> bool:
    result = await session.execute(delete(ScheduledJob).where(ScheduledJob.id == job_id))
    return bool(result.rowcount)


# --------------------------------------------------------------------------- #
# Idempotency ledger
# --------------------------------------------------------------------------- #
async def find_operation(session: AsyncSession, key: str) -> Operation | None:
    return (await session.scalars(select(Operation).where(Operation.key == key))).first()


async def record_operation(
    session: AsyncSession,
    *,
    key: str,
    kind: str,
    task_id: str | None,
    result: dict[str, Any] | None = None,
) -> Operation | None:
    """Insert an operation record; returns None when the key already exists."""
    op = Operation(id=new_id(), key=key, kind=kind, task_id=task_id, result=result or {})
    session.add(op)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return None
    return op


# --------------------------------------------------------------------------- #
# App settings (runtime key/value, e.g. active LLM)
# --------------------------------------------------------------------------- #
async def get_setting(session: AsyncSession, key: str) -> dict[str, Any] | None:
    row = await session.get(AppSetting, key)
    return dict(row.value) if row and isinstance(row.value, dict) else None


async def set_setting(session: AsyncSession, key: str, value: dict[str, Any]) -> None:
    row = await session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=value))
    else:
        row.value = value
        row.updated_at = utcnow()
        session.add(row)
    await session.flush()


# --------------------------------------------------------------------------- #
# Learned approval patterns
# --------------------------------------------------------------------------- #
async def record_approval_pattern(
    session: AsyncSession, signature: str, *, approved: bool
) -> ApprovalPattern:
    """Remember one owner decision about a shape of action."""
    row = await session.scalar(
        select(ApprovalPattern).where(ApprovalPattern.signature == signature)
    )
    if row is None:
        # Column defaults are applied by the database at INSERT, so a freshly
        # constructed row still has None counters until it is flushed.
        row = ApprovalPattern(signature=signature, approved_count=0, rejected_count=0)
        session.add(row)
    if approved:
        row.approved_count += 1
        row.last_decision = "approved"
    else:
        # A refusal resets earned trust: the agent must re-earn it.
        row.rejected_count += 1
        row.approved_count = 0
        row.last_decision = "rejected"
    await session.flush()
    return row


async def approval_stats(session: AsyncSession, signature: str) -> dict[str, int]:
    row = await session.scalar(
        select(ApprovalPattern).where(ApprovalPattern.signature == signature)
    )
    if row is None:
        return {"approved": 0, "rejected": 0}
    return {"approved": row.approved_count, "rejected": row.rejected_count}


async def list_approval_patterns(
    session: AsyncSession, limit: int = 50
) -> list[ApprovalPattern]:
    stmt = (
        select(ApprovalPattern)
        .order_by(ApprovalPattern.approved_count.desc(), ApprovalPattern.updated_at.desc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())


async def forget_approval_pattern(session: AsyncSession, signature: str) -> bool:
    row = await session.scalar(
        select(ApprovalPattern).where(ApprovalPattern.signature == signature)
    )
    if row is None:
        return False
    await session.delete(row)
    return True


# --------------------------------------------------------------------------- #
# Chat sessions
# --------------------------------------------------------------------------- #
async def get_session(session: AsyncSession, chat_id: int) -> ChatSession | None:
    return await session.get(ChatSession, chat_id)


async def ensure_session(session: AsyncSession, chat_id: int) -> ChatSession:
    row = await session.get(ChatSession, chat_id)
    if row is None:
        row = ChatSession(chat_id=chat_id)
        session.add(row)
        await session.flush()
    return row


async def update_session(session: AsyncSession, chat_id: int, **values: Any) -> None:
    values.setdefault("last_active_at", utcnow())
    await session.execute(
        update(ChatSession).where(ChatSession.chat_id == chat_id).values(**values)
    )


async def reset_session(session: AsyncSession, chat_id: int) -> None:
    """Start a fresh thread: forget the active task and recent turns."""
    await session.execute(
        update(ChatSession)
        .where(ChatSession.chat_id == chat_id)
        .values(
            active_task_id=None, turn_count=0, title="", context={},
            started_at=utcnow(), last_active_at=utcnow(),
        )
    )
    await session.execute(delete(Conversation).where(Conversation.chat_id == chat_id))


async def latest_task_for_chat(session: AsyncSession, chat_id: int) -> Task | None:
    stmt = (
        select(Task)
        .where(Task.chat_id == chat_id)
        .order_by(Task.created_at.desc())
        .limit(1)
    )
    return (await session.scalars(stmt)).first()


# --------------------------------------------------------------------------- #
# Credentials (encrypted; see app.security.vault)
# --------------------------------------------------------------------------- #
async def list_credentials(session: AsyncSession) -> list[Credential]:
    return list((await session.scalars(select(Credential))).all())


async def set_credential(session: AsyncSession, *, name: str, value: str) -> None:
    row = await session.get(Credential, name)
    if row is None:
        session.add(Credential(name=name, value=value))
    else:
        row.value = value
        row.updated_at = utcnow()
        session.add(row)
    await session.flush()


async def delete_credential(session: AsyncSession, name: str) -> bool:
    result = await session.execute(delete(Credential).where(Credential.name == name))
    return bool(result.rowcount)


# --------------------------------------------------------------------------- #
# Inbound messages (WhatsApp / Teams)
# --------------------------------------------------------------------------- #
async def record_inbound(
    session: AsyncSession,
    *,
    channel: str,
    external_id: str,
    chat: str = "",
    sender: str = "",
    sender_name: str = "",
    text: str = "",
    media_path: str = "",
    data: dict[str, Any] | None = None,
) -> InboundMessage | None:
    """Store an inbound message; returns None if it was already recorded."""
    existing = (
        await session.scalars(
            select(InboundMessage).where(
                InboundMessage.channel == channel,
                InboundMessage.external_id == external_id,
            )
        )
    ).first()
    if existing is not None:
        return None

    message = InboundMessage(
        id=new_id(),
        channel=channel,
        external_id=external_id,
        chat=chat[:200],
        sender=sender[:200],
        sender_name=sender_name[:200],
        text=text[:8000],
        media_path=media_path[:500],
        data=data or {},
    )
    session.add(message)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return None
    return message


async def list_inbound(
    session: AsyncSession,
    *,
    channel: str | None = None,
    chat: str | None = None,
    unhandled_only: bool = False,
    limit: int = 20,
) -> list[InboundMessage]:
    stmt = select(InboundMessage).order_by(InboundMessage.created_at.desc()).limit(limit)
    if channel:
        stmt = stmt.where(InboundMessage.channel == channel)
    if chat:
        stmt = stmt.where(InboundMessage.chat == chat)
    if unhandled_only:
        stmt = stmt.where(InboundMessage.handled.is_(False))
    return list((await session.scalars(stmt)).all())


async def mark_inbound_handled(
    session: AsyncSession, message_id: str, task_id: str | None = None
) -> None:
    await session.execute(
        update(InboundMessage)
        .where(InboundMessage.id == message_id)
        .values(handled=True, task_id=task_id)
    )


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
async def stats(session: AsyncSession) -> dict[str, Any]:
    counts = await count_tasks_by_status(session)
    active = sum(counts.get(s.value, 0) for s in ACTIVE_STATUSES)
    pending_approvals = len(await list_pending_approvals(session, limit=100))
    jobs = await session.scalar(
        select(func.count()).select_from(ScheduledJob).where(ScheduledJob.enabled.is_(True))
    )
    memories = await session.scalar(select(func.count()).select_from(MemoryEntry))
    return {
        "tasks_by_status": counts,
        "active_tasks": active,
        "pending_approvals": pending_approvals,
        "enabled_jobs": int(jobs or 0),
        "memory_entries": int(memories or 0),
    }


# --------------------------------------------------------------------------- #
# Contacts / members (Telegram + WhatsApp people the agent has seen)
# --------------------------------------------------------------------------- #
async def upsert_contact(
    session: AsyncSession,
    *,
    channel: str,
    external_id: str,
    display_name: str,
    username_or_phone: str = "",
    seen_in: list[str] | None = None,
) -> Contact:
    """Insert or refresh a contact, keyed on (channel, external_id).

    A re-sync updates ``last_seen_at``/``display_name`` and unions ``seen_in``
    rather than creating a duplicate row.
    """
    existing = (
        await session.scalars(
            select(Contact).where(
                Contact.channel == channel, Contact.external_id == external_id
            )
        )
    ).first()
    if existing:
        existing.display_name = display_name or existing.display_name
        if username_or_phone:
            existing.username_or_phone = username_or_phone
        if seen_in:
            merged = list(dict.fromkeys(list(existing.seen_in or []) + list(seen_in)))
            existing.seen_in = merged
        existing.last_seen_at = utcnow()
        session.add(existing)
        await session.flush()
        return existing

    contact = Contact(
        id=new_id(),
        channel=channel,
        external_id=external_id,
        display_name=display_name,
        username_or_phone=username_or_phone or None,
        seen_in=list(seen_in or []),
    )
    session.add(contact)
    try:
        await session.flush()
    except IntegrityError:
        # Lost a race with another sync writing the same contact concurrently.
        await session.rollback()
        return await upsert_contact(
            session,
            channel=channel,
            external_id=external_id,
            display_name=display_name,
            username_or_phone=username_or_phone,
            seen_in=seen_in,
        )
    return contact


async def list_contacts(
    session: AsyncSession, channel: str | None = None, limit: int = 100
) -> list[Contact]:
    stmt = (
        select(Contact)
        .order_by(Contact.last_seen_at.desc(), Contact.id.desc())
        .limit(limit)
    )
    if channel:
        stmt = stmt.where(Contact.channel == channel)
    return list((await session.scalars(stmt)).all())


async def search_contacts(session: AsyncSession, query: str, limit: int = 20) -> list[Contact]:
    like = f"%{query.lower()}%"
    stmt = (
        select(Contact)
        .where(
            or_(
                func.lower(Contact.display_name).like(like),
                func.lower(Contact.username_or_phone).like(like),
            )
        )
        .order_by(Contact.last_seen_at.desc(), Contact.id.desc())
        .limit(limit)
    )
    return list((await session.scalars(stmt)).all())
