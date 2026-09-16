"""Persistent state model.

Everything the agent must survive a restart with lives here:
tasks, their tool calls, audit events, approvals, memory, scheduled jobs and
the idempotency ledger.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend.

    PostgreSQL keeps the offset; SQLite does not.  This decorator normalises
    both directions so application code never compares naive to aware values.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_FOR_EXTERNAL_EVENT = "WAITING_FOR_EXTERNAL_EVENT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
ACTIVE_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.RUNNING,
    TaskStatus.WAITING,
    TaskStatus.WAITING_FOR_USER,
    TaskStatus.WAITING_FOR_EXTERNAL_EVENT,
}


class ToolCallStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    DENIED = "DENIED"
    TIMEOUT = "TIMEOUT"
    PENDING_APPROVAL = "PENDING_APPROVAL"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class JobKind(str, Enum):
    ONCE = "ONCE"
    INTERVAL = "INTERVAL"
    CRON = "CRON"


class FailureKind(str, Enum):
    TEMPORARY = "TEMPORARY"
    PERMANENT = "PERMANENT"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_request: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(32), default=TaskStatus.PENDING.value, index=True)

    chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    permission: Mapped[str] = mapped_column(String(16), default="WRITE")

    current_step: Mapped[int] = mapped_column(Integer, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, default=14)
    plan: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    failure_kind: Mapped[str] = mapped_column(String(32), default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)

    output_files: Mapped[list] = mapped_column(JSON, default=list)
    context: Mapped[dict] = mapped_column(JSON, default=dict)

    parent_task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scheduled_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # worker lease -> lets another worker safely reclaim a crashed task
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    run_after: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True, index=True)

    notified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_tasks_status_created", "status", "created_at"),)

    def short(self) -> dict:
        return {
            "id": self.id,
            "title": self.title or self.user_request[:80],
            "status": self.status,
            "step": self.current_step,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    tool: Mapped[str] = mapped_column(String(64), index=True)
    args: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default=ToolCallStatus.OK.value)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    failure_kind: Mapped[str] = mapped_column(String(32), default="")
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    operation_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    task: Mapped[Task] = relationship(back_populates="tool_calls")


class Event(Base):
    """Append-only audit log."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    level: Mapped[str] = mapped_column(String(16), default="INFO")
    event: Mapped[str] = mapped_column(String(128), index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(String(32), index=True)
    tool: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict] = mapped_column(JSON, default=dict)
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default=ApprovalStatus.PENDING.value, index=True)
    decided_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class MemoryEntry(Base):
    """Long-term, non-sensitive memory (keyword search, no vector DB in V1)."""

    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(200), index=True)
    value: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="fact", index=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    source_task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (UniqueConstraint("key", name="uq_memory_key"),)


class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), default="")
    kind: Mapped[str] = mapped_column(String(16), default=JobKind.ONCE.value)
    instruction: Mapped[str] = mapped_column(Text)
    chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    cron_expr: Mapped[str] = mapped_column(String(120), default="")
    interval_s: Mapped[int | None] = mapped_column(Integer, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True, index=True
    )
    last_run_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    runs: Mapped[int] = mapped_column(Integer, default=0)
    max_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Operation(Base):
    """Idempotency ledger.

    Before performing an externally visible side effect (sending a Telegram
    message/file, writing an output artefact) the executor records the
    operation key here.  A crash-restart therefore cannot repeat the action.
    """

    __tablename__ = "operations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(64), default="")
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class AppSetting(Base):
    """Small key/value store for runtime settings (e.g. active LLM)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )


class ApprovalPattern(Base):
    """What the owner has decided about a *shape* of action, over time.

    Lets the agent stop asking about things the owner keeps approving, while a
    single rejection immediately revokes that trust.
    """

    __tablename__ = "approval_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    approved_count: Mapped[int] = mapped_column(Integer, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, default=0)
    last_decision: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class ChatSession(Base):
    """One conversation thread per Telegram chat.

    Separates *chatting* from *tasks*: a message may be a quick question
    (answered inline), a new job (creates a Task), or a follow-up to work that
    is already running (attaches to the existing Task).
    """

    __tablename__ = "chat_sessions"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    # auto | chat | task  -> how new messages are interpreted
    mode: Mapped[str] = mapped_column(String(16), default="auto")
    active_task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_thread_id: Mapped[str] = mapped_column(String(40), default="main")
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )


class Credential(Base):
    """Runtime secrets set by the owner (encrypted at rest).

    Only the ciphertext ever touches the database; decryption happens in
    app.security.vault using a key that is not stored here.
    """

    __tablename__ = "credentials"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )


class InboundMessage(Base):
    """A message received from WhatsApp / Teams via a bridge."""

    __tablename__ = "inbound_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    channel: Mapped[str] = mapped_column(String(16), index=True)   # whatsapp | teams
    external_id: Mapped[str] = mapped_column(String(200), index=True)
    chat: Mapped[str] = mapped_column(String(200), default="")
    sender: Mapped[str] = mapped_column(String(200), default="", index=True)
    sender_name: Mapped[str] = mapped_column(String(200), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    media_path: Mapped[str] = mapped_column(String(500), default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    handled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("channel", "external_id", name="uq_inbound_channel_external"),
    )


class Conversation(Base):
    """Short-term memory: recent chat turns per Telegram chat.

    ``thread_id`` partitions a chat_id's history into separate conversation
    threads (ChatGPT-style "New Chat"). Every existing row before this field
    was added belongs to the implicit "main" thread. A message is never
    deleted when a new thread starts - old threads stay fully readable.
    """

    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    thread_id: Mapped[str] = mapped_column(String(40), default="main", index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class Contact(Base):
    """A person the agent has seen across Telegram and WhatsApp.

    Synced from Telegram dialogs (the owner's userbot) and the WhatsApp
    bridge's contact list, so the agent does not need to rediscover who is
    who every time it is asked to message someone.
    """

    __tablename__ = "contacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    channel: Mapped[str] = mapped_column(String(16), index=True)   # telegram | whatsapp
    external_id: Mapped[str] = mapped_column(String(200), index=True)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    username_or_phone: Mapped[str | None] = mapped_column(String(200), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )
    seen_in: Mapped[list] = mapped_column(JSON, default=list)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("channel", "external_id", name="uq_contact_channel_external"),
    )
