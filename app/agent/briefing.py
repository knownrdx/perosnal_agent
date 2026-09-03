"""Daily briefing.

The owner should not have to ask what the agent did.  Once a day the scheduler
calls :func:`daily_briefing` and sends one message that answers three
questions: what is blocked on me, what broke, and what got done.

Two deliberate design rules:

1. :func:`collect` touches only the database - no LLM, no network.  A dead
   model server, an expired API key or a rate limit must never mean "no
   briefing"; it may only mean "a less pretty briefing".
2. The narrative from :func:`render` is decoration on top of the deterministic
   text, never a replacement for it.  Any LLM failure falls back to
   :meth:`Briefing.as_text`.

Scope of the time window: completed, failed and learned items are *history*
and are limited to ``period_hours``.  Waiting and running tasks are *current
state* and are always reported, however old they are - a task that has been
blocked on the owner for three days is more urgent than one blocked for an
hour, so ageing it out of the briefing would hide exactly the thing the
briefing exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, TaskStatus, utcnow
from app.llm import LLMError, Message
from app.logging_conf import get_logger

log = get_logger(__name__)

# Section sizing: a Telegram message the owner will actually read.
MAX_ITEMS_PER_SECTION = 5
TITLE_LIMIT = 70
DETAIL_LIMIT = 160
NARRATIVE_LIMIT = 700

UPCOMING_WINDOW_HOURS = 24
TASK_SCAN_LIMIT = 200
JOB_SCAN_LIMIT = 100
MEMORY_SCAN_LIMIT = 50

# Plain unicode only - the source file stays ASCII (see project style rules).
BULLET = "\u2022"
DASH = "\u2013"
ELLIPSIS = "\u2026"

_TERMINAL_REPORTED = [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]
_ACTIVE_VALUES = [status.value for status in ACTIVE_STATUSES]

NARRATIVE_SYSTEM = """You write the daily briefing for the owner of a private AI agent.

You are given the facts already collected from the agent's database. Write 2-4
short sentences that a busy person can read in ten seconds.

Rules:
- Lead with anything blocked on the owner or anything that failed.
- Plain text only. No markdown, no bullet points, no headings, no emoji.
- State only what is in the facts. Never invent a task, number or outcome.
- No greeting, no sign-off, no "here is your briefing".
"""


def _aware(value: datetime | None) -> datetime | None:
    """Normalise to timezone-aware UTC (SQLite hands back naive datetimes)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _shorten(text: str, limit: int = TITLE_LIMIT) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + ELLIPSIS


def _task_time(task: Any) -> datetime:
    """When a task last mattered: finish time, else last update, else birth."""
    stamp = _aware(task.completed_at) or _aware(task.updated_at) or _aware(task.created_at)
    return stamp or utcnow()


def _title_of(task: Any) -> str:
    return _shorten(task.title or task.user_request or task.id)


@dataclass
class Briefing:
    """Everything worth telling the owner about one period, already shaped."""

    period_hours: int = 24
    completed: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    waiting: list[dict] = field(default_factory=list)
    running: list[dict] = field(default_factory=list)
    upcoming: list[dict] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utcnow)

    def is_empty(self) -> bool:
        return not (
            self.completed
            or self.failed
            or self.waiting
            or self.running
            or self.upcoming
            or self.learned
        )

    # ------------------------------------------------------------------ #
    def _header(self) -> str:
        unit = "hour" if self.period_hours == 1 else "hours"
        return f"Daily briefing {DASH} last {self.period_hours} {unit}"

    def as_text(self) -> str:
        """The deterministic Telegram message. Always valid, never empty."""
        if self.is_empty():
            return f"{self._header()}: nothing happened and nothing needs you."

        lines: list[str] = [self._header()]

        # Attention first: what the owner must act on, before any good news.
        lines += _section(
            f"Needs you ({len(self.waiting)})",
            self.waiting,
            lambda item: _two_part(item["title"], item.get("reason", "")),
        )
        lines += _section(
            f"Failed ({len(self.failed)})",
            self.failed,
            lambda item: _two_part(item["title"], item.get("error", "")),
        )
        lines += _section(
            f"Completed ({len(self.completed)})",
            self.completed,
            lambda item: _two_part(item["title"], item.get("result_summary", "")),
        )
        lines += _section(
            f"Still running ({len(self.running)})",
            self.running,
            lambda item: _two_part(item["title"], item.get("status", "")),
        )
        lines += _section(
            f"Coming up ({len(self.upcoming)})",
            self.upcoming,
            lambda item: _two_part(item["name"], item.get("when", "")),
        )
        lines += _section(f"Learned ({len(self.learned)})", self.learned, _shorten)
        return "\n".join(lines)


def _two_part(head: str, tail: str) -> str:
    """'title - detail', or just the title when there is no detail."""
    tail = _shorten(tail, DETAIL_LIMIT)
    if not tail:
        return head
    return f"{head} {DASH} {tail}"


def _section(heading: str, items: list, fmt) -> list[str]:
    """Render one capped section, or nothing at all when it is empty."""
    if not items:
        return []
    lines = ["", f"{heading}:"]
    for item in items[:MAX_ITEMS_PER_SECTION]:
        lines.append(f"{BULLET} {fmt(item)}")
    hidden = len(items) - MAX_ITEMS_PER_SECTION
    if hidden > 0:
        lines.append(f"  +{hidden} more")
    return lines


# --------------------------------------------------------------------------- #
# Collection (database only)
# --------------------------------------------------------------------------- #
async def collect(period_hours: int = 24, chat_id: int | None = None) -> Briefing:
    """Gather the briefing facts. Pure database access: cannot fail on the LLM."""
    period_hours = max(1, int(period_hours))
    now = utcnow()
    cutoff = now - timedelta(hours=period_hours)
    horizon = now + timedelta(hours=UPCOMING_WINDOW_HOURS)

    briefing = Briefing(period_hours=period_hours, generated_at=now)

    async with session_scope() as session:
        finished = await repo.list_tasks(
            session, statuses=_TERMINAL_REPORTED, chat_id=chat_id, limit=TASK_SCAN_LIMIT
        )
        for task in finished:
            if _task_time(task) < cutoff:
                continue
            if task.status == TaskStatus.COMPLETED.value:
                briefing.completed.append(
                    {
                        "id": task.id,
                        "title": _title_of(task),
                        "result_summary": _shorten(task.result, DETAIL_LIMIT),
                    }
                )
            else:
                briefing.failed.append(
                    {
                        "id": task.id,
                        "title": _title_of(task),
                        "error": _shorten(task.error, DETAIL_LIMIT),
                    }
                )

        active = await repo.list_tasks(
            session, statuses=_ACTIVE_VALUES, chat_id=chat_id, limit=TASK_SCAN_LIMIT
        )
        for task in active:
            approval = await repo.pending_approval_for_task(session, task.id)
            if approval is not None:
                reason = approval.reason or f"approval needed for {approval.tool}"
                briefing.waiting.append(
                    {
                        "id": task.id,
                        "title": _title_of(task),
                        "reason": _shorten(f"approval: {reason}", DETAIL_LIMIT),
                        "status": task.status,
                    }
                )
            elif task.status == TaskStatus.WAITING_FOR_USER.value:
                briefing.waiting.append(
                    {
                        "id": task.id,
                        "title": _title_of(task),
                        "reason": "waiting for your reply",
                        "status": task.status,
                    }
                )
            else:
                briefing.running.append(
                    {
                        "id": task.id,
                        "title": _title_of(task),
                        "status": task.status,
                    }
                )

        jobs = await repo.list_jobs(session, limit=JOB_SCAN_LIMIT)
        for job in jobs:
            if not job.enabled or job.next_run_at is None:
                continue
            if chat_id is not None and job.chat_id != chat_id:
                continue
            due = _aware(job.next_run_at)
            if due is None or due > horizon:
                continue
            briefing.upcoming.append(
                {
                    "id": job.id,
                    "name": _shorten(job.name or job.instruction),
                    "when": due.strftime("%Y-%m-%d %H:%M UTC"),
                    "next_run_at": due,
                }
            )

        for entry in await repo.memory_recent(session, limit=MEMORY_SCAN_LIMIT):
            created = _aware(entry.created_at)
            if created is None or created < cutoff:
                continue
            briefing.learned.append(f"{entry.key}: {entry.value}")

    # Longest-blocked first: the oldest blocker is the one most likely forgotten.
    # ``list_tasks`` returns newest-first, so reversing the appends orders the
    # waiting list by age without a second query.
    briefing.waiting.reverse()
    briefing.upcoming.sort(key=lambda item: item["next_run_at"])
    return briefing


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _facts_for_llm(briefing: Briefing) -> str:
    return briefing.as_text()[:4000]


async def render(briefing: Briefing, *, llm: Any | None = None) -> str:
    """Structured briefing, optionally introduced by a short LLM narrative.

    The narrative is best-effort: any model failure (or an empty reply) is
    logged and dropped, and the owner still gets the deterministic text.
    """
    body = briefing.as_text()
    if llm is None or briefing.is_empty():
        return body

    try:
        response = await llm.chat(
            [Message("system", NARRATIVE_SYSTEM), Message("user", _facts_for_llm(briefing))]
        )
        narrative = " ".join(str(getattr(response, "content", "") or "").split())
    except LLMError as exc:
        log.warning("briefing_narrative_failed", extra={"error": str(exc)[:200]})
        return body
    except Exception as exc:  # noqa: BLE001 - a briefing must survive any model bug
        log.warning("briefing_narrative_error", extra={"error": str(exc)[:200]})
        return body

    if not narrative:
        return body
    return f"{narrative[:NARRATIVE_LIMIT]}\n\n{body}"


async def daily_briefing(
    chat_id: int | None = None,
    period_hours: int = 24,
    llm: Any | None = None,
) -> str:
    """Collect and render in one call - the entry point for the scheduler."""
    briefing = await collect(period_hours=period_hours, chat_id=chat_id)
    return await render(briefing, llm=llm)


__all__ = ["Briefing", "collect", "daily_briefing", "render"]
