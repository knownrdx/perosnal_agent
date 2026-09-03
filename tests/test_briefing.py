"""Daily briefing: content, time window, caps and LLM resilience.

The load-bearing test here is ``test_render_falls_back_when_llm_raises``: a
broken model must degrade the briefing's prose, never its delivery.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.agent.briefing import (
    MAX_ITEMS_PER_SECTION,
    Briefing,
    collect,
    daily_briefing,
    render,
)
from app.db import repo
from app.db.base import session_scope
from app.db.models import JobKind, TaskStatus, utcnow
from app.llm import LLMError
from app.llm.base import LLMResponse


class FakeLLM:
    """Minimal LLMClient stand-in: returns a fixed line, or explodes."""

    def __init__(self, reply: str = "All quiet.", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[Any]] = []

    async def chat(
        self, messages: list[Any], *, temperature: float | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.reply, model="fake")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _task(
    request: str,
    *,
    status: TaskStatus | None = None,
    result: str = "",
    error: str = "",
    chat_id: int | None = 42,
    age_hours: float = 0.0,
) -> str:
    """Create one task in a given state, optionally aged into the past."""
    async with session_scope() as session:
        task = await repo.create_task(session, user_request=request, chat_id=chat_id)
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status.value
        if result:
            values["result"] = result
        if error:
            values["error"] = error
        if age_hours:
            stamp = utcnow() - timedelta(hours=age_hours)
            values.update(created_at=stamp, updated_at=stamp, completed_at=stamp)
        if values:
            await repo.update_task(session, task.id, **values)
            # update_task stamps updated_at=now; re-apply the backdate on top.
            if age_hours:
                stamp = utcnow() - timedelta(hours=age_hours)
                await repo.update_task(
                    session, task.id, created_at=stamp, updated_at=stamp, completed_at=stamp
                )
        return task.id


# --------------------------------------------------------------------------- #
# Empty database
# --------------------------------------------------------------------------- #
async def test_empty_database_says_nothing_happened(environment):
    briefing = await collect()
    assert briefing.is_empty()

    text = briefing.as_text()
    assert "nothing" in text.lower()
    assert len(text.splitlines()) == 1, "an empty period must not send a skeleton"
    for heading in ("Completed", "Failed", "Needs you", "Coming up"):
        assert heading not in text


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
async def test_completed_tasks_are_reported(environment):
    await _task("send the weekly invoice", status=TaskStatus.COMPLETED, result="invoice sent")

    briefing = await collect()
    assert not briefing.is_empty()
    assert len(briefing.completed) == 1
    assert briefing.completed[0]["title"] == "send the weekly invoice"
    assert briefing.completed[0]["result_summary"] == "invoice sent"

    text = briefing.as_text()
    assert "Completed (1)" in text
    assert "send the weekly invoice" in text


async def test_failed_tasks_include_the_error(environment):
    await _task(
        "download the portal report",
        status=TaskStatus.FAILED,
        error="network error: name resolution failed",
    )

    briefing = await collect()
    assert len(briefing.failed) == 1
    assert "name resolution failed" in briefing.failed[0]["error"]
    assert "name resolution failed" in briefing.as_text()


async def test_waiting_for_user_tasks_are_surfaced(environment):
    await _task("confirm the flight booking", status=TaskStatus.WAITING_FOR_USER)

    briefing = await collect()
    assert len(briefing.waiting) == 1
    assert briefing.waiting[0]["title"] == "confirm the flight booking"
    assert briefing.running == []
    assert "Needs you (1)" in briefing.as_text()


async def test_pending_approval_counts_as_waiting(environment):
    task_id = await _task("wire the money", status=TaskStatus.RUNNING)
    async with session_scope() as session:
        await repo.create_approval(
            session,
            task_id=task_id,
            tool="shell_run",
            args={"cmd": "rm -rf"},
            reason="destructive command",
        )

    briefing = await collect()
    assert [item["id"] for item in briefing.waiting] == [task_id]
    assert briefing.running == [], "a task blocked on approval is not silently running"
    assert "destructive command" in briefing.as_text()


async def test_running_tasks_are_listed_separately(environment):
    await _task("index the archive", status=TaskStatus.RUNNING)

    briefing = await collect()
    assert len(briefing.running) == 1
    assert briefing.waiting == []
    assert "Still running (1)" in briefing.as_text()


async def test_attention_sections_come_before_good_news(environment):
    await _task("approve the draft", status=TaskStatus.WAITING_FOR_USER)
    await _task("broken job", status=TaskStatus.FAILED, error="boom")
    await _task("finished job", status=TaskStatus.COMPLETED, result="ok")

    text = (await collect()).as_text()
    assert text.index("Needs you") < text.index("Failed") < text.index("Completed")


async def test_upcoming_jobs_within_24h(environment):
    async with session_scope() as session:
        await repo.create_job(
            session,
            name="morning digest",
            kind=JobKind.CRON.value,
            instruction="send the digest",
            chat_id=42,
            user_id=42,
            next_run_at=utcnow() + timedelta(hours=3),
            cron_expr="0 8 * * *",
        )
        await repo.create_job(
            session,
            name="next week cleanup",
            kind=JobKind.ONCE.value,
            instruction="clean up",
            chat_id=42,
            user_id=42,
            next_run_at=utcnow() + timedelta(days=7),
        )

    briefing = await collect()
    names = [job["name"] for job in briefing.upcoming]
    assert names == ["morning digest"], "only the next 24h belongs in a daily briefing"


async def test_new_memories_are_reported_as_learned(environment):
    async with session_scope() as session:
        await repo.memory_store(session, key="invoice_format", value="invoices go out as PDF")

    briefing = await collect()
    assert any("invoice_format" in line for line in briefing.learned)
    assert "invoices go out as PDF" in briefing.as_text()


# --------------------------------------------------------------------------- #
# Time window
# --------------------------------------------------------------------------- #
async def test_old_tasks_are_excluded(environment):
    await _task("ancient history", status=TaskStatus.COMPLETED, result="old", age_hours=72)
    await _task("todays work", status=TaskStatus.COMPLETED, result="new")

    briefing = await collect(period_hours=24)
    titles = [item["title"] for item in briefing.completed]
    assert titles == ["todays work"]
    assert "ancient history" not in briefing.as_text()


async def test_widening_the_period_includes_older_tasks(environment):
    await _task("two days ago", status=TaskStatus.COMPLETED, result="old", age_hours=48)

    assert (await collect(period_hours=24)).completed == []
    wide = await collect(period_hours=96)
    assert [item["title"] for item in wide.completed] == ["two days ago"]
    assert wide.period_hours == 96


async def test_old_waiting_task_is_still_reported(environment):
    """Age never hides a blocker: three days stuck is worse than one hour."""
    await _task("sign the contract", status=TaskStatus.WAITING_FOR_USER, age_hours=72)

    briefing = await collect(period_hours=24)
    assert len(briefing.waiting) == 1


async def test_other_chats_are_excluded_when_filtering(environment):
    await _task("mine", status=TaskStatus.COMPLETED, chat_id=42)
    await _task("someone elses", status=TaskStatus.COMPLETED, chat_id=99)

    briefing = await collect(chat_id=42)
    assert [item["title"] for item in briefing.completed] == ["mine"]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
async def test_sections_are_capped_with_a_more_line(environment):
    for index in range(8):
        await _task(f"job number {index}", status=TaskStatus.COMPLETED, result="ok")

    briefing = await collect()
    assert len(briefing.completed) == 8

    text = briefing.as_text()
    assert "Completed (8)" in text
    assert text.count("job number") == MAX_ITEMS_PER_SECTION
    assert "+3 more" in text


async def test_long_titles_are_truncated(environment):
    long_title = "remember to " + "very " * 40 + "long request"
    await _task(long_title, status=TaskStatus.COMPLETED, result="ok")

    briefing = await collect()
    rendered = briefing.completed[0]["title"]
    assert len(rendered) <= 70
    assert rendered.endswith("\u2026")
    assert max(len(line) for line in briefing.as_text().splitlines()) < 260


def test_as_text_is_ascii_safe_plain_text():
    briefing = Briefing(
        period_hours=24,
        completed=[{"id": "a", "title": "did a thing", "result_summary": "ok"}],
    )
    text = briefing.as_text()
    assert "|" not in text and "**" not in text and "`" not in text


# --------------------------------------------------------------------------- #
# Rendering with an LLM
# --------------------------------------------------------------------------- #
async def test_render_without_llm_is_deterministic(environment):
    await _task("a task", status=TaskStatus.COMPLETED, result="ok")
    briefing = await collect()
    assert await render(briefing) == briefing.as_text()


async def test_render_with_llm_prepends_the_summary(environment):
    await _task("pay the invoice", status=TaskStatus.WAITING_FOR_USER)
    briefing = await collect()
    llm = FakeLLM(reply="One task is waiting on your approval.")

    text = await render(briefing, llm=llm)
    assert text.startswith("One task is waiting on your approval.")
    assert briefing.as_text() in text, "the structured sections must survive"
    assert llm.calls, "the model should have been asked"
    assert "pay the invoice" in llm.calls[0][-1].content


async def test_render_falls_back_when_llm_raises(environment):
    """A dead model must cost prose, not the briefing."""
    await _task("critical thing", status=TaskStatus.FAILED, error="disk full")
    briefing = await collect()

    text = await render(briefing, llm=FakeLLM(error=LLMError("model offline")))
    assert text == briefing.as_text()
    assert "disk full" in text


async def test_render_survives_an_unexpected_llm_crash(environment):
    await _task("thing", status=TaskStatus.COMPLETED, result="ok")
    briefing = await collect()

    text = await render(briefing, llm=FakeLLM(error=RuntimeError("boom")))
    assert text == briefing.as_text()


async def test_render_ignores_an_empty_model_reply(environment):
    await _task("thing", status=TaskStatus.COMPLETED, result="ok")
    briefing = await collect()

    assert await render(briefing, llm=FakeLLM(reply="   ")) == briefing.as_text()


async def test_render_skips_the_llm_when_nothing_happened(environment):
    briefing = await collect()
    llm = FakeLLM()

    text = await render(briefing, llm=llm)
    assert llm.calls == [], "do not spend a model call on an empty day"
    assert "nothing" in text.lower()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
async def test_daily_briefing_end_to_end(environment):
    await _task("shipped the report", status=TaskStatus.COMPLETED, result="sent to owner")
    await _task("needs a decision", status=TaskStatus.WAITING_FOR_USER)

    text = await daily_briefing(chat_id=42)
    assert text.strip()
    assert "shipped the report" in text
    assert "needs a decision" in text


async def test_daily_briefing_with_llm_narrative(environment):
    await _task("did the thing", status=TaskStatus.COMPLETED, result="done")

    text = await daily_briefing(chat_id=42, llm=FakeLLM(reply="Quiet day, one task done."))
    assert text.startswith("Quiet day, one task done.")


async def test_daily_briefing_on_empty_database(environment):
    text = await daily_briefing()
    assert text.strip()
    assert "nothing" in text.lower()
