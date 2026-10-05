"""Self-learning gaps found in review.

Each section pins one behaviour the learning loop got wrong:

1. Bengali-script requests matched no memory (tokeniser was Latin-only).
2. Skill summaries were ranked like any lesson and cut to 300 characters.
3. Failure rules were re-saved after every task and never expired.
4. Skills were re-synthesised (one LLM call per topic) after every task.
5. Slow learning ran inside the task's timeout and could flip a finished
   task to FAILED.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

from sqlalchemy import select, update

from app.agent.learning import (
    _tokens,
    learn_from_failures,
    relevant_memories,
)
from app.agent.skills import synthesize_skills
from app.db import repo
from app.db.base import session_scope
from app.db.models import MemoryEntry, TaskStatus, ToolCall, ToolCallStatus, utcnow
from app.llm import LLMError, Message


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
async def _memory(key: str) -> MemoryEntry | None:
    async with session_scope() as session:
        return await session.scalar(select(MemoryEntry).where(MemoryEntry.key == key))


async def _age_memory(key: str, days: float) -> None:
    async with session_scope() as session:
        await session.execute(
            update(MemoryEntry)
            .where(MemoryEntry.key == key)
            .values(updated_at=utcnow() - timedelta(days=days))
        )


async def _record_failures(count: int, *, age_days: float = 0.0) -> None:
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="flaky download")
        for step in range(count):
            await repo.record_tool_call(
                session, task_id=task.id, step=step, tool="file_download",
                args={"url": f"https://x.com/{step}"}, status=ToolCallStatus.ERROR,
                error="network error: name resolution failed for 'host'",
            )
        if age_days:
            await session.execute(
                update(ToolCall)
                .where(ToolCall.task_id == task.id)
                .values(created_at=utcnow() - timedelta(days=age_days))
            )


class CountingLLM:
    """Synthesis stub that counts calls; ``summary=None`` simulates an outage."""

    def __init__(self, summary: str | None = "Reports go out as PDF.") -> None:
        self.summary = summary
        self.calls = 0

    async def chat_json(self, messages: list[Message], *, temperature=None):
        self.calls += 1
        if self.summary is None:
            raise LLMError("simulated model outage")
        return {"summary": self.summary}


async def _seed_lessons(prefix: str, count: int, *, start: int = 0) -> None:
    async with session_scope() as session:
        for i in range(start, start + count):
            await repo.memory_store(
                session, key=f"{prefix}_{i}", value=f"{prefix} lesson number {i}",
                kind="workflow", tags=["learned"],
            )


# --------------------------------------------------------------------------- #
# 1. Bengali script reaches memories
# --------------------------------------------------------------------------- #
def test_tokeniser_reads_bengali_script():
    tokens = _tokens("হোয়াটসঅ্যাপ স্টক দেখাও")
    assert {"whatsapp", "stock"} <= tokens


def test_tokeniser_drops_banglish_filler():
    tokens = _tokens("amar report ta ekhon pathao koro")
    assert "report" in tokens
    assert not tokens & {"amar", "ekhon", "pathao", "koro"}


async def test_bengali_request_finds_banglish_memory(environment):
    async with session_scope() as session:
        await repo.memory_store(
            session, key="whatsapp_stock",
            value="whatsapp stock files are kept in the bangladesh folder", kind="fact",
        )
        await repo.memory_store(
            session, key="server_ip", value="build server is 10.0.0.5", kind="fact",
        )

    picked = await relevant_memories("হোয়াটসঅ্যাপ স্টক দেখাও", limit=3)
    assert any("whatsapp_stock" in item for item in picked)
    assert not any("server_ip" in item for item in picked)


async def test_memory_written_in_bengali_matches_banglish_request(environment):
    async with session_scope() as session:
        await repo.memory_store(
            session, key="delivery_channel", value="মালিক টেলিগ্রাম এ ফাইল চায়", kind="fact",
        )

    picked = await relevant_memories("file ta telegram e pathao", limit=3)
    assert any("delivery_channel" in item for item in picked)


# --------------------------------------------------------------------------- #
# 2. Skills are ranked up and not truncated to a one-liner
# --------------------------------------------------------------------------- #
async def test_skill_outranks_a_single_lesson_on_its_topic(environment):
    async with session_scope() as session:
        await repo.memory_store(
            session, key="report_deadline",
            value="the report portal times out after 6pm", kind="gotcha",
        )
        await repo.memory_store(
            session, key="skill_report",
            value="Build it from the portal before 6pm, export PDF, send on Telegram.",
            kind="skill",
        )

    picked = await relevant_memories("make the weekly report", limit=1)
    assert picked and picked[0].startswith("[skill] skill_report")


async def test_unrelated_skill_is_not_injected(environment):
    async with session_scope() as session:
        await repo.memory_store(
            session, key="skill_invoice", value="Invoices are PDF, sent monthly.",
            kind="skill",
        )
    assert await relevant_memories("restart the whatsapp bot", limit=6) == []


async def test_skill_survives_past_300_characters(environment):
    body = "Invoice workflow: " + "check totals, attach the PDF, cc finance; " * 16
    value = (body + "ENDMARKER").strip()
    assert 600 < len(value) < 900

    async with session_scope() as session:
        await repo.memory_store(session, key="skill_invoice", value=value, kind="skill")

    picked = await relevant_memories("prepare the invoice", limit=3)
    assert picked and "ENDMARKER" in picked[0]


async def test_injected_memories_stay_bounded(environment):
    from app.agent import learning

    async with session_scope() as session:
        for i in range(6):
            await repo.memory_store(
                session, key=f"skill_invoice_{i}", value="invoice " + "x" * 1990,
                kind="skill",
            )
        for i in range(6):
            await repo.memory_store(
                session, key=f"invoice_note_{i}", value="invoice " + "y" * 490, kind="fact",
            )

    picked = await relevant_memories("invoice", limit=6)
    assert picked
    assert sum(len(item) for item in picked) <= learning.MEMORY_CONTEXT_CHARS
    assert all(len(item) <= learning.SKILL_CHARS for item in picked)
    assert all(len(item) <= learning.ENTRY_CHARS for item in picked if "[fact]" in item)


# --------------------------------------------------------------------------- #
# 3. Failure rules: written on news only, and they expire
# --------------------------------------------------------------------------- #
async def test_failure_rule_is_not_rewritten_without_new_failures(environment):
    # Failures 3 days ago, rule written 2 days ago: a pinned, comparable past.
    await _record_failures(4, age_days=3)
    first = await learn_from_failures()
    assert first
    key = first[0]["key"]

    await _age_memory(key, days=2)
    before = (await _memory(key)).updated_at

    assert await learn_from_failures() == [], "nothing new happened"
    assert (await _memory(key)).updated_at == before, "timestamp must not be refreshed"


async def test_stale_failure_rule_stops_being_injected(environment):
    await _record_failures(4, age_days=15)
    # A rule that existed back then (written by an older run).
    async with session_scope() as session:
        await repo.memory_store(
            session, key="avoid_file_download_network_error",
            value="file_download keeps failing with: network error (seen 4x)",
            kind="gotcha", tags=["learned", "failure"],
        )
    await _age_memory("avoid_file_download_network_error", days=15)

    picked = await relevant_memories("download the file", limit=6)
    assert not any("avoid_file_download" in item for item in picked)


async def test_stale_failures_do_not_create_or_revive_a_rule(environment):
    await _record_failures(4, age_days=20)
    assert await learn_from_failures() == [], "20-day-old failures are not news"

    picked = await relevant_memories("download the file", limit=6)
    assert not any("avoid_file_download" in item for item in picked)


async def test_fresh_failure_revives_an_expired_rule(environment):
    await _record_failures(4)
    first = await learn_from_failures()
    key = first[0]["key"]
    await _age_memory(key, days=15)
    assert not any(key in item for item in await relevant_memories("download it", limit=6))

    await _record_failures(1)
    again = await learn_from_failures()
    assert [rule["key"] for rule in again] == [key]
    assert any(key in item for item in await relevant_memories("download it", limit=6))


# --------------------------------------------------------------------------- #
# 4. Skills are only re-synthesised when their lessons changed
# --------------------------------------------------------------------------- #
async def test_unchanged_topic_is_not_resynthesised(environment):
    await _seed_lessons("report", 3)
    first = CountingLLM()
    assert len(await synthesize_skills(llm=first)) == 1
    assert first.calls == 1

    second = CountingLLM()
    assert await synthesize_skills(llm=second) == []
    assert second.calls == 0, "no lesson changed, so no LLM call"


async def test_only_the_changed_topic_is_resynthesised(environment):
    await _seed_lessons("report", 3)
    await _seed_lessons("invoice", 3)
    await synthesize_skills(llm=CountingLLM())

    await _seed_lessons("report", 1, start=3)          # new member
    llm = CountingLLM(summary="updated report skill")
    written = await synthesize_skills(llm=llm)
    assert llm.calls == 1 and [w["topic"] for w in written] == ["report"]

    async with session_scope() as session:            # edited member
        await repo.memory_store(
            session, key="invoice_0", value="invoices now go out in USD", kind="workflow",
        )
    llm = CountingLLM(summary="updated invoice skill")
    written = await synthesize_skills(llm=llm)
    assert llm.calls == 1 and [w["topic"] for w in written] == ["invoice"]


async def test_topics_beyond_the_per_run_cap_get_their_turn(environment):
    from app.agent import skills

    topics = [f"topic{chr(97 + i)}" for i in range(skills.MAX_CLUSTERS_PER_RUN + 2)]
    for topic in topics:
        await _seed_lessons(topic, 3)

    first = CountingLLM()
    await synthesize_skills(llm=first)
    assert first.calls == skills.MAX_CLUSTERS_PER_RUN

    second = CountingLLM()
    await synthesize_skills(llm=second)
    assert second.calls == 2, "the run after must do the remaining topics, not repeat"


async def test_failure_rules_are_not_folded_into_a_permanent_skill(environment):
    """They expire on their own; a skill_avoid summary would outlive them."""
    async with session_scope() as session:
        for i in range(3):
            await repo.memory_store(
                session, key=f"avoid_tool_{i}", value=f"tool {i} keeps failing",
                kind="gotcha", tags=["learned", "failure"],
            )
    llm = CountingLLM()
    assert await synthesize_skills(llm=llm) == []
    assert llm.calls == 0


async def test_fallback_summary_is_retried_once_the_model_is_back(environment):
    await _seed_lessons("report", 3)
    await synthesize_skills(llm=CountingLLM(summary=None))
    assert "report lesson number" in (await _memory("skill_report")).value

    llm = CountingLLM(summary="Reports go out as PDF.")
    await synthesize_skills(llm=llm)
    assert llm.calls == 1
    assert (await _memory("skill_report")).value == "Reports go out as PDF."

    idle = CountingLLM()
    await synthesize_skills(llm=idle)
    assert idle.calls == 0


# --------------------------------------------------------------------------- #
# 5. Learning cannot change a finished task's outcome
# --------------------------------------------------------------------------- #
class RecordingNotifier:
    def __init__(self, completed_delay: float = 0.0) -> None:
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.completed_delay = completed_delay

    async def task_completed(self, task_id: str) -> None:
        if self.completed_delay:
            await asyncio.sleep(self.completed_delay)
        self.completed.append(task_id)

    async def task_failed(self, task_id: str) -> None:
        self.failed.append(task_id)

    async def task_question(self, task_id: str, question: str) -> None:  # pragma: no cover
        pass


async def _simple_task() -> str:
    async with session_scope() as session:
        task = await repo.create_task(
            session, user_request="write the summary", chat_id=42, max_steps=4
        )
        return task.id


async def test_slow_learning_cannot_fail_a_completed_task(environment, echo_llm, monkeypatch):
    from app.agent import engine as engine_mod
    from app.agent import learning
    from app.agent.engine import AgentEngine
    from app.workers.task_worker import TaskWorker

    async def slow_reflection(task_id, llm=None):
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(learning, "reflect_on_task", slow_reflection)
    monkeypatch.setattr(engine_mod, "LEARNING_TIMEOUT_S", 0.2, raising=False)

    task_id = await _simple_task()
    echo_llm.script = [{"action": "final", "final_answer": "Summary written."}]
    notifier = RecordingNotifier()
    worker = TaskWorker(engine=AgentEngine(llm=echo_llm, notifier=notifier), notifier=notifier)
    worker.settings = worker.settings.model_copy(update={"task_timeout_s": 0.5})

    await worker._run_one(task_id, "test-worker")

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
    assert task.status == TaskStatus.COMPLETED.value, task.error
    assert notifier.completed == [task_id] and notifier.failed == []


async def test_timeout_after_completion_does_not_overwrite_it(environment, echo_llm):
    """Even if the deadline hits while announcing, the result stands."""
    from app.agent.engine import AgentEngine
    from app.workers.task_worker import TaskWorker

    task_id = await _simple_task()
    echo_llm.script = [{"action": "final", "final_answer": "Summary written."}]
    notifier = RecordingNotifier(completed_delay=0.6)
    worker = TaskWorker(engine=AgentEngine(llm=echo_llm, notifier=notifier), notifier=notifier)
    worker.settings = worker.settings.model_copy(update={"task_timeout_s": 0.3})

    await worker._run_one(task_id, "test-worker")
    await asyncio.sleep(0.5)          # let the shielded send finish

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
    assert task.status == TaskStatus.COMPLETED.value
    assert notifier.failed == [], "a completed task must not be announced as failed"


async def test_worker_still_learns_after_the_task(environment, echo_llm):
    from app.agent.engine import AgentEngine
    from app.workers.task_worker import TaskWorker

    task_id = await _simple_task()
    async with session_scope() as session:
        await repo.update_task(session, task_id, current_step=2)
    echo_llm.script = [
        {"action": "final", "final_answer": "Summary written."},
        {"lessons": [{"key": "summary_location",
                      "value": "summaries belong in output/summary.txt", "kind": "workflow"}]},
    ]
    worker = TaskWorker(engine=AgentEngine(llm=echo_llm))
    await worker._run_one(task_id, "test-worker")

    assert await _memory("summary_location") is not None


async def test_learning_errors_and_hangs_are_contained(environment, echo_llm, monkeypatch):
    from app.agent import engine as engine_mod
    from app.agent import learning, skills
    from app.agent.engine import AgentEngine

    async def hang(task_id, llm=None):
        await asyncio.sleep(5)

    async def boom(llm=None):
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(learning, "reflect_on_task", hang)
    monkeypatch.setattr(skills, "synthesize_skills", boom)
    monkeypatch.setattr(engine_mod, "LEARNING_TIMEOUT_S", 0.2, raising=False)

    task_id = await _simple_task()
    echo_llm.script = [{"action": "final", "final_answer": "Summary written."}]
    started = time.monotonic()
    status = await AgentEngine(llm=echo_llm).run_task(task_id)

    assert status == TaskStatus.COMPLETED.value
    assert time.monotonic() - started < 3, "learning must have its own bounded timeout"
