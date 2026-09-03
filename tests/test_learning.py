"""Self-learning: reflection, failure rules, relevance retrieval."""

from __future__ import annotations

import pytest

from app.agent.learning import (
    learn_from_failures,
    reflect_on_task,
    relevant_memories,
    score_memory,
    _tokens,
)
from app.db import repo
from app.db.base import session_scope
from app.db.models import TaskStatus, ToolCallStatus


async def _finished_task(request: str, steps: int = 3, status: str = "COMPLETED") -> str:
    async with session_scope() as session:
        task = await repo.create_task(session, user_request=request, chat_id=42)
        await repo.update_task(session, task.id, current_step=steps, status=status)
        await repo.record_tool_call(
            session, task_id=task.id, step=1, tool="file_download",
            args={"url": "https://x.com/a.pdf"}, status=ToolCallStatus.OK,
            result={"path": "downloads/a.pdf"},
        )
        await repo.record_tool_call(
            session, task_id=task.id, step=2, tool="telegram_send_file",
            args={"path": "downloads/a.pdf"}, status=ToolCallStatus.OK,
            result={"message_id": 5},
        )
        return task.id


# --------------------------------------------------------------------------- #
# Reflection
# --------------------------------------------------------------------------- #
async def test_reflection_stores_lessons(environment, echo_llm):
    task_id = await _finished_task("download the weekly report and send it as PDF")
    echo_llm.script = [{
        "lessons": [
            {"key": "report_format", "value": "owner wants reports as PDF",
             "kind": "preference"},
            {"key": "report_source", "value": "weekly reports come from the x.com portal",
             "kind": "fact"},
        ]
    }]

    stored = await reflect_on_task(task_id, llm=echo_llm)
    assert len(stored) == 2

    async with session_scope() as session:
        found = await repo.memory_search(session, "pdf")
    assert found and found[0].kind == "preference"
    assert "learned" in found[0].tags


async def test_reflection_refuses_to_store_secrets(environment, echo_llm):
    task_id = await _finished_task("log into the portal")
    echo_llm.script = [{
        "lessons": [
            {"key": "portal_login", "value": "password = hunter2 for the portal",
             "kind": "fact"},
            {"key": "portal_url", "value": "the portal lives at portal.example.com",
             "kind": "fact"},
        ]
    }]

    stored = await reflect_on_task(task_id, llm=echo_llm)
    keys = {s["key"] for s in stored}
    assert "portal_login" not in keys, "credentials must never be learned"
    assert "portal_url" in keys


async def test_reflection_skips_trivial_tasks(environment, echo_llm):
    task_id = await _finished_task("hi", steps=1)
    echo_llm.script = [{"lessons": [{"key": "x", "value": "y", "kind": "fact"}]}]
    assert await reflect_on_task(task_id, llm=echo_llm) == []


async def test_reflection_caps_lesson_count(environment, echo_llm):
    task_id = await _finished_task("do a big multi step job", steps=6)
    echo_llm.script = [{
        "lessons": [
            {"key": f"lesson_{i}", "value": f"fact number {i}", "kind": "fact"}
            for i in range(10)
        ]
    }]
    stored = await reflect_on_task(task_id, llm=echo_llm)
    assert len(stored) == 3


async def test_reflection_survives_bad_model_output(environment, echo_llm):
    task_id = await _finished_task("something", steps=4)
    echo_llm.script = [{"not_lessons": "garbage"}]
    assert await reflect_on_task(task_id, llm=echo_llm) == []


async def test_failed_task_also_teaches(environment, echo_llm):
    task_id = await _finished_task("broken job", steps=4, status="FAILED")
    echo_llm.script = [{
        "lessons": [{"key": "portal_needs_vpn",
                     "value": "the portal is unreachable without the VPN",
                     "kind": "gotcha"}]
    }]
    stored = await reflect_on_task(task_id, llm=echo_llm)
    assert stored and stored[0]["kind"] == "gotcha"


async def test_engine_learns_after_completing_a_task(environment, echo_llm):
    """End to end: finishing a task writes a lesson without being asked."""
    from app.agent.engine import AgentEngine

    async with session_scope() as session:
        task = await repo.create_task(
            session, user_request="write the summary file", chat_id=42, max_steps=6
        )
        task_id = task.id

    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/summary.txt", "content": "data"}},
        {"action": "tool", "tool": "file_read", "args": {"path": "output/summary.txt"}},
        {"action": "final", "final_answer": "Summary written."},
        # consumed by the reflection pass
        {"lessons": [{"key": "summary_location",
                      "value": "summaries belong in output/summary.txt",
                      "kind": "workflow"}]},
    ]

    await AgentEngine(llm=echo_llm).run_task(task_id)

    async with session_scope() as session:
        learned = await repo.memory_search(session, "summaries")
    assert learned, "the agent should have learned from the task on its own"


# --------------------------------------------------------------------------- #
# Failure-pattern learning (no LLM)
# --------------------------------------------------------------------------- #
async def test_repeated_failures_become_a_rule(environment):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="flaky work")
        for step in range(4):
            await repo.record_tool_call(
                session, task_id=task.id, step=step, tool="file_download",
                args={"url": f"https://x.com/{step}"}, status=ToolCallStatus.ERROR,
                error="network error: name resolution failed for 'host'",
            )

    learned = await learn_from_failures()
    assert learned, "3+ identical failures should produce an avoid-rule"
    assert learned[0]["kind"] == "gotcha"
    assert "file_download" in learned[0]["value"]


async def test_single_failure_does_not_create_a_rule(environment):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="one-off")
        await repo.record_tool_call(
            session, task_id=task.id, step=1, tool="file_read",
            args={}, status=ToolCallStatus.ERROR, error="missing file",
        )
    assert await learn_from_failures() == []


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_tokeniser_drops_noise():
    tokens = _tokens("Please send me the WEEKLY report")
    assert "weekly" in tokens and "report" in tokens
    assert "the" not in tokens and "me" not in tokens


def test_scoring_prefers_matching_memories():
    request = _tokens("send the invoice to the accountant")
    match = score_memory(request, "invoice_format", "invoices go out as PDF", "fact")
    unrelated = score_memory(request, "server_ip", "the build server is 10.0.0.5", "fact")
    assert match > unrelated


def test_preferences_are_surfaced_even_on_weak_match():
    request = _tokens("do something completely different")
    preference = score_memory(request, "tone", "owner likes short replies", "preference")
    plain_fact = score_memory(request, "tone", "owner likes short replies", "fact")
    assert preference > plain_fact


async def test_relevant_memories_ranks_by_request(environment):
    async with session_scope() as session:
        await repo.memory_store(session, key="invoice_format",
                                value="invoices must be PDF", kind="fact")
        await repo.memory_store(session, key="server_ip",
                                value="build server is 10.0.0.5", kind="fact")
        await repo.memory_store(session, key="reply_style",
                                value="owner prefers short replies", kind="preference")

    picked = await relevant_memories("prepare the invoice for this month", limit=3)
    assert any("invoice" in item for item in picked)
    assert not any("build server" in item for item in picked)


async def test_relevant_memories_empty_when_nothing_stored(environment):
    assert await relevant_memories("anything at all") == []


async def test_engine_injects_learned_memory_into_prompt(environment, echo_llm):
    """A learned preference must actually reach the model's context."""
    from app.agent.engine import AgentEngine

    async with session_scope() as session:
        await repo.memory_store(session, key="invoice_format",
                                value="invoices must always be PDF", kind="preference")
        task = await repo.create_task(
            session, user_request="prepare the invoice", chat_id=42, max_steps=4
        )
        task_id = task.id

    echo_llm.script = [{"action": "final", "final_answer": "done"}]
    await AgentEngine(llm=echo_llm).run_task(task_id)

    prompt = "\n".join(m.content for call in echo_llm.calls for m in call)
    assert "invoices must always be PDF" in prompt
