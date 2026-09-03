"""Agent engine, permissions, approvals and idempotency."""

from __future__ import annotations

import pytest

from app.agent.engine import AgentEngine
from app.agent.executor import Executor
from app.db import repo
from app.db.base import session_scope
from app.db.models import ApprovalStatus, TaskStatus
from app.security import ApprovalRequired, Permission
from app.tools.base import ToolContext


async def _create_task(request: str, permission: str = "WRITE", max_steps: int = 8) -> str:
    async with session_scope() as session:
        task = await repo.create_task(
            session,
            user_request=request,
            chat_id=42,
            user_id=42,
            permission=permission,
            max_steps=max_steps,
        )
        return task.id


async def test_engine_runs_tool_then_finishes(environment, echo_llm):
    task_id = await _create_task("write a file")
    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/note.txt", "content": "hello"}, "thought": "writing"},
        {"action": "final", "final_answer": "File written.", "output_files": ["output/note.txt"]},
    ]

    status = await AgentEngine(llm=echo_llm).run_task(task_id)
    assert status == TaskStatus.COMPLETED.value

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
        calls = await repo.list_tool_calls(session, task_id)

    assert task.result == "File written."
    assert "output/note.txt" in task.output_files
    assert [c.tool for c in calls] == ["file_write"]
    assert (environment.workspace / "output" / "note.txt").read_text() == "hello"


async def test_engine_recovers_from_bad_tool_name(environment, echo_llm):
    task_id = await _create_task("do something")
    echo_llm.script = [
        {"action": "tool", "tool": "does_not_exist", "args": {}},
        {"action": "final", "final_answer": "Recovered."},
    ]
    status = await AgentEngine(llm=echo_llm).run_task(task_id)
    assert status == TaskStatus.COMPLETED.value

    async with session_scope() as session:
        calls = await repo.list_tool_calls(session, task_id)
    assert calls[0].status == "DENIED" and "unknown tool" in calls[0].error


async def test_step_limit_fails_task(environment, echo_llm):
    task_id = await _create_task("loop forever", max_steps=3)
    echo_llm.handler = lambda messages: {
        "action": "tool", "tool": "file_read",
        "args": {"path": "output/missing.txt"}, "thought": "retrying",
    }
    status = await AgentEngine(llm=echo_llm).run_task(task_id)
    assert status == TaskStatus.FAILED.value

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
    assert "step limit" in task.error


async def test_ask_user_pauses_task(environment, echo_llm):
    task_id = await _create_task("ambiguous request")
    echo_llm.script = [{"action": "ask_user", "question": "Which file exactly?"}]
    status = await AgentEngine(llm=echo_llm).run_task(task_id)
    assert status == TaskStatus.WAITING_FOR_USER.value

    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
    assert "Which file" in task.result


async def test_read_only_task_cannot_write(environment, echo_llm):
    task_id = await _create_task("read only job", permission="READ")
    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/x.txt", "content": "nope"}},
        {"action": "final", "final_answer": "Could not write."},
    ]
    await AgentEngine(llm=echo_llm).run_task(task_id)

    async with session_scope() as session:
        calls = await repo.list_tool_calls(session, task_id)
    assert calls[0].status == "DENIED" and "WRITE" in calls[0].error
    assert not (environment.workspace / "output" / "x.txt").exists()


async def test_high_risk_requires_approval_then_proceeds(environment, echo_llm):
    from app.security import safe_path

    victim = safe_path("output/delete_me.txt")
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("bye", encoding="utf-8")

    task_id = await _create_task("delete the file", permission="HIGH_RISK")
    echo_llm.script = [
        {"action": "tool", "tool": "file_delete", "args": {"path": "output/delete_me.txt"}},
    ]

    engine = AgentEngine(llm=echo_llm)
    status = await engine.run_task(task_id)
    assert status == TaskStatus.WAITING_FOR_USER.value
    assert victim.exists(), "file must NOT be deleted before approval"

    async with session_scope() as session:
        approval = await repo.pending_approval_for_task(session, task_id)
        assert approval is not None and approval.tool == "file_delete"
        decided = await repo.decide_approval(
            session, approval.id, approved=True, user_id=42
        )
        assert decided.status == ApprovalStatus.APPROVED.value
        await repo.update_task(session, task_id, status=TaskStatus.PENDING.value)

    echo_llm.script = [
        {"action": "tool", "tool": "file_delete", "args": {"path": "output/delete_me.txt"}},
        {"action": "final", "final_answer": "Deleted."},
    ]
    status = await engine.run_task(task_id)
    assert status == TaskStatus.COMPLETED.value
    assert not victim.exists()


async def test_rejected_approval_blocks_action(environment, echo_llm):
    from app.security import safe_path

    victim = safe_path("output/keep_me.txt")
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("stay", encoding="utf-8")

    task_id = await _create_task("delete it", permission="HIGH_RISK")
    echo_llm.script = [{"action": "tool", "tool": "file_delete", "args": {"path": "output/keep_me.txt"}}]
    engine = AgentEngine(llm=echo_llm)
    await engine.run_task(task_id)

    async with session_scope() as session:
        approval = await repo.pending_approval_for_task(session, task_id)
        await repo.decide_approval(session, approval.id, approved=False, user_id=42)
        await repo.update_task(session, task_id, status=TaskStatus.PENDING.value)

    echo_llm.script = [
        {"action": "tool", "tool": "file_delete", "args": {"path": "output/keep_me.txt"}},
        {"action": "final", "final_answer": "Owner said no."},
    ]
    await engine.run_task(task_id)
    assert victim.exists(), "rejected action must never execute"


async def test_idempotent_side_effect_not_repeated(environment, fake_telegram):
    task_id = await _create_task("send twice")
    executor = Executor()
    ctx = ToolContext(task_id=task_id, chat_id=42, permission=Permission.WRITE, step=1)

    first = await executor.execute("telegram_send_message", {"text": "same text"}, ctx)
    second = await executor.execute("telegram_send_message", {"text": "same text"}, ctx)

    assert first.ok and second.ok
    assert second.data.get("idempotent_replay") is True
    assert len(fake_telegram.sent_messages()) == 1, "identical send must not repeat"


async def test_different_args_are_not_deduplicated(environment, fake_telegram):
    task_id = await _create_task("send two different")
    executor = Executor()
    ctx = ToolContext(task_id=task_id, chat_id=42, permission=Permission.WRITE, step=1)
    await executor.execute("telegram_send_message", {"text": "one"}, ctx)
    await executor.execute("telegram_send_message", {"text": "two"}, ctx)
    assert fake_telegram.sent_messages() == ["one", "two"]


async def test_history_is_rebuilt_after_restart(environment, echo_llm):
    """A restarted engine must see previous tool calls, not start blind."""
    task_id = await _create_task("multi step")
    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/step1.txt", "content": "one"}},
    ]
    engine = AgentEngine(llm=echo_llm)
    echo_llm.script.append({"action": "ask_user", "question": "pause here"})
    await engine.run_task(task_id)

    history = await AgentEngine(llm=echo_llm)._load_history(task_id)
    assert history and "file_write" in history[0]
