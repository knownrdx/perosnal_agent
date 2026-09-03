"""PRIMARY V1 SUCCESS TEST (master prompt section 30).

    Owner -> Telegram: "start this task; when it finishes, get the resulting
    file and send it to my Telegram."

    -> task created -> worker runs it -> file produced -> file validated ->
       file sent to Telegram -> delivery verified -> owner notified

Also covers the offline/restart requirement: the whole flow must survive the
process dying mid-way, and must not double-send after recovery.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.engine import AgentEngine
from app.db import repo
from app.db.base import session_scope
from app.db.models import TaskStatus
from app.telegram.notifier import Notifier
from app.workers.task_worker import TaskWorker

REQUEST = (
    "Download the daily report, verify it, then send the resulting file to my Telegram "
    "and tell me when it's done."
)

# The "remote job" the agent monitors: a file that appears a moment later.
PRODUCE_SCRIPT = """
import pathlib, time
target = pathlib.Path('output/daily_report.txt')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text('REPORT DATA 2026\\nrows=128\\nstatus=ok\\n', encoding='utf-8')
print('produced', target)
"""


def _full_workflow_script() -> list[dict]:
    return [
        # 1. run the long job that produces the file
        {"action": "tool", "tool": "python_execute", "args": {"code": PRODUCE_SCRIPT},
         "thought": "running the job that produces the report"},
        # 2. detect completion / validate the artefact exists and is non-empty
        {"action": "tool", "tool": "file_exists", "args": {"path": "output/daily_report.txt"},
         "thought": "verifying the file exists before sending"},
        # 3. read it to confirm content
        {"action": "tool", "tool": "file_read", "args": {"path": "output/daily_report.txt"},
         "thought": "checking the contents"},
        # 4. deliver to Telegram
        {"action": "tool", "tool": "telegram_send_file",
         "args": {"path": "output/daily_report.txt", "caption": "Daily report"},
         "thought": "sending the verified file"},
        # 5. report
        {"action": "final", "final_answer": "Report generated, verified and sent to Telegram.",
         "output_files": ["output/daily_report.txt"]},
    ]


async def test_primary_success_test_end_to_end(environment, echo_llm, fake_telegram):
    async with session_scope() as session:
        task = await repo.create_task(
            session, user_request=REQUEST, title=REQUEST[:80], chat_id=42, user_id=42,
            permission="WRITE", max_steps=environment.max_task_steps,
        )
        task_id = task.id

    echo_llm.script = _full_workflow_script()

    notifier = Notifier()
    engine = AgentEngine(llm=echo_llm, notifier=notifier)
    worker = TaskWorker(engine=engine, notifier=notifier)
    worker.poll_interval = 0.05
    await worker.start()
    try:
        for _ in range(150):
            async with session_scope() as session:
                current = await repo.get_task(session, task_id)
            if current.status in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}:
                break
            await asyncio.sleep(0.1)
    finally:
        await worker.stop()

    # --- task state ----------------------------------------------------- #
    assert current.status == TaskStatus.COMPLETED.value, current.error
    assert "output/daily_report.txt" in current.output_files

    # --- the file really exists on disk with real content ---------------- #
    produced = environment.workspace / "output" / "daily_report.txt"
    assert produced.exists() and produced.stat().st_size > 0
    assert "REPORT DATA" in produced.read_text(encoding="utf-8")

    # --- the file was actually delivered and delivery was verified ------- #
    documents = fake_telegram.sent_documents()
    assert len(documents) == 1
    assert documents[0]["chat_id"] == 42

    async with session_scope() as session:
        calls = await repo.list_tool_calls(session, task_id)
    send_call = next(c for c in calls if c.tool == "telegram_send_file")
    assert send_call.status == "OK"
    assert send_call.result["message_id"], "delivery must be confirmed by a message_id"
    assert send_call.result["file_id"]

    # --- the owner was notified of completion ---------------------------- #
    texts = "\n".join(fake_telegram.sent_messages())
    assert "Task completed" in texts
    assert "daily_report.txt" in texts

    # --- full audit trail persisted -------------------------------------- #
    assert [c.tool for c in calls] == [
        "python_execute", "file_exists", "file_read", "telegram_send_file",
    ]


async def test_crash_midway_then_restart_completes_without_double_send(
    environment, echo_llm, fake_telegram
):
    """Simulates the VPS dying after the file was sent but before completion."""
    async with session_scope() as session:
        task = await repo.create_task(
            session, user_request=REQUEST, chat_id=42, user_id=42, max_steps=10
        )
        task_id = task.id

    # --- process 1: delivers the file, then the process dies ------------- #
    engine = AgentEngine(llm=echo_llm, notifier=Notifier())

    # Crash state a dead worker leaves behind: task stuck in RUNNING.
    async with session_scope() as session:
        await repo.update_task(
            session, task_id, status=TaskStatus.RUNNING.value, worker_id="dead-worker"
        )

    # Deliver the file exactly as process 1 would have, then "die".
    from app.agent.executor import Executor
    from app.security import Permission, safe_path
    from app.tools.base import ToolContext

    report = safe_path("output/daily_report.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("REPORT DATA 2026\n", encoding="utf-8")

    executor = Executor()
    ctx = ToolContext(task_id=task_id, chat_id=42, permission=Permission.WRITE, step=4)
    first = await executor.execute(
        "telegram_send_file", {"path": "output/daily_report.txt", "caption": "Daily report"}, ctx
    )
    assert first.ok
    assert len(fake_telegram.sent_documents()) == 1

    # --- process 2: restart, recover, retry the same send ---------------- #
    recovered = await TaskWorker(engine=engine).recover()
    assert recovered == 1

    replay = await executor.execute(
        "telegram_send_file", {"path": "output/daily_report.txt", "caption": "Daily report"}, ctx
    )
    assert replay.ok and replay.data.get("idempotent_replay") is True
    assert len(fake_telegram.sent_documents()) == 1, "restart must not resend the file"

    # --- finish the task after recovery ---------------------------------- #
    echo_llm.script = [
        {"action": "final", "final_answer": "Report already delivered; task closed.",
         "output_files": ["output/daily_report.txt"]}
    ]
    async with session_scope() as session:
        await repo.update_task(session, task_id, status=TaskStatus.PENDING.value)
    status = await AgentEngine(llm=echo_llm, notifier=Notifier()).run_task(task_id)
    assert status == TaskStatus.COMPLETED.value


async def test_completion_notification_is_not_duplicated(environment, echo_llm, fake_telegram):
    async with session_scope() as session:
        task = await repo.create_task(session, user_request="tiny job", chat_id=42, user_id=42)
        task_id = task.id
    echo_llm.script = [{"action": "final", "final_answer": "Nothing to do."}]

    notifier = Notifier()
    await AgentEngine(llm=echo_llm, notifier=notifier).run_task(task_id)
    await notifier.task_completed(task_id)
    await notifier.task_completed(task_id)

    completed = [t for t in fake_telegram.sent_messages() if "Task completed" in t]
    assert len(completed) == 1
