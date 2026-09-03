#!/usr/bin/env bash
# Verify chat-vs-task routing against the LIVE agent, using the real model.
# Runs inside the agent container so it uses the deployed code and DB.
set -uo pipefail
cd /opt/ai-agent

docker compose exec -T agent python - <<'PY'
import asyncio

from app.agent.conversation import handle_message
from app.db import repo
from app.db.base import create_all, init_engine, session_scope
from app.db.models import TaskStatus

CHAT = 999000001   # scratch chat id, not the owner's
USER = 999000001


async def task_count() -> int:
    async with session_scope() as session:
        rows = await repo.list_tasks(session, chat_id=CHAT, limit=100)
        return len(rows)


async def main() -> int:
    init_engine()
    await create_all()

    async with session_scope() as session:
        await repo.ensure_session(session, CHAT)
        await repo.reset_session(session, CHAT)

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    # 1. chatting must not create work
    reply = await handle_message(CHAT, USER, "hi, what can you do?")
    check("greeting -> CHAT, no task", reply.intent.value == "CHAT" and not reply.created_task,
          f"intent={reply.intent.value}")
    print(f"        reply: {reply.text[:110]!r}")

    # 2. a real instruction creates exactly one task
    reply = await handle_message(
        CHAT, USER, "write a file at output/session_probe.txt containing OK")
    first_id = reply.task_id
    check("instruction -> TASK", reply.intent.value == "TASK" and reply.created_task,
          f"id={first_id}")
    check("exactly one task so far", await task_count() == 1)

    # 3. follow-up attaches instead of duplicating
    async with session_scope() as session:
        await repo.update_task(session, first_id, status=TaskStatus.RUNNING.value)
    reply = await handle_message(CHAT, USER, "and also make it uppercase")
    check("follow-up -> same task", reply.task_id == first_id,
          f"intent={reply.intent.value}")
    check("still one task", await task_count() == 1)

    async with session_scope() as session:
        task = await repo.get_task(session, first_id)
    check("follow-up text merged in", "uppercase" in task.user_request)

    # 4. status question answers from state, no new task
    reply = await handle_message(CHAT, USER, "how is it going?")
    check("status -> CONTROL, no task", reply.intent.value == "CONTROL" and not reply.created_task)
    print(f"        reply: {reply.text[:110]!r}")
    check("still one task after status", await task_count() == 1)

    # cleanup
    async with session_scope() as session:
        await repo.set_task_status(session, first_id, TaskStatus.CANCELLED,
                                   error="routing probe")
        await repo.reset_session(session, CHAT)

    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("LIVE ROUTING OK")
    return 0


raise SystemExit(asyncio.run(main()))
PY
