"""Chat sessions: chat vs task routing, follow-ups, controls."""

from __future__ import annotations

import pytest

from app.agent.conversation import handle_message
from app.agent.router import Intent, classify
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, TaskStatus

CHAT_ID = 42
USER_ID = 42


async def _task_count() -> int:
    async with session_scope() as session:
        return len(await repo.list_tasks(session, limit=100))


# --------------------------------------------------------------------------- #
# OTP-bot deterministic triggers hooked into handle_message - must intercept
# BEFORE the LLM router so they work even when every LLM provider is down.
# --------------------------------------------------------------------------- #
async def test_otp_start_trigger_asks_for_tag_without_creating_a_task(environment):
    from app.automation import otp_bot

    rel = "uploads/plain_numbers.txt"
    from app.config import get_settings

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "plain_numbers.txt").write_text("+880***1111\n", encoding="utf-8")

    await otp_bot.ensure_thread(CHAT_ID)
    await otp_bot.enqueue_file(rel, "plain_numbers.txt")
    before = await _task_count()

    reply = await handle_message(CHAT_ID, USER_ID, "start")

    assert "tag" in reply.text.lower()
    assert reply.intent is Intent.CONTROL
    assert await _task_count() == before  # no task created - handled deterministically
    assert await otp_bot.get_awaiting_tag_entry() is not None


async def test_otp_triggers_ignored_outside_the_dedicated_thread(environment):
    """A plain "start" in a normal conversation must stay a normal message."""
    from app.automation import otp_bot
    from app.config import get_settings

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "plain_numbers.txt").write_text("+880***1111\n", encoding="utf-8")

    await otp_bot.ensure_thread(CHAT_ID)
    await otp_bot.enqueue_file("uploads/plain_numbers.txt", "plain_numbers.txt")

    # Owner switches to some other conversation.
    async with session_scope() as session:
        await repo.reset_session(session, CHAT_ID)

    reply = await handle_message(CHAT_ID, USER_ID, "start")
    # Not intercepted by the automation - no tag prompt was armed.
    assert await otp_bot.get_awaiting_tag_entry() is None
    assert "ki tag dibo" not in reply.text


async def test_otp_tag_answer_intercepted_before_router(environment, monkeypatch):
    from app.automation import otp_bot
    from app.config import get_settings
    from app.integrations.telegram_user import set_userbot

    uploads = get_settings().workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "plain_numbers.txt").write_text("+880***1111\n", encoding="utf-8")
    await otp_bot.ensure_thread(CHAT_ID)
    await otp_bot.enqueue_file("uploads/plain_numbers.txt", "plain_numbers.txt")
    await handle_message(CHAT_ID, USER_ID, "start")  # arms the tag prompt

    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(otp_bot, "_sleep", _no_sleep)

    class FakeUserbot:
        def __init__(self):
            self._reply_id = 0

        async def send_message(self, target, text, reply_to=None):
            return {"sent": True, "message_id": 1}

        async def send_file(self, target, path, caption="", reply_to=None):
            return {"sent": True, "message_id": 2}

        async def read_messages(self, target, limit=20):
            # Fresh id each read - the automation waits for a NEWER message.
            self._reply_id += 1
            return [{"text": "Added.", "id": self._reply_id, "out": False}]

    set_userbot(FakeUserbot())
    try:
        reply = await handle_message(CHAT_ID, USER_ID, "BD")
        assert "Shuru hoye geche" in reply.text
        assert (await otp_bot.get_config())["enabled"] is True
    finally:
        set_userbot(None)


async def test_otp_stop_trigger_intercepted_before_router(environment):
    from app.automation import otp_bot

    await otp_bot.ensure_thread(CHAT_ID)
    await otp_bot.save_config({"enabled": True})
    reply = await handle_message(CHAT_ID, USER_ID, "stop")
    assert (await otp_bot.get_config())["enabled"] is False
    assert reply.intent is Intent.CONTROL


# --------------------------------------------------------------------------- #
# Router: deterministic rules (no LLM needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "hi", "hello", "thanks", "ok", "who are you", "what can you do",
        "how do you work", "kemon acho",
    ],
)
async def test_small_talk_is_chat(environment, text):
    decision = await classify(text)
    assert decision.intent is Intent.CHAT, text


@pytest.mark.parametrize(
    "text",
    [
        "download https://example.com/report.pdf and send it to me",
        "create a summary of the logs",
        "schedule a check every morning",
        "send me the invoice file",
        "convert that video to mp4",
    ],
)
async def test_real_work_is_a_task(environment, text):
    decision = await classify(text)
    assert decision.intent is Intent.TASK, text


@pytest.mark.parametrize(
    "text",
    ["status", "how is it going", "are you done", "any update", "cancel it"],
)
async def test_status_questions_are_control(environment, text):
    decision = await classify(text, active_task_id="abc123")
    assert decision.intent is Intent.CONTROL, text


@pytest.mark.parametrize(
    "text",
    ["and also send it as PDF", "actually make it smaller", "yes", "do it", "continue"],
)
async def test_continuations_are_follow_ups(environment, text):
    decision = await classify(text, active_task_id="abc123")
    assert decision.intent is Intent.FOLLOW_UP, text
    assert decision.target_task_id == "abc123"


async def test_follow_up_without_active_task_becomes_task(environment):
    decision = await classify("and also send it as PDF", active_task_id=None)
    assert decision.intent is not Intent.FOLLOW_UP


async def test_forced_modes_win(environment):
    assert (await classify("download the file", mode="chat")).intent is Intent.CHAT
    assert (await classify("hello", mode="task")).intent is Intent.TASK


async def test_router_falls_back_to_task_when_llm_dies(environment):
    """A broken model must never silently swallow real work."""
    from app.llm import LLMError
    from app.llm.base import LLMClient

    class Dead(LLMClient):
        name = "dead"

        async def chat(self, messages, *, temperature=None):
            raise LLMError("down")

        async def health(self):
            return {"ok": False}

    # Deliberately ambiguous phrasing so the rules do not settle it.
    decision = await classify(
        "the quarterly numbers for the northern region please", llm=Dead()
    )
    assert decision.intent is Intent.TASK


# --------------------------------------------------------------------------- #
# Conversation: chatting creates no tasks
# --------------------------------------------------------------------------- #
async def test_greeting_does_not_create_a_task(environment, echo_llm):
    echo_llm.script = [{"reply": "hello"}]
    reply = await handle_message(CHAT_ID, USER_ID, "hi", llm=echo_llm)

    assert reply.intent is Intent.CHAT
    assert reply.created_task is False
    assert await _task_count() == 0


async def test_task_message_creates_exactly_one_task(environment, echo_llm):
    reply = await handle_message(
        CHAT_ID, USER_ID, "download https://x.com/a.pdf and send it to me", llm=echo_llm
    )
    assert reply.intent is Intent.TASK
    assert reply.created_task is True
    assert await _task_count() == 1

    async with session_scope() as session:
        row = await repo.get_session(session, CHAT_ID)
    assert row.active_task_id == reply.task_id


async def test_three_greetings_still_create_no_tasks(environment, echo_llm):
    for text in ["hi", "thanks", "ok"]:
        await handle_message(CHAT_ID, USER_ID, text, llm=echo_llm)
    assert await _task_count() == 0


# --------------------------------------------------------------------------- #
# A file the owner uploaded before instructing gets attached automatically
# --------------------------------------------------------------------------- #
async def test_pending_upload_is_attached_to_the_next_task(environment, echo_llm):
    """Regression: /tglogin bug's sibling - uploading a file then instructing
    in a separate message left the agent with no reference to the file at
    all, so it asked the owner "which file?" even though they had just sent
    one. The Telegram document handler stores {"pending_upload": {...}} on
    the chat session's context; handle_message must fold that into the new
    task's request and then clear it so a later unrelated task does not
    re-attach a stale file.
    """
    async with session_scope() as session:
        row = await repo.ensure_session(session, CHAT_ID)
        await repo.update_session(
            session, CHAT_ID,
            context={"pending_upload": {"path": "uploads/numbers.txt", "name": "numbers.txt"}},
        )

    reply = await handle_message(
        CHAT_ID, USER_ID, "add these numbers to @PBDxbot", llm=echo_llm
    )
    assert reply.created_task is True

    async with session_scope() as session:
        task = await repo.get_task(session, reply.task_id)
        chat_row = await repo.get_session(session, CHAT_ID)

    assert "uploads/numbers.txt" in task.user_request
    # Consumed, not left dangling for the next unrelated message to inherit.
    assert "pending_upload" not in (chat_row.context or {})


async def test_no_pending_upload_leaves_request_untouched(environment, echo_llm):
    reply = await handle_message(
        CHAT_ID, USER_ID, "search the web for mikrotik hotspot setup", llm=echo_llm
    )
    async with session_scope() as session:
        task = await repo.get_task(session, reply.task_id)
    assert "Attached file" not in task.user_request


# --------------------------------------------------------------------------- #
# Follow-ups attach instead of duplicating
# --------------------------------------------------------------------------- #
async def test_follow_up_attaches_to_running_task(environment, echo_llm):
    first = await handle_message(
        CHAT_ID, USER_ID, "create a report of last week", llm=echo_llm
    )
    assert first.created_task

    async with session_scope() as session:
        await repo.update_task(session, first.task_id, status=TaskStatus.RUNNING.value)

    second = await handle_message(
        CHAT_ID, USER_ID, "and also send it as PDF", llm=echo_llm
    )

    assert second.intent is Intent.FOLLOW_UP
    assert second.task_id == first.task_id
    assert await _task_count() == 1, "a follow-up must not spawn a second task"

    async with session_scope() as session:
        task = await repo.get_task(session, first.task_id)
    assert "send it as PDF" in task.user_request
    assert task.context.get("follow_ups") == ["and also send it as PDF"]


async def test_follow_up_after_completion_continues_as_child_task(environment, echo_llm):
    first = await handle_message(CHAT_ID, USER_ID, "create the report", llm=echo_llm)

    async with session_scope() as session:
        await repo.set_task_status(
            session, first.task_id, TaskStatus.COMPLETED, result="report done"
        )
        await repo.update_session(session, CHAT_ID, active_task_id=first.task_id)

    second = await handle_message(CHAT_ID, USER_ID, "also send it as PDF", llm=echo_llm)

    assert second.intent is Intent.FOLLOW_UP
    assert second.task_id != first.task_id

    async with session_scope() as session:
        child = await repo.get_task(session, second.task_id)
    assert child.parent_task_id == first.task_id
    assert "PREVIOUS RESULT" in child.user_request
    assert "report done" in child.user_request


# --------------------------------------------------------------------------- #
# Control questions answer from state, without an LLM or a task
# --------------------------------------------------------------------------- #
async def test_status_question_reports_running_work(environment, echo_llm):
    created = await handle_message(CHAT_ID, USER_ID, "download the big file", llm=echo_llm)
    async with session_scope() as session:
        await repo.update_task(session, created.task_id, status=TaskStatus.RUNNING.value)

    reply = await handle_message(CHAT_ID, USER_ID, "how is it going", llm=echo_llm)

    assert reply.intent is Intent.CONTROL
    assert reply.created_task is False
    assert "Working on" in reply.text
    assert await _task_count() == 1


async def test_status_with_nothing_running(environment, echo_llm):
    reply = await handle_message(CHAT_ID, USER_ID, "any update", llm=echo_llm)
    assert reply.intent is Intent.CONTROL
    assert "Nothing running" in reply.text


async def test_status_reports_last_failure(environment, echo_llm):
    created = await handle_message(CHAT_ID, USER_ID, "download the file", llm=echo_llm)
    async with session_scope() as session:
        await repo.set_task_status(
            session, created.task_id, TaskStatus.FAILED, error="host unreachable"
        )
        await repo.update_session(session, CHAT_ID, active_task_id=None)

    reply = await handle_message(CHAT_ID, USER_ID, "are you done", llm=echo_llm)
    assert "failed" in reply.text.lower()
    assert "host unreachable" in reply.text


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
async def test_session_mode_pins_behaviour(environment, echo_llm):
    async with session_scope() as session:
        await repo.ensure_session(session, CHAT_ID)
        await repo.update_session(session, CHAT_ID, mode="chat")

    reply = await handle_message(
        CHAT_ID, USER_ID, "download https://x.com/a.pdf", llm=echo_llm
    )
    assert reply.intent is Intent.CHAT
    assert await _task_count() == 0, "chat mode must never start jobs"


async def test_reset_clears_thread_but_keeps_tasks(environment, echo_llm):
    created = await handle_message(CHAT_ID, USER_ID, "build the report", llm=echo_llm)

    async with session_scope() as session:
        old_thread_id = (await repo.ensure_session(session, CHAT_ID)).current_thread_id
        new_thread_id = await repo.reset_session(session, CHAT_ID)
        row = await repo.get_session(session, CHAT_ID)
        # The NEW thread has no messages yet - old_thread_id's history is untouched.
        turns = await repo.recent_messages(session, CHAT_ID, limit=10, thread_id=new_thread_id)
        old_turns = await repo.recent_messages(session, CHAT_ID, limit=10, thread_id=old_thread_id)
        task = await repo.get_task(session, created.task_id)

    assert row.active_task_id is None
    assert new_thread_id != old_thread_id
    assert turns == []
    assert len(old_turns) == 2, "the owner asked to keep full session history, not delete it"
    assert task is not None, "/new must not delete work"


async def test_turns_are_recorded_for_context(environment, echo_llm):
    await handle_message(CHAT_ID, USER_ID, "hi", llm=echo_llm)
    async with session_scope() as session:
        turns = await repo.recent_messages(session, CHAT_ID, limit=10)
    roles = [t.role for t in turns]
    assert roles == ["user", "assistant"]


async def test_engine_releases_the_session_on_completion(environment, echo_llm):
    """After a job finishes it must stop capturing follow-ups as 'active'."""
    from app.agent.engine import AgentEngine

    created = await handle_message(CHAT_ID, USER_ID, "write the summary file", llm=echo_llm)

    echo_llm.script = [
        {"action": "tool", "tool": "file_write",
         "args": {"path": "output/s.txt", "content": "x"}},
        {"action": "final", "final_answer": "done"},
        {"lessons": []},
    ]
    await AgentEngine(llm=echo_llm).run_task(created.task_id)

    async with session_scope() as session:
        row = await repo.get_session(session, CHAT_ID)
    assert row.active_task_id is None
    assert row.last_task_id == created.task_id
