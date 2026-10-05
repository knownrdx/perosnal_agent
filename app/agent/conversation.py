"""Conversation handling.

Turns an incoming Telegram message into the right kind of work:

    CHAT       -> answer inline in one LLM call (no task, no worker)
    TASK       -> create a background Task
    FOLLOW_UP  -> feed into the task that is already running / just finished
    CONTROL    -> answer from persisted state (status, cancel)

The session (``chat_sessions``) remembers which task the chat is currently
about, so "and also send it as PDF" attaches to that work instead of starting
something new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.router import Decision, Intent, classify
from app.config import get_settings
from app.agent import language
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, TERMINAL_STATUSES, TaskStatus
from app.llm import LLMError, Message, get_llm
from app.llm.prompts import REPLY_STYLE
from app.logging_conf import get_logger

log = get_logger(__name__)

MAX_TURNS = 10

CHAT_PROMPT = """You are the owner's private AI agent, talking to them on Telegram.

You are mid-conversation. Answer directly and briefly - this is chat, not a
report. Understand whatever language the owner writes in (English, Banglish
or Bengali script), and always answer in English.

You have tools and can run background jobs, but this particular message did not
require any. If the owner is actually asking you to DO something, say what you
will do and that they should confirm.

Never invent facts about files, tasks or messages you have not seen. You can
see only what is written in this conversation: you do NOT know what is
running, what finished, or where their files are unless it is stated above.
If asked about something you cannot see, say so and point at /status - never
guess a number, a country or a state. Admitting you do not know is always
better than a confident wrong answer.

Keep it under 6 sentences unless the owner asked for detail.

""" + REPLY_STYLE

# The one reply guaranteed to arrive when every model is down. English like
# every other reply, and it says what to do next rather than just "error".
MODEL_DOWN_REPLY = (
    "\u26a0\ufe0f Sorry, I couldn't reach the model just now.\n\n"
    "Please try again in a moment, or check /status."
)


@dataclass(slots=True)
class Reply:
    """What the Telegram layer should send back."""

    text: str
    intent: Intent
    task_id: str | None = None
    created_task: bool = False
    # Telegram-HTML rendering of ``text``, when there is one (the OTP status
    # table). ``text`` stays plain for the web chat and the stored history.
    html: str | None = None


async def _session_snapshot(chat_id: int) -> dict[str, Any]:
    """Current session state plus the task it is about, if any."""
    async with session_scope() as session:
        row = await repo.ensure_session(session, chat_id)
        mode, active_id, last_id = row.mode, row.active_task_id, row.last_task_id
        thread_id = row.current_thread_id
        pending_upload = dict(row.context or {}).get("pending_upload")

        active_request, active_status = "", ""
        if active_id:
            task = await repo.get_task(session, active_id)
            if task is None:
                active_id = None
            else:
                active_request, active_status = task.user_request, task.status
                # Finished long ago? stop treating it as "the" active task.
                if task.status in {s.value for s in TERMINAL_STATUSES}:
                    active_id = task.id  # still followable, but not running

        turns = [
            f"{m.role}: {m.content[:200]}"
            for m in await repo.recent_messages(session, chat_id, limit=6, thread_id=thread_id)
        ]

    return {
        "mode": mode,
        "thread_id": thread_id,
        "active_task_id": active_id,
        "last_task_id": last_id,
        "active_request": active_request,
        "active_status": active_status,
        "turns": turns,
        "pending_upload": pending_upload,
    }


async def chat_reply(
    chat_id: int, text: str, llm: Any | None = None, *, thread_id: str | None = None,
) -> str:
    """One-shot conversational answer, with memory and recent turns."""
    from app.agent.learning import relevant_memories

    client = llm or get_llm()
    memories = await relevant_memories(text, limit=4)

    async with session_scope() as session:
        if thread_id is None:
            thread_id = (await repo.ensure_session(session, chat_id)).current_thread_id
        turns = await repo.recent_messages(session, chat_id, limit=MAX_TURNS, thread_id=thread_id)

    messages = [Message("system", CHAT_PROMPT)]
    # Always answer in English, whatever the owner wrote in. CHAT_PROMPT says
    # so once; for a Bengali or Banglish message this repeats it next to the
    # message, because models mirror the language in front of them and a rule
    # stated only at the top loses to a Bengali question at the bottom.
    directive = language.directive(text)
    if directive:
        messages.append(Message("system", directive))
    if memories:
        messages.append(
            Message("system", "What you know about the owner:\n" + "\n".join(memories))
        )
    for turn in turns:
        role = "assistant" if turn.role == "assistant" else "user"
        messages.append(Message(role, turn.content[:1500]))
    messages.append(Message("user", text))

    try:
        response = await client.chat(messages)
    except LLMError as exc:
        log.warning("chat_reply_failed", extra={"error": str(exc)[:200]})
        return MODEL_DOWN_REPLY

    answer = (response.content or "").strip()
    # The task engine speaks JSON; if a model slips into that here, unwrap it.
    if answer.startswith("{"):
        from app.llm import extract_json

        data = extract_json(answer) or {}
        answer = str(
            data.get("final_answer") or data.get("question") or data.get("thought") or answer
        )
    return answer[:3500] or "..."


async def _control_reply(snapshot: dict[str, Any], chat_id: int) -> str:
    """Answer status/cancel questions from persisted state - no LLM needed."""
    async with session_scope() as session:
        active = await repo.list_tasks(
            session, statuses=[s.value for s in ACTIVE_STATUSES], chat_id=chat_id, limit=5
        )
        latest = await repo.latest_task_for_chat(session, chat_id)
        rows = [
            {"id": t.id, "title": t.title, "status": t.status, "step": t.current_step,
             "max_steps": t.max_steps}
            for t in active
        ]
        last = None
        if latest is not None:
            last = {"id": latest.id, "title": latest.title, "status": latest.status,
                    "result": latest.result, "error": latest.error}

    if rows:
        lines = ["\u23F3 Working on:"]
        for row in rows:
            lines.append(
                f"  {row['title'][:60]} - {row['status'].lower()} "
                f"(step {row['step']}/{row['max_steps']})"
            )
        lines.append("")
        lines.append("I will message you the moment it is done.")
        return "\n".join(lines)

    if last is None:
        return "Nothing running right now. Send me something to do."

    if last["status"] == TaskStatus.COMPLETED.value:
        return f"\u2705 Last job finished: {last['title'][:60]}\n\n{last['result'][:600]}"
    if last["status"] == TaskStatus.FAILED.value:
        return f"\u274C Last job failed: {last['title'][:60]}\n\n{last['error'][:400]}"
    return f"Nothing active. Last job ({last['title'][:60]}) is {last['status'].lower()}."


async def handle_message(
    chat_id: int,
    user_id: int,
    text: str,
    *,
    llm: Any | None = None,
) -> Reply:
    """Route one message and produce the reply to send."""
    settings = get_settings()
    snapshot = await _session_snapshot(chat_id)
    thread_id = snapshot["thread_id"]

    # ------------------------------------------------------------------ #
    # Deterministic OTP-bot number-file automation triggers. These are
    # checked BEFORE the LLM classifier on purpose: the whole point of this
    # workflow is that it keeps working even when every LLM provider is
    # down/rate-limited, and a plain "start"/"off"/tag-answer message must
    # never get reinterpreted as small talk by a router that's just guessing.
    #
    # Scoped to the dedicated OTP thread so these very ordinary words stay
    # ordinary everywhere else - "start" in a general conversation should
    # not silently fire an automation.
    from app.automation import otp_bot

    async with session_scope() as session:
        await repo.add_message(session, chat_id=chat_id, role="user", content=text, thread_id=thread_id)

    if await otp_bot.is_otp_thread(chat_id):
        # A custom value the owner was asked for (interval / restock level)
        # is answered by the next message, so it is consumed before anything
        # else can reinterpret it.
        pending = await otp_bot.handle_pending_input(text)
        if pending is not None:
            await _record_reply(chat_id, pending, thread_id=thread_id)
            return Reply(pending, Intent.CONTROL)

        # The command words below are checked BEFORE the open service / run
        # length questions. Those questions take free text, and used to take
        # these words too: "stop" while the service question was open was
        # stored as the service name - and, being the last answer needed,
        # STARTED the run under a tag called "stop". No service or run
        # length is spelled "stop", "status" or "sob bad dao"; "bad"/"skip"
        # stay answers (they mean "not this one") and still reach them.
        if otp_bot.is_clear_trigger(text):
            answer = await otp_bot.handle_clear_queue()
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        removal_target = otp_bot.country_to_remove(text)
        if removal_target:
            answer = await otp_bot.handle_remove_country(removal_target)
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if otp_bot.is_stop_trigger(text):
            answer = await otp_bot.handle_stop_trigger()
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if otp_bot.is_status_trigger(text):
            from app.telegram import otp_panel

            # The status is a Telegram-HTML table; the web chat and the
            # stored history show text as-is, so they get it without tags.
            rich = await otp_bot.handle_status_trigger()
            answer = otp_panel.html_to_plain(rich)
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL, html=rich)

        if otp_bot.is_help_trigger(text):
            answer = otp_bot.help_text()
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if otp_bot.is_start_trigger(text):
            # With a question still open this re-asks it rather than
            # starting, which is what handle_start_trigger does anyway.
            answer = await otp_bot.handle_start_trigger()
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if await otp_bot.get_awaiting_tag_entry() is not None:
            answer = await otp_bot.handle_tag_answer(text)
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        # "How long should this run?" is answered by the next message too.
        # Checked after the tag question (that one is asked first), so a bare
        # "20h" is read as the answer it is rather than as small talk.
        if await otp_bot.get_awaiting_runtime():
            answer = await otp_bot.handle_runtime_answer(text)
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if otp_bot.is_resume_trigger(text):
            answer = await otp_bot.handle_resume_trigger()
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

        if otp_bot.is_skip_trigger(text) and (
            " ".join(language.normalise(text).lower().split()) not in {"no", "na", "eta na"}
        ):
            # "bad dao" with no question open has nothing to apply to. Passed
            # on, the router filed it as a background JOB - an agent told
            # just "remove", with no idea what - or, with the model down,
            # answered "could not reach the model". Say what does work.
            # (A plain "no"/"na" is left alone: it is as likely to be a reply
            # to the chat model's own question.)
            answer = (
                "\U0001F5D1 Nothing is waiting for an answer right now.\n"
                "\n"
                "\u2022 To drop one country: type '<country> bad dao' (e.g. 'Nigeria bad dao')\n"
                "\u2022 To empty the queue: type 'sob bad dao'\n"
                "\u2022 To stop everything: type 'stop'"
            )
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.CONTROL)

    decision: Decision = await classify(
        text,
        active_task_id=snapshot["active_task_id"],
        active_task_request=snapshot["active_request"],
        recent_turns=snapshot["turns"],
        mode=snapshot["mode"],
        llm=llm,
    )

    log.info(
        "message_routed",
        extra={"chat_id": chat_id, "intent": decision.intent.value,
               "reason": decision.reason, "task_id": decision.target_task_id},
    )

    # ------------------------------------------------------------------ #
    if decision.intent is Intent.CONTROL:
        answer = await _control_reply(snapshot, chat_id)
        await _record_reply(chat_id, answer, thread_id=thread_id)
        return Reply(answer, decision.intent, snapshot["active_task_id"])

    # ------------------------------------------------------------------ #
    if decision.intent is Intent.CHAT:
        answer = await chat_reply(chat_id, text, llm=llm, thread_id=thread_id)
        await _record_reply(chat_id, answer, thread_id=thread_id)
        return Reply(answer, decision.intent)

    # ------------------------------------------------------------------ #
    if decision.intent is Intent.FOLLOW_UP and decision.target_task_id:
        target_id = decision.target_task_id
        async with session_scope() as session:
            task = await repo.get_task(session, target_id)
            if task is None:
                decision = Decision(Intent.TASK, "target task vanished")
            else:
                still_running = task.status in {s.value for s in ACTIVE_STATUSES}
                context = dict(task.context or {})
                context.setdefault("follow_ups", []).append(text)

                if still_running:
                    # Feed it in and let the running task pick it up.
                    await repo.update_task(
                        session, target_id,
                        user_request=f"{task.user_request}\n\n[owner adds] {text}",
                        context=context,
                    )
                    await repo.log_event(session, "task_follow_up", task_id=target_id,
                                         data={"text": text[:300]})
                    answer = (
                        f"\U0001F504 Added to the running job:\n{text[:200]}"
                    )
                else:
                    # Finished: continue it as a linked new task with its history.
                    new_task = await repo.create_task(
                        session,
                        user_request=(
                            f"Continue this earlier work.\n\n"
                            f"PREVIOUS REQUEST: {task.user_request[:600]}\n"
                            f"PREVIOUS RESULT: {(task.result or task.error)[:600]}\n\n"
                            f"NOW THE OWNER SAYS: {text}"
                        ),
                        title=text[:80],
                        chat_id=chat_id,
                        user_id=user_id,
                        permission="WRITE",
                        max_steps=settings.max_task_steps,
                        max_retries=settings.max_task_retries,
                        parent_task_id=target_id,
                        context={"source": "follow_up", "parent": target_id},
                    )
                    target_id = new_task.id
                    await repo.update_session(
                        session, chat_id, active_task_id=target_id, last_task_id=target_id
                    )
                    answer = f"\U0001F504 Continuing from the last job.\nID: {target_id}"

        if decision.intent is Intent.FOLLOW_UP:
            await _record_reply(chat_id, answer, thread_id=thread_id)
            return Reply(answer, Intent.FOLLOW_UP, target_id, created_task=True)

    # ------------------------------------------------------------------ #
    # New task
    request_text = text
    pending_upload = snapshot.get("pending_upload")
    if pending_upload and pending_upload.get("path"):
        request_text = (
            f"{text}\n\n"
            f"[Attached file: {pending_upload['path']} "
            f"(uploaded as \"{pending_upload.get('name', '')}\")]"
        )

    async with session_scope() as session:
        task = await repo.create_task(
            session,
            user_request=request_text,
            title=text[:80],
            chat_id=chat_id,
            user_id=user_id,
            permission="WRITE",
            max_steps=settings.max_task_steps,
            max_retries=settings.max_task_retries,
            context={"source": "telegram"},
        )
        task_id = task.id
        row = await repo.ensure_session(session, chat_id)
        session_ctx = dict(row.context or {})
        if pending_upload:
            session_ctx.pop("pending_upload", None)
        await repo.update_session(
            session, chat_id,
            active_task_id=task_id,
            last_task_id=task_id,
            turn_count=row.turn_count + 1,
            title=row.title or text[:80],
            context=session_ctx,
        )

    answer = (
        f"\U0001F680 On it\n\n{text[:200]}\n\n"
        f"ID: {task_id}\nI will message you when it is done."
    )
    await _record_reply(chat_id, answer, thread_id=thread_id)
    log.info("task_created_from_chat", extra={"task_id": task_id, "chat_id": chat_id})
    return Reply(answer, Intent.TASK, task_id, created_task=True)


async def _record_reply(chat_id: int, text: str, *, thread_id: str | None = None) -> None:
    async with session_scope() as session:
        await repo.add_message(session, chat_id=chat_id, role="assistant", content=text, thread_id=thread_id)
        row = await repo.ensure_session(session, chat_id)
        await repo.update_session(session, chat_id, turn_count=row.turn_count + 1)
