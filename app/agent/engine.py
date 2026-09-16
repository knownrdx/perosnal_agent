"""Agent engine.

One task = one deterministic loop:

    load state -> build context -> ask LLM for ONE decision -> execute ->
    persist observation -> repeat until final / ask_user / limits reached

State lives in PostgreSQL, never in memory, so a restart resumes correctly.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import FailureKind, Task, TaskStatus
from app.llm import LLMClient, LLMError, Message, get_llm
from app.llm.prompts import SYSTEM_PROMPT, render_context, render_tools
from app.logging_conf import get_logger
from app.security import ApprovalRequired, Permission
from app.tools import registry
from app.tools.base import ToolContext

log = get_logger(__name__)


@dataclass(slots=True)
class StepOutcome:
    kind: str  # continue | final | ask_user | error
    message: str = ""
    tool: str = ""
    data: dict[str, Any] | None = None


class AgentEngine:
    def __init__(self, llm: LLMClient | None = None, executor: Any | None = None, notifier: Any | None = None) -> None:
        from app.agent.executor import Executor  # local import avoids a cycle

        self.llm = llm or get_llm()
        self.notifier = notifier
        self.executor = executor or Executor(notifier=notifier)

    # ------------------------------------------------------------------ #
    async def run_task(self, task_id: str) -> str:
        """Drive a task to a terminal (or waiting) state. Returns final status."""
        settings = get_settings()

        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                raise ValueError(f"task not found: {task_id}")
            await repo.log_event(session, "task_started", task_id=task_id,
                                 data={"request": task.user_request[:300]})

        history: list[str] = await self._load_history(task_id)

        # Think before acting. Resumed tasks keep the plan they already have.
        plan = snapshot_plan = None
        if settings.planning_enabled:
            async with session_scope() as session:
                existing = await repo.get_task(session, task_id)
                stored = (existing.context or {}).get("plan") if existing else None
            if stored:
                plan = stored
            elif not history:
                from app.agent.planner import make_plan

                built = await make_plan(
                    task.user_request,
                    permission=Permission(task.permission),
                    llm=self.llm,
                    task_id=task_id,
                )
                plan = built.as_dict() if not built.is_empty else None
        snapshot_plan = plan
        verify_attempts = 0

        while True:
            async with session_scope() as session:
                task = await repo.get_task(session, task_id)
                if task is None:
                    return TaskStatus.FAILED.value
                status = task.status
                step = task.current_step
                snapshot = _snapshot(task)

            if status in {TaskStatus.CANCELLED.value, TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}:
                return status

            if step >= snapshot["max_steps"]:
                await self._fail(
                    task_id,
                    f"step limit reached ({snapshot['max_steps']} steps) without completing the task",
                    FailureKind.PERMANENT,
                )
                return TaskStatus.FAILED.value

            step += 1
            async with session_scope() as session:
                await repo.update_task(session, task_id, current_step=step,
                                       status=TaskStatus.RUNNING.value)

            try:
                decision = await self._decide(snapshot, step, history, snapshot_plan)
            except LLMError as exc:
                await self._fail(task_id, f"LLM failure: {exc}", FailureKind.TEMPORARY)
                return TaskStatus.FAILED.value

            outcome = await self._apply(snapshot, step, decision, history)

            if outcome.kind == "final":
                # Check the claim against what actually happened, once.
                verified = await self._verify(snapshot, outcome, history, verify_attempts)
                if verified is not None:
                    verify_attempts += 1
                    history.append(verified)
                    await self._append_history(task_id, verified)
                    continue
                await self._complete(task_id, outcome.message, outcome.data or {})
                return TaskStatus.COMPLETED.value
            if outcome.kind == "ask_user":
                await self._wait_for_user(task_id, outcome.message)
                return TaskStatus.WAITING_FOR_USER.value
            if outcome.kind == "approval":
                await self._wait_for_approval(task_id, outcome.tool)
                return TaskStatus.WAITING_FOR_USER.value
            if outcome.kind == "error":
                await self._fail(task_id, outcome.message, FailureKind.PERMANENT)
                return TaskStatus.FAILED.value
            # else: continue looping

    # ------------------------------------------------------------------ #
    async def _decide(
        self,
        snapshot: dict[str, Any],
        step: int,
        history: list[str],
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        permission = Permission(snapshot["permission"])
        specs = registry.specs(max_permission=permission)

        # Relevance-scored recall: what the agent learned that matters HERE.
        from app.agent.context import attention_notes, compact
        from app.agent.learning import relevant_memories

        memories = await relevant_memories(snapshot["user_request"], limit=6)
        conversation: list[str] = []
        async with session_scope() as session:
            if snapshot["chat_id"]:
                for msg in await repo.recent_messages(session, snapshot["chat_id"], limit=6):
                    conversation.append(f"{msg.role}: {msg.content[:300]}")

        # Long runs keep their early context as a summary instead of losing it.
        packed = await compact(history, llm=self.llm)

        messages = [
            Message("system", SYSTEM_PROMPT + "\n\n" + render_tools(specs)),
            Message(
                "user",
                render_context(
                    task_id=snapshot["id"],
                    user_request=snapshot["user_request"],
                    step=step,
                    max_steps=snapshot["max_steps"],
                    history=packed.recent,
                    memories=memories,
                    conversation=conversation,
                    plan=plan,
                    summary=packed.summary,
                    attention=attention_notes(history, snapshot["user_request"]),
                ),
            ),
        ]
        decision = await self.llm.chat_json(messages)
        log.info(
            "agent_decision",
            extra={
                "task_id": snapshot["id"],
                "step": step,
                "action": decision.get("action"),
                "tool": decision.get("tool"),
            },
        )
        return decision

    async def _apply(
        self, snapshot: dict[str, Any], step: int, decision: dict[str, Any], history: list[str]
    ) -> StepOutcome:
        action = str(decision.get("action", "")).lower().strip()
        thought = str(decision.get("thought", ""))[:300]
        task_id = snapshot["id"]

        if action == "final":
            answer = str(decision.get("final_answer") or thought or "Task completed.")
            files = decision.get("output_files") or []
            return StepOutcome("final", answer, data={"output_files": files})

        if action == "ask_user":
            question = str(decision.get("question") or decision.get("final_answer") or thought)
            return StepOutcome("ask_user", question or "The agent needs more information.")

        if action != "tool":
            history.append(
                f"[step {step}] invalid action {action!r}: you must use tool | final | ask_user"
            )
            await self._append_history(task_id, history[-1])
            return StepOutcome("continue")

        tool_name = str(decision.get("tool") or "").strip()
        args = decision.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        ctx = ToolContext(
            task_id=task_id,
            chat_id=snapshot["chat_id"],
            user_id=snapshot["user_id"],
            step=step,
            permission=Permission(snapshot["permission"]),
        )

        try:
            result = await self.executor.execute(tool_name, args, ctx)
        except ApprovalRequired as exc:
            return StepOutcome("approval", exc.reason, tool=exc.tool)

        observation = json.dumps(
            {"tool": tool_name, "args": args, "tool_result": result.observation()},
            default=str,
        )[:4000]
        entry = f"[step {step}] {thought}\n{observation}"
        history.append(entry)
        await self._append_history(task_id, entry)

        # Track produced artefacts automatically.
        if result.ok:
            produced = result.data.get("path")
            if isinstance(produced, str) and produced:
                async with session_scope() as session:
                    await repo.add_output_file(session, task_id, produced)

        return StepOutcome("continue")

    async def _verify(
        self,
        snapshot: dict[str, Any],
        outcome: StepOutcome,
        history: list[str],
        attempts: int,
    ) -> str | None:
        """Return a correction to feed back, or None to accept the completion.

        Only ever challenges once. A second opinion is useful; an argument
        between two models is not, and would burn the owner's step budget.
        """
        if attempts >= 1 or not get_settings().verify_completion:
            return None

        from app.agent.planner import verify_completion

        result = await verify_completion(
            task_id=snapshot["id"],
            user_request=snapshot["user_request"],
            claim=outcome.message,
            trace=history,
            output_files=list((outcome.data or {}).get("output_files") or []),
            llm=self.llm,
        )
        if result.verified:
            return None

        missing = "; ".join(result.missing) if result.missing else result.reason
        log.info(
            "completion_rejected",
            extra={"task_id": snapshot["id"], "reason": result.reason[:200]},
        )
        return (
            "[self-check] Not finished yet: "
            f"{result.reason}. Still missing: {missing}. "
            "Continue working, or use ask_user if you genuinely cannot proceed."
        )

    # ------------------------------------------------------------------ #
    async def _load_history(self, task_id: str) -> list[str]:
        """Rebuild the reasoning trace from persisted tool calls (restart safe)."""
        async with session_scope() as session:
            calls = await repo.list_tool_calls(session, task_id, limit=30)
        history: list[str] = []
        for call in calls:
            payload = {
                "tool": call.tool,
                "args": call.args,
                "tool_result": (
                    {"ok": True, "result": call.result}
                    if call.status == "OK"
                    else {"ok": False, "error": call.error, "failure_kind": call.failure_kind}
                ),
            }
            history.append(f"[step {call.step}] {json.dumps(payload, default=str)[:3000]}")
        return history

    async def _append_history(self, task_id: str, entry: str) -> None:
        async with session_scope() as session:
            await repo.log_event(session, "agent_step", task_id=task_id, data={"entry": entry[:2000]})

    async def _complete(self, task_id: str, answer: str, data: dict[str, Any]) -> None:
        async with session_scope() as session:
            for path in data.get("output_files") or []:
                if isinstance(path, str) and path:
                    await repo.add_output_file(session, task_id, path)
            await repo.set_task_status(session, task_id, TaskStatus.COMPLETED, result=answer[:8000])
            await repo.log_event(session, "task_completed", task_id=task_id,
                                 data={"answer": answer[:500]})
            await self._release_session(session, task_id)
        log.info("task_completed", extra={"task_id": task_id})
        await self._notify(task_id, "task_completed")
        await self._learn(task_id)

    async def _fail(self, task_id: str, error: str, kind: FailureKind) -> None:
        settings = get_settings()
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            never_retry = {
                FailureKind.PERMANENT,
                FailureKind.USER_ACTION_REQUIRED,
                FailureKind.AUTH,
                FailureKind.INVALID_INPUT,
            }
            retriable = (
                kind is FailureKind.TEMPORARY
                and kind not in never_retry
                and task is not None
                and (
                    settings.task_persist_forever
                    or task.retry_count < task.max_retries
                )
            )
            if retriable:
                # Exponential backoff, capped so a long-stuck task still
                # gets retried roughly every 30 minutes instead of storming
                # or being abandoned; retry_count itself is not a stop
                # condition when task_persist_forever is on.
                delay_s = min(30 * (2 ** task.retry_count), 1800)
                await repo.requeue_task(session, task_id, delay_s=delay_s)
                await repo.log_event(session, "task_retry", task_id=task_id,
                                     level="WARNING", data={"error": error[:500]})
                log.warning("task_retry", extra={"task_id": task_id, "error": error[:300]})
                return
            await repo.set_task_status(
                session, task_id, TaskStatus.FAILED, error=error[:4000], failure_kind=kind.value
            )
            await repo.log_event(session, "task_failed", task_id=task_id,
                                 level="ERROR", data={"error": error[:500], "kind": kind.value})
            await self._release_session(session, task_id)
        log.error("task_failed", extra={"task_id": task_id, "error": error[:300]})
        await self._notify(task_id, "task_failed")
        await self._learn(task_id)

    async def _notify(self, task_id: str, event: str) -> None:
        """Tell the owner a task reached a terminal state.

        Shielded from cancellation: the status is already committed, so if the
        process is stopping we must still finish delivering the message.
        Otherwise a task completes in the database and the owner is never told -
        exactly the "did it actually finish?" ambiguity the agent exists to
        remove. Any leftovers are swept up by ``notify_unnotified`` at startup.
        """
        if self.notifier is None:
            return
        handler = getattr(self.notifier, event, None)
        if handler is None:
            return
        try:
            await asyncio.shield(handler(task_id))
        except asyncio.CancelledError:
            log.warning("notify_interrupted", extra={"task_id": task_id, "event": event})
            raise
        except Exception as exc:  # noqa: BLE001 - never let a message break a task
            log.error(
                "notify_error",
                extra={"task_id": task_id, "event": event, "error": str(exc)[:200]},
            )

    @staticmethod
    async def _release_session(session, task_id: str) -> None:
        """A finished task stops being the chat's 'active' work.

        It stays as last_task_id so a later follow-up can still continue it.
        """
        task = await repo.get_task(session, task_id)
        if task is None or task.chat_id is None:
            return
        row = await repo.get_session(session, task.chat_id)
        if row is not None and row.active_task_id == task_id:
            await repo.update_session(
                session, task.chat_id, active_task_id=None, last_task_id=task_id
            )

    async def _learn(self, task_id: str) -> None:
        """Reflect on a finished task. Never allowed to break the task itself."""
        if not get_settings().learning_enabled:
            return
        from app.agent.learning import learn_from_failures, reflect_on_task
        from app.agent.skills import synthesize_skills

        try:
            await reflect_on_task(task_id, llm=self.llm)
            await learn_from_failures()
            await synthesize_skills(llm=self.llm)
        except Exception as exc:  # noqa: BLE001
            log.warning("learning_failed", extra={"task_id": task_id, "error": str(exc)[:200]})

    async def _wait_for_user(self, task_id: str, question: str) -> None:
        async with session_scope() as session:
            await repo.set_task_status(
                session, task_id, TaskStatus.WAITING_FOR_USER, result=question[:4000]
            )
            await repo.log_event(session, "task_waiting_for_user", task_id=task_id,
                                 data={"question": question[:500]})
        if self.notifier is not None:
            await self.notifier.task_question(task_id, question)

    async def _wait_for_approval(self, task_id: str, tool: str) -> None:
        async with session_scope() as session:
            await repo.set_task_status(
                session,
                task_id,
                TaskStatus.WAITING_FOR_USER,
                result=f"Waiting for approval of {tool}",
            )
        log.info("task_waiting_approval", extra={"task_id": task_id, "tool": tool})


def _snapshot(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "user_request": task.user_request,
        "permission": task.permission,
        "chat_id": task.chat_id,
        "user_id": task.user_id,
        "max_steps": task.max_steps,
        "status": task.status,
    }
