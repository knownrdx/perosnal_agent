"""Planning and self-verification.

Two things separate an agent that looks capable from one that is:

    it thinks before it acts        -> make_plan()
    it checks its own claim         -> verify_completion()

Both are advisory. If the model is unavailable or answers badly, the task still
runs - a broken planner must never block real work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.llm import LLMClient, LLMError, Message
from app.llm.prompts import (
    PLANNER_PROMPT,
    VERIFY_PROMPT,
    render_plan_request,
    render_tools,
    render_verification,
)
from app.logging_conf import get_logger
from app.security import Permission
from app.tools import registry

log = get_logger(__name__)

# Requests below this length that contain no connective words are almost
# certainly one-shot; planning them wastes a model call and adds latency.
_TRIVIAL_LENGTH = 60
_MULTI_STEP_HINTS = (
    " and ", " then ", " after ", " finally ", " also ", ",", ";",
    "each", "every", "all ", "compare", "summarise", "summarize", "report",
)


@dataclass(slots=True)
class Plan:
    complexity: str = "simple"
    steps: list[str] = field(default_factory=list)
    success_criteria: str = ""
    risks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity,
            "steps": self.steps,
            "success_criteria": self.success_criteria,
            "risks": self.risks,
        }

    @property
    def is_empty(self) -> bool:
        return not self.steps


@dataclass(slots=True)
class Verification:
    verified: bool
    reason: str = ""
    missing: list[str] = field(default_factory=list)


def looks_multi_step(request: str) -> bool:
    """Cheap check: is this worth planning at all?"""
    text = (request or "").strip().lower()
    if len(text) > _TRIVIAL_LENGTH:
        return True
    return any(hint in text for hint in _MULTI_STEP_HINTS)


def _extract_steps(raw: dict[str, Any]) -> list[str]:
    """Pull the step list out of a model reply, tolerating its phrasing.

    Models reliably produce a plan but not reliably under the key we asked for:
    "plan", "actions" and "tasks" all appear in practice, and steps sometimes
    arrive as objects ({"step": "..."}) rather than plain strings. Rejecting
    those would silently discard a perfectly good plan, so accept the shapes
    that carry the same meaning.
    """
    candidates: list[Any] = []
    for key in ("steps", "plan", "actions", "tasks", "step_list"):
        value = raw.get(key)
        if isinstance(value, list) and value:
            candidates = value
            break

    steps: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            # {"step": "..."} / {"action": "..."} / {"description": "..."}
            text = str(
                item.get("step")
                or item.get("action")
                or item.get("description")
                or item.get("name")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            steps.append(text[:300])
    return steps[:6]


async def make_plan(
    user_request: str,
    *,
    permission: Permission,
    llm: LLMClient,
    task_id: str = "",
) -> Plan:
    """Produce a short plan for non-trivial work. Never raises."""
    if not looks_multi_step(user_request):
        return Plan(complexity="simple")

    specs = registry.specs(max_permission=permission)
    messages = [
        Message("system", PLANNER_PROMPT),
        Message("user", render_plan_request(user_request, render_tools(specs))),
    ]

    try:
        raw = await llm.chat_json(messages)
    except (LLMError, json.JSONDecodeError) as exc:
        log.warning("planning_failed", extra={"task_id": task_id, "error": str(exc)[:200]})
        return Plan(complexity="standard")

    steps = _extract_steps(raw)
    plan = Plan(
        complexity=str(raw.get("complexity") or "standard").lower().strip(),
        steps=steps,
        success_criteria=str(raw.get("success_criteria") or "").strip()[:400],
        risks=[str(r).strip() for r in (raw.get("risks") or []) if str(r).strip()][:3],
    )

    if task_id and not plan.is_empty:
        async with session_scope() as session:
            await repo.update_task_context(session, task_id, {"plan": plan.as_dict()})
            await repo.log_event(
                session, "plan_created", task_id=task_id,
                data={"steps": len(plan.steps), "complexity": plan.complexity},
            )
    log.info(
        "plan_created",
        extra={"task_id": task_id, "steps": len(plan.steps), "complexity": plan.complexity},
    )
    return plan


async def verify_completion(
    *,
    task_id: str,
    user_request: str,
    claim: str,
    trace: list[str],
    output_files: list[str],
    llm: LLMClient,
) -> Verification:
    """Check the agent's success claim against what actually happened.

    A failed check does not fail the task: it hands the agent one more chance to
    finish the job properly, which is what a careful person would do.
    """
    if not trace:
        # Nothing was executed. Answering from knowledge alone is legitimate
        # (a question), so accept it rather than inventing a problem.
        return Verification(True, "no tool calls were needed")

    messages = [
        Message("system", VERIFY_PROMPT),
        Message(
            "user",
            render_verification(
                user_request=user_request,
                claim=claim,
                trace=trace,
                output_files=output_files,
            ),
        ),
    ]

    try:
        raw = await llm.chat_json(messages)
    except (LLMError, json.JSONDecodeError) as exc:
        # If the checker is broken, trust the agent. Refusing to finish because
        # a secondary model is down would strand real, completed work.
        log.warning("verification_failed", extra={"task_id": task_id, "error": str(exc)[:200]})
        return Verification(True, "verifier unavailable")

    verified = bool(raw.get("verified", True))
    result = Verification(
        verified=verified,
        reason=str(raw.get("reason") or "")[:300],
        missing=[str(m).strip() for m in (raw.get("missing") or []) if str(m).strip()][:5],
    )

    async with session_scope() as session:
        await repo.log_event(
            session,
            "completion_verified" if verified else "completion_rejected",
            task_id=task_id,
            level="INFO" if verified else "WARNING",
            data={"reason": result.reason, "missing": result.missing},
        )
    log.info(
        "verification",
        extra={"task_id": task_id, "verified": verified, "reason": result.reason[:120]},
    )
    return result
