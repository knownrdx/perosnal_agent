"""Self-learning.

After every task the agent reflects on what happened and writes durable
lessons to memory, so it does not need to be told the same thing twice.

Three sources of learning:

1. Reflection - the LLM reads the finished task's tool trace and extracts
   owner preferences, reusable procedures and gotchas.
2. Failure patterns - deterministic, no LLM: when the same tool fails the same
   way repeatedly, an avoid-rule is written automatically.
3. Retrieval - before each step, the most relevant memories are scored and
   injected into the prompt, so what was learned actually gets used.

Everything is filtered through the same secret guard as the memory tool: no
credentials are ever stored.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.db.models import Task
from app.llm import LLMError, Message, get_llm
from app.logging_conf import get_logger
from app.tools.memory_tools import looks_like_secret

log = get_logger(__name__)

MAX_LESSONS_PER_TASK = 3
MIN_STEPS_TO_REFLECT = 2
FAILURE_THRESHOLD = 3

REFLECTION_PROMPT = """You are the memory of a personal AI agent.

Below is a task the agent just finished. Extract ONLY durable lessons that will
make future tasks go better. Return ONE JSON object:

{
  "lessons": [
    {"key": "short_snake_case_key",
     "value": "one sentence, specific and actionable",
     "kind": "preference" | "fact" | "workflow" | "gotcha"}
  ]
}

Rules:
- 0 to 3 lessons. Fewer is better. Return {"lessons": []} if nothing is durable.
- A lesson must still be true next week. No task ids, no timestamps, no counts.
- Write facts, not commands: "owner prefers PDF reports" not "always send PDF".
- NEVER include passwords, tokens, API keys, phone numbers or file contents.
- Skip anything already obvious from the tool list.
"""


def _clean_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", str(raw).lower().strip())
    return re.sub(r"_+", "_", key).strip("_")[:80]


def _summarise_task(task: Task, calls: list[Any]) -> str:
    lines = [
        f"REQUEST: {task.user_request[:600]}",
        f"OUTCOME: {task.status}",
    ]
    if task.result:
        lines.append(f"RESULT: {task.result[:400]}")
    if task.error:
        lines.append(f"ERROR: {task.error[:400]}")
    if calls:
        lines.append("STEPS:")
        for call in calls[:12]:
            detail = "ok" if call.status == "OK" else f"{call.status}: {call.error[:120]}"
            lines.append(f"  {call.step}. {call.tool} -> {detail}")
    return "\n".join(lines)


async def reflect_on_task(task_id: str, llm: Any | None = None) -> list[dict[str, str]]:
    """Extract and store durable lessons from a finished task."""
    async with session_scope() as session:
        task = await repo.get_task(session, task_id)
        if task is None:
            return []
        calls = await repo.list_tool_calls(session, task_id, limit=20)
        snapshot = _summarise_task(task, calls)
        steps = task.current_step
        status = task.status

    # Trivial tasks teach nothing.
    if steps < MIN_STEPS_TO_REFLECT and status == "COMPLETED":
        return []

    client = llm or get_llm()
    try:
        data = await client.chat_json(
            [Message("system", REFLECTION_PROMPT), Message("user", snapshot)]
        )
    except LLMError as exc:
        log.warning("reflection_failed", extra={"task_id": task_id, "error": str(exc)[:200]})
        return []
    except Exception as exc:  # noqa: BLE001 - learning must never break a task
        log.warning("reflection_error", extra={"task_id": task_id, "error": str(exc)[:200]})
        return []

    lessons = data.get("lessons")
    if not isinstance(lessons, list):
        return []

    stored: list[dict[str, str]] = []
    async with session_scope() as session:
        for raw in lessons[:MAX_LESSONS_PER_TASK]:
            if not isinstance(raw, dict):
                continue
            key = _clean_key(raw.get("key", ""))
            value = str(raw.get("value", "")).strip()
            kind = str(raw.get("kind", "fact")).strip().lower()
            if not key or not value or len(value) > 500:
                continue
            if looks_like_secret(f"{key} {value}"):
                log.warning("reflection_secret_blocked", extra={"task_id": task_id})
                continue
            if kind not in {"preference", "fact", "workflow", "gotcha"}:
                kind = "fact"

            await repo.memory_store(
                session, key=key, value=value, kind=kind,
                tags=["learned"], source_task_id=task_id,
            )
            stored.append({"key": key, "value": value, "kind": kind})

    if stored:
        log.info("agent_learned", extra={"task_id": task_id, "count": len(stored),
                                         "keys": [s["key"] for s in stored]})
    return stored


async def learn_from_failures() -> list[dict[str, str]]:
    """Deterministic learning: turn repeated identical failures into rules."""
    from sqlalchemy import select

    from app.db.models import ToolCall

    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(ToolCall)
                .where(ToolCall.status != "OK")
                .order_by(ToolCall.created_at.desc())
                .limit(200)
            )
        ).all()

        patterns: Counter[tuple[str, str]] = Counter()
        for call in rows:
            if not call.error:
                continue
            # Normalise the error to its shape, not its specific values.
            shape = re.sub(r"['\"][^'\"]{1,120}['\"]", "'X'", call.error[:160])
            shape = re.sub(r"\d+", "N", shape)
            patterns[(call.tool, shape.strip())] += 1

        learned: list[dict[str, str]] = []
        for (tool_name, shape), count in patterns.most_common(5):
            if count < FAILURE_THRESHOLD:
                continue
            key = _clean_key(f"avoid_{tool_name}_{shape[:40]}")
            value = (
                f"{tool_name} keeps failing with: {shape[:200]} "
                f"(seen {count}x) - check the arguments before calling it again."
            )
            if looks_like_secret(value):
                continue
            await repo.memory_store(
                session, key=key, value=value, kind="gotcha", tags=["learned", "failure"]
            )
            learned.append({"key": key, "value": value, "kind": "gotcha"})

    if learned:
        log.info("agent_learned_failures", extra={"count": len(learned)})
    return learned


# --------------------------------------------------------------------------- #
# Retrieval: pick the memories that actually matter for this task
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "be", "it", "this", "that", "my", "me", "i", "you",
    "please", "can", "could", "would", "then", "when", "send", "get", "do",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def score_memory(request_tokens: set[str], key: str, value: str, kind: str) -> float:
    """Relevance score for one memory against the current request."""
    memory_tokens = _tokens(key) | _tokens(value)
    if not memory_tokens:
        return 0.0
    overlap = len(request_tokens & memory_tokens)
    score = overlap / (len(request_tokens) ** 0.5 + 1)
    # Preferences and gotchas are worth surfacing even on a weak match.
    if kind in {"preference", "gotcha"}:
        score += 0.35
    if kind == "workflow":
        score += 0.15
    return score


async def relevant_memories(user_request: str, limit: int = 6) -> list[str]:
    """Top-scoring memories for a request, formatted for the prompt."""
    async with session_scope() as session:
        entries = await repo.memory_recent(session, limit=200)

    request_tokens = _tokens(user_request)
    scored = [
        (score_memory(request_tokens, entry.key, entry.value, entry.kind), entry)
        for entry in entries
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    out: list[str] = []
    for score, entry in scored[:limit]:
        if score <= 0.05:
            continue
        out.append(f"[{entry.kind}] {entry.key}: {entry.value}"[:300])
    return out
