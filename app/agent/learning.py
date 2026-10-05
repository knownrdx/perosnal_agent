"""Self-learning.

After every task the agent reflects on what happened and writes durable
lessons to memory, so it does not need to be told the same thing twice.

Three sources of learning:

1. Reflection - the LLM reads the finished task's tool trace and extracts
   owner preferences, reusable procedures and gotchas.
2. Failure patterns - deterministic, no LLM: when the same tool fails the same
   way repeatedly, an avoid-rule is written automatically. It lapses after
   FAILURE_RULE_TTL_DAYS without a recurrence.
3. Retrieval - before each step, the most relevant memories are scored and
   injected into the prompt, so what was learned actually gets used. Bengali
   script is transliterated first, so it matches Banglish and English.

Everything is filtered through the same secret guard as the memory tool: no
credentials are ever stored.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from app.agent import language
from app.db import repo
from app.db.base import session_scope
from app.db.models import MemoryEntry, Task, utcnow
from app.llm import LLMError, Message, get_llm
from app.logging_conf import get_logger
from app.tools.memory_tools import looks_like_secret

log = get_logger(__name__)

MAX_LESSONS_PER_TASK = 3
MIN_STEPS_TO_REFLECT = 2
FAILURE_THRESHOLD = 3

# A failure rule with no fresh occurrence for this long stops being injected.
# 14 days because the failures behind these rules are mostly circumstantial -
# a site was down, a login expired, a network blip - and once that is fixed a
# rule that keeps saying "this tool fails" steers the model away from a tool
# that now works. Two weeks still covers a weekly job hitting the same error
# twice in a row, so a genuinely recurring problem is never forgotten between
# runs; and any fresh occurrence revives the rule immediately.
FAILURE_RULE_TTL_DAYS = 14

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
    """Deterministic learning: turn repeated identical failures into rules.

    A rule is (re)written only when there is NEW EVIDENCE for it: it does not
    exist yet, or the same failure happened again after the rule was last
    written. The rule's timestamp therefore means "last seen failing", which
    is what lets retrieval drop rules nobody has hit for
    FAILURE_RULE_TTL_DAYS. Re-saving every rule after every task, as this
    used to, kept every rule forever fresh - and parked them all at the top
    of the newest-200 window that retrieval scans, crowding out real lessons.

    A changed count with no fresh occurrence (old failures sliding out of the
    200-row scan) is not news and does not refresh the rule. Failures that are
    already older than the TTL never create a rule in the first place.
    """
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
        last_seen: dict[tuple[str, str], datetime] = {}
        for call in rows:
            if not call.error:
                continue
            # Normalise the error to its shape, not its specific values.
            shape = re.sub(r"['\"][^'\"]{1,120}['\"]", "'X'", call.error[:160])
            shape = re.sub(r"\d+", "N", shape)
            pattern = (call.tool, shape.strip())
            patterns[pattern] += 1
            # Rows arrive newest first, so the first one seen is the latest.
            if call.created_at is not None:
                last_seen.setdefault(pattern, call.created_at)

        fresh_since = utcnow() - timedelta(days=FAILURE_RULE_TTL_DAYS)
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

            latest = last_seen.get((tool_name, shape))
            if latest is None or latest < fresh_since:
                continue  # stale evidence: not worth a rule any more
            existing = await session.scalar(select(MemoryEntry).where(MemoryEntry.key == key))
            written_at = existing.updated_at if existing is not None else None
            if written_at is not None and latest <= written_at:
                continue  # nothing happened since the rule was written

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
    # Banglish filler - pronouns, particles and the do/give/send/take verbs
    # that end almost every request ("report ta pathao", "eta koro"). Without
    # these, two unrelated Banglish sentences "match" on grammar alone.
    "ami", "amar", "amake", "tumi", "tomar", "tumar", "apni", "apnar",
    "eta", "ota", "oita", "ekta", "sob", "shob", "ache", "nai", "hobe", "holo",
    "koro", "kore", "kori", "korbo", "korbe", "dao", "diye", "deo", "nao",
    "pathao", "pathiye", "theke", "abar", "ekhon", "ekhoni", "aro", "keno",
    "kemon", "kivabe", "kokhon", "kothay", "kono", "accha", "haa", "valo",
}

# Retrieval budget. A plain lesson is one sentence, so 300 characters is
# plenty; a skill is the distilled summary of several lessons and is stored
# at up to 2000, so it gets more room - cutting it to a one-liner threw away
# exactly the part synthesis was for. The total cap keeps the injected block
# bounded however many long skills match (6 x 300 was the old ceiling).
ENTRY_CHARS = 300
SKILL_CHARS = 900
MEMORY_CONTEXT_CHARS = 2400
SKILL_BOOST = 0.45


def _tokens(text: str) -> set[str]:
    """Content words, after folding Bengali script into Banglish.

    The owner writes English, Banglish and Bengali script, and memories end
    up in all three. A Latin-only tokeniser saw a Bengali-script request as
    zero words and matched nothing. ``language.normalise`` is the same
    transliteration the deterministic triggers use, so "হোয়াটসঅ্যাপ স্টক"
    and "whatsapp stock" become the same tokens.
    """
    words = re.findall(r"[a-z0-9]+", language.normalise(str(text)).lower())
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
    # A skill summarises several lessons on one topic, so on that topic it
    # should outrank any single one of them (hence more than the gotcha
    # boost). Only on an actual word match, though: unlike a preference, a
    # skill is ABOUT something, and an unrelated one is noise in the prompt.
    if kind == "skill" and overlap:
        score += SKILL_BOOST
    return score


def _is_stale_failure_rule(entry: MemoryEntry, stale_before: datetime) -> bool:
    """An auto-written avoid-rule that has not fired for FAILURE_RULE_TTL_DAYS.

    learn_from_failures only touches a rule when the failure recurs, so its
    updated_at is "last seen failing".
    """
    if "failure" not in (entry.tags or []) or entry.updated_at is None:
        return False
    return entry.updated_at < stale_before


async def relevant_memories(user_request: str, limit: int = 6) -> list[str]:
    """Top-scoring memories for a request, formatted for the prompt.

    At most ``limit`` entries and MEMORY_CONTEXT_CHARS characters in total.
    An entry that would overflow the budget is skipped rather than ending
    the scan, so a shorter, lower-ranked lesson can still use the space left.
    """
    async with session_scope() as session:
        entries = await repo.memory_recent(session, limit=200)

    stale_before = utcnow() - timedelta(days=FAILURE_RULE_TTL_DAYS)
    request_tokens = _tokens(user_request)
    scored = [
        (score_memory(request_tokens, entry.key, entry.value, entry.kind), entry)
        for entry in entries
        if not _is_stale_failure_rule(entry, stale_before)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    out: list[str] = []
    used = 0
    for score, entry in scored:
        if len(out) >= limit or score <= 0.05:
            break  # sorted, so nothing after this scores higher
        cap = SKILL_CHARS if entry.kind == "skill" else ENTRY_CHARS
        line = f"[{entry.kind}] {entry.key}: {entry.value}"[:cap]
        if used + len(line) > MEMORY_CONTEXT_CHARS:
            continue
        out.append(line)
        used += len(line)
    return out
