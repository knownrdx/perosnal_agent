"""Self-updating 'skills' memory layer.

The per-task reflection in ``app.agent.learning`` writes fine-grained lessons
(one fact/preference/workflow/gotcha at a time). This module is the coarser
synthesis layer on top of it: periodically it looks at the recent lessons,
groups the ones that are about the same topic, and folds each sufficiently
large group into ONE durable 'skill' memory entry - a short document the
agent keeps refining rather than a scattered pile of one-liners.

Never allowed to break anything else: if the LLM synthesis fails for any
reason, this degrades to a deterministic bullet-list join of the source
entries instead of raising.

Runs after every task, so it must be cheap when nothing changed: each skill
row carries a fingerprint of the lessons it was made from, and a topic is
only sent to the model again when that fingerprint no longer matches.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from app.db import repo
from app.db.base import session_scope
from app.db.models import MemoryEntry
from app.llm import LLMError, Message, get_llm
from app.logging_conf import get_logger
from app.tools.memory_tools import looks_like_secret

log = get_logger(__name__)

SOURCE_KINDS = ("workflow", "gotcha", "preference")
MIN_CLUSTER_SIZE = 3
SCAN_LIMIT = 200
MAX_CLUSTERS_PER_RUN = 10
SYNTHESIS_INPUTS = 12          # lessons per topic actually shown to the model
SKILL_SCAN_LIMIT = 500         # existing skill rows read to compare fingerprints
_FINGERPRINT_TAG = "fp:"

SYNTHESIS_PROMPT = """You maintain a personal AI agent's durable 'skills' memory.

Below are several related lessons the agent learned individually. Merge them
into ONE short, coherent skill summary the agent can consult later.

Return ONE JSON object:

{"summary": "2-4 sentences, specific and actionable, no task ids or timestamps"}

Rules:
- Combine overlapping points, drop duplicates, keep it concrete.
- NEVER include passwords, tokens, API keys, phone numbers or file contents.
"""


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 2]


def _topic_for_key(key: str) -> str:
    """Coarse topic from a memory key: shared prefix / leading tokens.

    Keys are snake_case, e.g. 'report_format', 'report_source',
    'invoice_pdf_preferred' -> topic 'report' / 'invoice'.
    """
    tokens = _tokens(key.replace("_", " "))
    return tokens[0] if tokens else key


def cluster_entries(entries: list[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """Group entries by coarse topic using leading-token overlap on the key.

    Simple and deterministic: no embeddings, no external calls - just the
    first meaningful token of the memory key, which for this codebase's
    naming convention (subject_detail) is a decent proxy for "same topic".
    """
    clusters: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        topic = _topic_for_key(entry.key)
        clusters[topic].append(entry)
    return clusters


def _fallback_summary(topic: str, entries: list[MemoryEntry]) -> str:
    """Deterministic bullet-list join, used when the LLM is unavailable."""
    bullets = "; ".join(f"{e.key}: {e.value}"[:200] for e in entries[:8])
    return f"[{topic}] {bullets}"[:2000]


async def _synthesise_cluster(
    topic: str, entries: list[MemoryEntry], llm: Any
) -> tuple[str, bool]:
    """Return (summary, came_from_model). False means the fallback was used."""
    source_lines = "\n".join(
        f"- ({e.kind}) {e.key}: {e.value}" for e in entries[:SYNTHESIS_INPUTS]
    )
    try:
        data = await llm.chat_json(
            [Message("system", SYNTHESIS_PROMPT), Message("user", source_lines)]
        )
        summary = str(data.get("summary", "")).strip()
        if not summary or len(summary) > 2000 or looks_like_secret(summary):
            raise ValueError("unusable synthesis output")
        return summary, True
    except LLMError as exc:
        log.warning("skill_synthesis_llm_failed", extra={"topic": topic, "error": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001 - learning must never break other things
        log.warning("skill_synthesis_error", extra={"topic": topic, "error": str(exc)[:200]})
    return _fallback_summary(topic, entries), False


def _fingerprint(entries: list[MemoryEntry]) -> str:
    """Stable hash of the lessons a skill is built from.

    Keys, kinds and values only - not timestamps - so re-learning a lesson
    word for word (which bumps updated_at) does not count as a change.
    Sorted by key so the newest-first scan order does not matter either.
    """
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda e: e.key):
        digest.update(f"{entry.key}\x1f{entry.kind}\x1f{entry.value}\x1e".encode("utf-8"))
    return digest.hexdigest()[:16]


def _stored_fingerprint(entry: MemoryEntry) -> str:
    """The fingerprint a skill row was last synthesised from ('' if none).

    Kept in ``tags`` because the memory table has no metadata column and the
    existing upsert already replaces tags on every write.
    """
    for tag in entry.tags or []:
        if isinstance(tag, str) and tag.startswith(_FINGERPRINT_TAG):
            return tag[len(_FINGERPRINT_TAG):]
    return ""


def _clean_topic_key(topic: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", topic.lower().strip())
    return re.sub(r"_+", "_", key).strip("_")[:80] or "general"


async def synthesize_skills(llm: Any | None = None) -> list[dict]:
    """Consolidate recent fine-grained lessons into coarse 'skill' entries.

    Reads recent workflow/gotcha/preference memories, clusters them by topic,
    and for every cluster with >= MIN_CLUSTER_SIZE entries upserts ONE 'skill'
    memory entry keyed by the topic (so re-running updates rather than
    duplicates). Returns what was written/updated, for logging/inspection.

    Only topics whose lessons changed since their last synthesis are sent to
    the model. This runs after EVERY task, and without that check it spent
    up to MAX_CLUSTERS_PER_RUN model calls each time re-deriving summaries of
    lessons that had not moved - and, because the same first topics always
    filled the cap, topics past it were never synthesised at all. Skipping
    unchanged topics first means the cap now rotates through the backlog.

    A fallback (model unavailable) summary is stored WITHOUT a fingerprint,
    so the next run retries it with the model; it costs one call per such
    topic and stops as soon as the model answers once.
    """
    async with session_scope() as session:
        entries = await repo.memory_recent(session, limit=SCAN_LIMIT, kinds=SOURCE_KINDS)
        # Auto-written failure rules expire when the failure stops recurring
        # (see learning.FAILURE_RULE_TTL_DAYS). Folding them into a skill -
        # they all share the 'avoid' prefix - would make them permanent.
        entries = [e for e in entries if "failure" not in (e.tags or [])]
        if not entries:
            return []
        skills = await repo.memory_recent(session, limit=SKILL_SCAN_LIMIT, kinds=("skill",))

    synthesised_from = {skill.key: _stored_fingerprint(skill) for skill in skills}
    pending: list[tuple[str, str, list[MemoryEntry], str, int]] = []
    for topic, items in cluster_entries(entries).items():
        if len(items) < MIN_CLUSTER_SIZE:
            continue
        key = f"skill_{_clean_topic_key(topic)}"
        members = items[:SYNTHESIS_INPUTS]
        fingerprint = _fingerprint(members)
        if synthesised_from.get(key) == fingerprint:
            continue
        pending.append((topic, key, members, fingerprint, len(items)))
    if not pending:
        return []

    client = llm or get_llm()
    written: list[dict] = []
    async with session_scope() as session:
        for topic, key, members, fingerprint, size in pending[:MAX_CLUSTERS_PER_RUN]:
            summary, from_model = await _synthesise_cluster(topic, members, client)
            if looks_like_secret(summary):
                continue
            tags = (
                ["skill", "synthesized", f"{_FINGERPRINT_TAG}{fingerprint}"]
                if from_model
                else ["skill", "fallback"]
            )
            entry = await repo.memory_store(
                session,
                key=key,
                value=summary,
                kind="skill",
                tags=tags,
                source_task_id=None,
            )
            written.append({"key": entry.key, "value": entry.value, "topic": topic,
                             "source_count": size})

    if written:
        log.info("skills_synthesized", extra={"count": len(written)})
    return written
