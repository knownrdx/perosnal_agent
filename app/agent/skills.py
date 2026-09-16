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
"""

from __future__ import annotations

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


async def _synthesise_cluster(topic: str, entries: list[MemoryEntry], llm: Any) -> str:
    source_lines = "\n".join(f"- ({e.kind}) {e.key}: {e.value}" for e in entries[:12])
    try:
        data = await llm.chat_json(
            [Message("system", SYNTHESIS_PROMPT), Message("user", source_lines)]
        )
        summary = str(data.get("summary", "")).strip()
        if not summary or len(summary) > 2000 or looks_like_secret(summary):
            raise ValueError("unusable synthesis output")
        return summary
    except LLMError as exc:
        log.warning("skill_synthesis_llm_failed", extra={"topic": topic, "error": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001 - learning must never break other things
        log.warning("skill_synthesis_error", extra={"topic": topic, "error": str(exc)[:200]})
    return _fallback_summary(topic, entries)


def _clean_topic_key(topic: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", topic.lower().strip())
    return re.sub(r"_+", "_", key).strip("_")[:80] or "general"


async def synthesize_skills(llm: Any | None = None) -> list[dict]:
    """Consolidate recent fine-grained lessons into coarse 'skill' entries.

    Reads recent workflow/gotcha/preference memories, clusters them by topic,
    and for every cluster with >= MIN_CLUSTER_SIZE entries upserts ONE 'skill'
    memory entry keyed by the topic (so re-running updates rather than
    duplicates). Returns what was written/updated, for logging/inspection.
    """
    async with session_scope() as session:
        entries = await repo.memory_recent(session, limit=SCAN_LIMIT, kinds=SOURCE_KINDS)

    if not entries:
        return []

    clusters = cluster_entries(entries)
    eligible = [
        (topic, items) for topic, items in clusters.items() if len(items) >= MIN_CLUSTER_SIZE
    ]
    if not eligible:
        return []

    client = llm or get_llm()
    written: list[dict] = []
    async with session_scope() as session:
        for topic, items in eligible[:MAX_CLUSTERS_PER_RUN]:
            summary = await _synthesise_cluster(topic, items, client)
            if looks_like_secret(summary):
                continue
            key = f"skill_{_clean_topic_key(topic)}"
            entry = await repo.memory_store(
                session,
                key=key,
                value=summary,
                kind="skill",
                tags=["skill", "synthesized"],
                source_task_id=None,
            )
            written.append({"key": entry.key, "value": entry.value, "topic": topic,
                             "source_count": len(items)})

    if written:
        log.info("skills_synthesized", extra={"count": len(written)})
    return written
