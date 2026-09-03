"""Memory tools (long-term, non-sensitive, keyword search)."""

from __future__ import annotations

import re
from typing import Any

from app.db.base import session_scope
from app.db import repo
from app.security import Permission
from app.tools.base import Arg, InvalidInput, ToolContext
from app.tools.registry import tool

# Refuse to persist anything that looks like a credential (master prompt 15).
_SECRET_PATTERNS = [
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),
    re.compile(r"\b(sk|pk|xoxb|ghp|gho|hf)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret|bot[_-]?token|access[_-]?token)\b\s*[:=]"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def looks_like_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


@tool(
    "memory_store",
    description=(
        "Save a durable, NON-SENSITIVE fact or preference for future tasks. "
        "Never store passwords, tokens, keys or cookies."
    ),
    permission=Permission.WRITE,
    args={
        "key": Arg("string", True, "Short unique key, e.g. 'preferred_report_format'"),
        "value": Arg("string", True, "The fact to remember"),
        "kind": Arg("string", False, "fact | preference | workflow | decision", default="fact"),
        "tags": Arg("array", False, "Optional tags", default=None),
    },
    timeout_s=30,
    max_retries=0,
)
async def memory_store_tool(
    key: str,
    value: str,
    kind: str = "fact",
    tags: list[str] | None = None,
    ctx: ToolContext | None = None,
) -> dict[str, Any]:
    if looks_like_secret(f"{key} {value}"):
        raise InvalidInput("refusing to store credentials in memory")
    if len(value) > 4000:
        raise InvalidInput("memory value too long (max 4000 chars)")
    async with session_scope() as session:
        entry = await repo.memory_store(
            session,
            key=key.strip()[:200],
            value=value.strip(),
            kind=kind,
            tags=tags or [],
            source_task_id=ctx.task_id if ctx else None,
        )
        return {"stored": True, "key": entry.key, "kind": entry.kind}


@tool(
    "memory_search",
    description="Search long-term memory by keyword.",
    permission=Permission.READ,
    args={
        "query": Arg("string", True, "Keyword to search for"),
        "limit": Arg("integer", False, "Max results", default=5),
    },
    timeout_s=30,
    max_retries=0,
)
async def memory_search_tool(query: str, limit: int = 5) -> dict[str, Any]:
    async with session_scope() as session:
        rows = await repo.memory_search(session, query, limit=max(1, min(limit, 25)))
        return {
            "count": len(rows),
            "results": [{"key": r.key, "value": r.value, "kind": r.kind} for r in rows],
        }
