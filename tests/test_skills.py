"""Self-updating skills memory layer: clustering + synthesis."""

from __future__ import annotations

import pytest

from app.agent.skills import cluster_entries, synthesize_skills
from app.db import repo
from app.db.base import session_scope
from app.llm import LLMError, Message


class EchoingLLM:
    """Returns a scripted synthesis, or raises to exercise the fallback."""

    def __init__(self, summary: str | None = "PDF reports go to the finance folder.") -> None:
        self.summary = summary
        self.calls = 0

    async def chat_json(self, messages: list[Message], *, temperature=None):
        self.calls += 1
        if self.summary is None:
            raise LLMError("simulated model outage")
        return {"summary": self.summary}


async def _seed_lessons(prefix: str, count: int, kind: str = "workflow") -> None:
    async with session_scope() as session:
        for i in range(count):
            await repo.memory_store(
                session,
                key=f"{prefix}_{i}",
                value=f"{prefix} lesson number {i}",
                kind=kind,
                tags=["learned"],
            )


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_cluster_entries_groups_by_leading_token():
    from app.db.models import MemoryEntry

    entries = [
        MemoryEntry(key="report_format", value="v1", kind="workflow"),
        MemoryEntry(key="report_source", value="v2", kind="workflow"),
        MemoryEntry(key="report_delivery", value="v3", kind="preference"),
        MemoryEntry(key="invoice_currency", value="v4", kind="gotcha"),
    ]
    clusters = cluster_entries(entries)
    assert len(clusters["report"]) == 3
    assert len(clusters["invoice"]) == 1


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
async def test_synthesize_skills_ignores_small_clusters(environment):
    await _seed_lessons("report", 2)
    written = await synthesize_skills(llm=EchoingLLM())
    assert written == []

    async with session_scope() as session:
        skills = await repo.memory_recent(session, limit=20, kinds=["skill"])
    assert skills == []


async def test_synthesize_skills_creates_one_entry_per_cluster(environment):
    await _seed_lessons("report", 3)
    await _seed_lessons("invoice", 4)

    llm = EchoingLLM()
    written = await synthesize_skills(llm=llm)
    assert len(written) == 2
    assert llm.calls == 2

    async with session_scope() as session:
        skills = await repo.memory_recent(session, limit=20, kinds=["skill"])
    assert len(skills) == 2
    keys = {s.key for s in skills}
    assert keys == {"skill_report", "skill_invoice"}


async def test_synthesize_skills_upserts_not_duplicates(environment):
    await _seed_lessons("report", 3)
    llm = EchoingLLM(summary="first summary")
    await synthesize_skills(llm=llm)

    llm2 = EchoingLLM(summary="second, refined summary")
    await synthesize_skills(llm=llm2)

    async with session_scope() as session:
        skills = await repo.memory_recent(session, limit=20, kinds=["skill"])
    assert len(skills) == 1, "re-running must update the same entry, not duplicate"
    assert skills[0].value == "second, refined summary"


async def test_synthesize_skills_falls_back_when_llm_fails(environment):
    await _seed_lessons("report", 3)
    written = await synthesize_skills(llm=EchoingLLM(summary=None))
    assert len(written) == 1

    async with session_scope() as session:
        skills = await repo.memory_recent(session, limit=20, kinds=["skill"])
    assert len(skills) == 1
    # Deterministic fallback: a bullet-list join of the source entries.
    assert "report_0" in skills[0].value or "report lesson number 0" in skills[0].value


async def test_synthesize_skills_never_stores_secrets(environment):
    async with session_scope() as session:
        for i in range(3):
            await repo.memory_store(
                session, key=f"portal_login_{i}",
                value=f"password = hunter{i}", kind="gotcha",
            )
    written = await synthesize_skills(llm=EchoingLLM(summary="password = hunter2 for portal"))
    assert written == []

    async with session_scope() as session:
        skills = await repo.memory_recent(session, limit=20, kinds=["skill"])
    assert skills == []


async def test_synthesize_skills_returns_empty_with_no_lessons(environment):
    assert await synthesize_skills(llm=EchoingLLM()) == []


# --------------------------------------------------------------------------- #
# /skills command
# --------------------------------------------------------------------------- #
async def test_skills_command_lists_entries(environment, fake_telegram):
    from app.telegram.bot import AgentBot

    async with session_scope() as session:
        await repo.memory_store(
            session, key="skill_report", value="Reports go out as PDF.", kind="skill",
        )

    bot = AgentBot()
    try:
        commands = {
            filt.callback.commands
            for handler in bot.dp.message.handlers
            for filt in handler.filters
            if hasattr(filt.callback, "commands")
        }
        flat = {c for group in commands for c in group}
        assert "skills" in flat
    finally:
        await bot.stop()
