"""Telegram interface: import safety, allowlist enforcement, notifications."""

from __future__ import annotations

import importlib

import pytest

from app.db import repo
from app.db.base import session_scope


def test_all_modules_import_cleanly(environment):
    """Catches syntax/import errors in modules the other tests never load."""
    for name in [
        "app.main",
        "app.api",
        "app.telegram.bot",
        "app.telegram.notifier",
        "app.tools.browser_tools",
        "app.workers.task_worker",
        "app.workers.scheduler_worker",
        "app.monitoring",
    ]:
        assert importlib.import_module(name) is not None


async def test_unauthorized_user_is_rejected(environment):
    from app.telegram.bot import AgentBot

    bot = AgentBot()
    try:
        class _User:
            def __init__(self, uid: int) -> None:
                self.id = uid

        class _Message:
            def __init__(self, uid: int) -> None:
                self.from_user = _User(uid)
                self.replies: list[str] = []

            async def answer(self, text: str, **kwargs) -> None:
                self.replies.append(text)

        stranger = _Message(999999)
        assert await bot._guard(stranger) is False
        assert "Not authorized" in stranger.replies[0]

        owner = _Message(42)
        assert await bot._guard(owner) is True
    finally:
        await bot.stop()


async def test_rate_limit_blocks_flood(environment):
    from app.telegram.bot import AgentBot

    bot = AgentBot()
    try:
        limit = environment.telegram_rate_limit_per_min
        for _ in range(limit):
            assert bot._rate_ok(42) is True
        assert bot._rate_ok(42) is False
    finally:
        await bot.stop()


async def test_notifier_reports_failure_with_reason(environment, fake_telegram):
    from app.db.models import FailureKind, TaskStatus
    from app.telegram.notifier import Notifier

    async with session_scope() as session:
        task = await repo.create_task(session, user_request="doomed job", chat_id=42)
        task_id = task.id
        await repo.set_task_status(
            session, task_id, TaskStatus.FAILED,
            error="remote server refused the connection",
            failure_kind=FailureKind.TEMPORARY.value,
        )

    await Notifier().task_failed(task_id)
    messages = fake_telegram.sent_messages()
    assert len(messages) == 1
    assert "Task failed" in messages[0]
    assert "TEMPORARY" in messages[0]
    assert "refused the connection" in messages[0]


async def test_notifier_without_chat_id_does_not_crash(environment, fake_telegram, monkeypatch):
    from app.config import reload_settings
    from app.telegram.notifier import Notifier

    monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")
    reload_settings()
    try:
        result = await Notifier().send(None, "orphan message")
        assert result["sent"] is False
    finally:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42")
        monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "42")
        reload_settings()


async def test_approval_request_notification_contains_commands(environment, fake_telegram):
    from app.telegram.notifier import Notifier

    await Notifier().approval_request(
        chat_id=42, approval_id="abc123", task_id="task9",
        tool="file_delete", args={"path": "output/x.txt"},
    )
    text = fake_telegram.sent_messages()[0]
    assert "Approval required" in text
    assert "/approve abc123" in text and "/reject abc123" in text
    assert "file_delete" in text
