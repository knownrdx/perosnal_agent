"""Shared pytest fixtures.

The whole suite runs offline: SQLite instead of PostgreSQL, the echo LLM
instead of Ollama, and a fake Telegram API instead of the real Bot API.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def environment(tmp_path, monkeypatch):
    """Fresh workspace + fresh SQLite database for every test."""
    workspace = tmp_path / "data"
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "agent.sqlite"

    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("LLM_PROVIDER", "echo")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "42")
    monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "42")
    monkeypatch.setenv("API_TOKEN", "test-api-token")
    monkeypatch.setenv("REQUIRE_APPROVAL_HIGH_RISK", "true")
    monkeypatch.setenv("MAX_TASK_STEPS", "8")
    monkeypatch.setenv("START_TELEGRAM_BOT", "false")
    monkeypatch.setenv("START_WORKER", "false")
    monkeypatch.setenv("START_SCHEDULER", "false")

    from app.config import reload_settings
    from app.db.base import create_all, dispose_engine, init_engine

    settings = reload_settings()
    settings.ensure_workspace()

    await dispose_engine()
    init_engine(settings.database_url)
    await create_all()

    yield settings

    await dispose_engine()
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def echo_llm():
    from app.llm import EchoClient, set_llm

    client = EchoClient()
    set_llm(client)
    yield client
    set_llm(None)


class FakeTelegramAPI:
    """Records calls and returns Bot-API-shaped responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict | None]] = []
        self.next_message_id = 1000
        self.fail_times = 0

    async def call(self, method: str, data: dict | None = None, files: dict | None = None) -> dict:
        self.calls.append((method, dict(data or {}), files))
        if self.fail_times > 0:
            self.fail_times -= 1
            from app.tools.base import TemporaryToolError

            raise TemporaryToolError("simulated telegram network error")
        self.next_message_id += 1
        result: dict = {"message_id": self.next_message_id, "chat": {"id": (data or {}).get("chat_id")}}
        if method == "sendDocument":
            name = "file.bin"
            if files and "document" in files:
                name = files["document"][0]
            result["document"] = {"file_id": f"fileid-{self.next_message_id}", "file_name": name}
        return result

    async def close(self) -> None:
        return None

    def sent_documents(self) -> list[dict]:
        return [data for method, data, _ in self.calls if method == "sendDocument"]

    def sent_messages(self) -> list[str]:
        return [data.get("text", "") for method, data, _ in self.calls if method == "sendMessage"]


@pytest.fixture
def fake_telegram():
    from app.tools.telegram_tools import set_api

    api = FakeTelegramAPI()
    set_api(api)
    yield api
    set_api(None)
