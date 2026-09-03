"""Tool framework + individual tool behaviour."""

from __future__ import annotations

import pytest

from app.security import Permission
from app.tools import registry
from app.tools.base import Arg, TemporaryToolError, Tool, ToolContext


async def test_registry_has_core_tools(environment):
    names = registry.names()
    for expected in [
        "file_write", "file_read", "file_download", "file_delete",
        "telegram_send_message", "telegram_send_file",
        "task_create", "task_status", "scheduler_create",
        "memory_store", "memory_search",
    ]:
        assert expected in names, f"missing tool: {expected}"


async def test_tool_validates_missing_and_unknown_args(environment):
    tool = registry.get("file_write")
    result = await tool.run({"path": "output/x.txt"}, ToolContext())
    assert not result.ok and "content" in result.error

    result = await tool.run(
        {"path": "output/x.txt", "content": "hi", "bogus": 1}, ToolContext()
    )
    assert not result.ok and "unknown argument" in result.error


async def test_file_write_read_roundtrip(environment):
    write = registry.get("file_write")
    read = registry.get("file_read")

    result = await write.run({"path": "output/a.txt", "content": "hello world"}, ToolContext())
    assert result.ok and result.data["size_bytes"] == 11
    assert result.data["sha256"]

    back = await read.run({"path": "output/a.txt"}, ToolContext())
    assert back.ok and back.data["content"] == "hello world"


async def test_file_tools_block_traversal(environment):
    write = registry.get("file_write")
    result = await write.run({"path": "../escape.txt", "content": "x"}, ToolContext())
    assert not result.ok and "escapes workspace" in result.error


async def test_retry_then_success(environment):
    attempts = {"n": 0}

    async def flaky() -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TemporaryToolError("boom")
        return {"ok": True}

    tool = Tool(
        name="flaky", description="", args={}, permission=Permission.READ,
        handler=flaky, timeout_s=5, max_retries=2, retry_backoff_s=0.01,
    )
    result = await tool.run({}, ToolContext())
    assert result.ok and result.attempts == 3


async def test_permanent_error_is_not_retried(environment):
    from app.tools.base import PermanentToolError

    attempts = {"n": 0}

    async def broken() -> dict:
        attempts["n"] += 1
        raise PermanentToolError("nope")

    tool = Tool(
        name="broken", description="", args={}, permission=Permission.READ,
        handler=broken, timeout_s=5, max_retries=3, retry_backoff_s=0.01,
    )
    result = await tool.run({}, ToolContext())
    assert not result.ok and attempts["n"] == 1


async def test_timeout_is_enforced(environment):
    import asyncio

    async def slow() -> dict:
        await asyncio.sleep(5)
        return {}

    tool = Tool(
        name="slow", description="", args={}, permission=Permission.READ,
        handler=slow, timeout_s=1, max_retries=0, retry_backoff_s=0.01,
    )
    result = await tool.run({}, ToolContext())
    assert not result.ok and "timed out" in result.error


async def test_verify_hook_rejects_bad_result(environment):
    async def liar() -> dict:
        return {"size_bytes": 0}

    tool = Tool(
        name="liar", description="", args={}, permission=Permission.READ,
        handler=liar, timeout_s=5, max_retries=0, retry_backoff_s=0.01,
        verify=lambda data: data.get("size_bytes", 0) > 0,
    )
    result = await tool.run({}, ToolContext())
    assert not result.ok and "verification failed" in result.error


async def test_shell_allowlist_and_metacharacters(environment):
    shell = registry.get("safe_shell_execute")
    assert shell is not None

    blocked = await shell.run({"command": "rm -rf /"}, ToolContext())
    assert not blocked.ok and "not allowlisted" in blocked.error

    chained = await shell.run({"command": "ls; rm -rf /"}, ToolContext())
    assert not chained.ok and "metacharacters" in chained.error

    piped = await shell.run({"command": "cat /etc/passwd | grep root"}, ToolContext())
    assert not piped.ok

    escape = await shell.run({"command": "cat ../../../etc/passwd"}, ToolContext())
    assert not escape.ok

    write_tool = registry.get("file_write")
    await write_tool.run({"path": "output/list_me.txt", "content": "data"}, ToolContext())
    allowed = await shell.run({"command": "ls output"}, ToolContext())
    assert allowed.ok and "list_me.txt" in allowed.data["stdout"]


async def test_python_execute_runs_and_captures_output(environment):
    py = registry.get("python_execute")
    result = await py.run({"code": "print('sum', 2 + 2)"}, ToolContext())
    assert result.ok and "sum 4" in result.data["stdout"]


async def test_python_execute_reports_failure(environment):
    py = registry.get("python_execute")
    result = await py.run({"code": "raise SystemExit(3)"}, ToolContext())
    assert not result.ok


async def test_telegram_send_message_verified(environment, fake_telegram):
    tool = registry.get("telegram_send_message")
    result = await tool.run({"text": "hello"}, ToolContext(chat_id=42))
    assert result.ok and result.data["message_id"] > 0
    assert fake_telegram.sent_messages() == ["hello"]


async def test_telegram_send_file_rejects_empty_and_sends_real_file(environment, fake_telegram):
    from app.security import safe_path

    empty = safe_path("output/empty.bin")
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")

    tool = registry.get("telegram_send_file")
    bad = await tool.run({"path": "output/empty.bin"}, ToolContext(chat_id=42))
    assert not bad.ok and "empty" in bad.error

    good_path = safe_path("output/report.txt")
    good_path.write_text("real content", encoding="utf-8")
    good = await tool.run({"path": "output/report.txt", "caption": "here"}, ToolContext(chat_id=42))
    assert good.ok and good.data["message_id"] and good.data["file_id"]


async def test_telegram_retries_transient_failure(environment, fake_telegram):
    fake_telegram.fail_times = 1
    tool = registry.get("telegram_send_message")
    tool.retry_backoff_s = 0.01
    result = await tool.run({"text": "retry me"}, ToolContext(chat_id=42))
    assert result.ok and result.attempts == 2


async def test_memory_tools_store_and_search(environment):
    store = registry.get("memory_store")
    search = registry.get("memory_search")

    ok = await store.run(
        {"key": "report_format", "value": "owner prefers PDF"}, ToolContext(task_id="t1")
    )
    assert ok.ok

    denied = await store.run(
        {"key": "creds", "value": "password = hunter2"}, ToolContext(task_id="t1")
    )
    assert not denied.ok and "credentials" in denied.error

    found = await search.run({"query": "pdf"}, ToolContext())
    assert found.ok and found.data["count"] == 1
