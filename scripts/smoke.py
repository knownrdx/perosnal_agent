#!/usr/bin/env python
"""Smoke test: boot the real application process and exercise it over HTTP.

Runs fully offline (SQLite + echo LLM + no Telegram token) so it can be used to
verify a deployment before wiring real credentials.

    python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="agent-smoke-"))
os.environ.update(
    {
        "WORKSPACE_DIR": str(TMP / "data"),
        "DATABASE_URL": f"sqlite+aiosqlite:///{(TMP / 'agent.sqlite').as_posix()}",
        "LLM_PROVIDER": "echo",
        "API_TOKEN": "smoke-token",
        "API_PORT": "8099",
        "API_HOST": "127.0.0.1",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_ALLOWED_USER_IDS": "",
        "START_TELEGRAM_BOT": "false",
        "LOG_LEVEL": "WARNING",
    }
)

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.logging_conf import setup_logging  # noqa: E402
from app.main import Application  # noqa: E402

BASE = "http://127.0.0.1:8099"
HEADERS = {"X-API-Token": "smoke-token"}
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


async def wait_for_api(client: httpx.AsyncClient, attempts: int = 60) -> bool:
    for _ in range(attempts):
        try:
            response = await client.get(f"{BASE}/health/live", timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.25)
    return False


async def main() -> int:
    setup_logging("WARNING")
    get_settings.cache_clear()

    app = Application()
    runner = asyncio.create_task(app.run())
    print("\nBooting the real agent process...\n")

    async with httpx.AsyncClient() as client:
        try:
            if not await wait_for_api(client):
                check("application boots and serves HTTP", False, "no response on /health/live")
                return 1
            check("application boots and serves HTTP", True)

            health = (await client.get(f"{BASE}/health")).json()
            check("database reachable", health["checks"]["database"]["ok"])
            check("worker running", "task_worker" in health["workers"]["running"],
                  health["workers"]["running"])
            check("scheduler running", "scheduler" in health["workers"]["running"])

            unauth = await client.get(f"{BASE}/api/tasks")
            check("API rejects requests without a token", unauth.status_code == 401)

            tools = (await client.get(f"{BASE}/api/tools", headers=HEADERS)).json()
            check("tools registered", tools["count"] >= 15, f"{tools['count']} tools")

            created = await client.post(
                f"{BASE}/api/tasks",
                json={"instruction": "write a file named output/smoke.txt"},
                headers=HEADERS,
            )
            task_id = created.json()["task_id"]
            check("task created via API", created.status_code == 201, task_id)

            final = {}
            for _ in range(80):
                final = (await client.get(f"{BASE}/api/tasks/{task_id}", headers=HEADERS)).json()
                if final["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    break
                await asyncio.sleep(0.25)

            check("worker executed the task", final.get("status") == "COMPLETED",
                  f"status={final.get('status')} {str(final.get('error'))[:120]}")
            check("tool call recorded", bool(final.get("tool_calls")),
                  ", ".join(c["tool"] for c in final.get("tool_calls", [])))

            produced = Path(os.environ["WORKSPACE_DIR"]) / "output" / "echo.txt"
            check("file actually written to workspace", produced.exists(),
                  str(produced) if produced.exists() else "missing")
        finally:
            app._stop.set()
            if app.server is not None:
                app.server.should_exit = True
            try:
                await asyncio.wait_for(runner, timeout=20)
            except Exception:  # noqa: BLE001
                runner.cancel()

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("SMOKE TEST OK - the agent boots, serves, and completes a real task.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
