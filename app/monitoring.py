"""Health / monitoring.

Small on purpose (master prompt section 26): one snapshot function used by the
HTTP endpoints and by /status in Telegram.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import text

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.logging_conf import get_logger

log = get_logger(__name__)

START_TIME = time.time()

# Populated by app.main so /health can report what is actually running.
RUNTIME: dict[str, Any] = {"worker": None, "scheduler": None, "bot": None}


async def check_database() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300]}


async def check_llm() -> dict[str, Any]:
    """Health probe with a hard timeout.

    A gateway that hangs (rather than erroring) instead of the normal fast
    401/502 would otherwise make every /health call - including the reverse
    proxy's own healthcheck and the public domain's first byte - wait on it.
    Timing out and reporting unhealthy is strictly better than a slow health
    endpoint pretending everything is fine.
    """
    import asyncio

    from app.llm import get_llm

    try:
        return await asyncio.wait_for(get_llm().health(), timeout=5.0)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "health probe timed out after 5s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300]}


def resources() -> dict[str, Any]:
    settings = get_settings()
    try:
        import psutil

        disk = psutil.disk_usage(str(settings.workspace) if settings.workspace.exists() else "/")
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": memory.percent,
            "memory_available_mb": round(memory.available / 1048576),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1073741824, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def workers_state() -> dict[str, Any]:
    worker = RUNTIME.get("worker")
    scheduler = RUNTIME.get("scheduler")
    bot = RUNTIME.get("bot")
    running = []
    if worker is not None and getattr(worker, "_tasks", None):
        running.append(f"task_worker x{len(worker._tasks)}")
    if scheduler is not None and getattr(scheduler, "_task", None) is not None:
        running.append("scheduler")
    if bot is not None and getattr(bot, "_task", None) is not None:
        running.append("telegram")
    return {"running": ", ".join(running) or "none", "count": len(running)}


async def check_bridges() -> dict[str, Any]:
    """Health of the optional WhatsApp / Teams bridges."""
    settings = get_settings()
    out: dict[str, Any] = {}
    if settings.whatsapp_enabled:
        from app.integrations import whatsapp_bridge

        out["whatsapp"] = await whatsapp_bridge().health()
    if settings.teams_enabled:
        from app.integrations import teams_bridge

        out["teams"] = await teams_bridge().health()
    return out


async def health_snapshot() -> dict[str, Any]:
    database = await check_database()
    llm = await check_llm()
    bridges = await check_bridges()
    try:
        async with session_scope() as session:
            task_stats = await repo.stats(session)
    except Exception as exc:  # noqa: BLE001
        task_stats = {"error": str(exc)[:200], "tasks_by_status": {}, "active_tasks": 0,
                      "pending_approvals": 0, "enabled_jobs": 0, "memory_entries": 0}

    res = resources()
    degraded = (
        not database["ok"]
        or res.get("disk_percent", 0) > 95
        or res.get("memory_percent", 0) > 97
    )
    return {
        "ok": not degraded,
        "uptime_s": int(time.time() - START_TIME),
        "checks": {"database": database, "llm": llm, "bridges": bridges},
        "resources": res,
        "workers": workers_state(),
        "stats": task_stats,
    }
