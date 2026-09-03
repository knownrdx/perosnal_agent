"""HTTP API.

Deliberately small: health endpoints for monitoring plus a token-protected
control surface (create/list/inspect tasks) so the agent can be driven without
Telegram if needed.  Bound to 127.0.0.1 by docker-compose.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import ACTIVE_STATUSES, TaskStatus
from app.monitoring import health_snapshot


class CreateTaskRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=8000)
    chat_id: int | None = None
    permission: str = Field(default="WRITE", pattern="^(READ|WRITE|HIGH_RISK)$")


async def require_token(x_api_token: str = Header(default="")) -> None:
    settings = get_settings()
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="API_TOKEN is not configured")
    if x_api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid API token")


def create_app() -> FastAPI:
    app = FastAPI(title="Personal AI Agent", version="1.0.0", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        snapshot = await health_snapshot()
        return snapshot

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready() -> dict[str, Any]:
        snapshot = await health_snapshot()
        if not snapshot["checks"]["database"]["ok"]:
            raise HTTPException(status_code=503, detail="database unavailable")
        return {"status": "ready"}

    @app.get("/api/tasks", dependencies=[Depends(require_token)])
    async def list_tasks(active_only: bool = False, limit: int = 20) -> dict[str, Any]:
        statuses = [s.value for s in ACTIVE_STATUSES] if active_only else None
        async with session_scope() as session:
            rows = await repo.list_tasks(session, statuses=statuses, limit=min(limit, 100))
            return {"count": len(rows), "tasks": [t.short() for t in rows]}

    @app.post("/api/tasks", dependencies=[Depends(require_token)], status_code=201)
    async def create_task(payload: CreateTaskRequest) -> dict[str, Any]:
        settings = get_settings()
        async with session_scope() as session:
            task = await repo.create_task(
                session,
                user_request=payload.instruction,
                title=payload.instruction[:80],
                chat_id=payload.chat_id or settings.owner_chat_id,
                permission=payload.permission,
                max_steps=settings.max_task_steps,
                max_retries=settings.max_task_retries,
                context={"source": "api"},
            )
            return {"task_id": task.id, "status": task.status}

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(require_token)])
    async def get_task(task_id: str) -> dict[str, Any]:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            calls = await repo.list_tool_calls(session, task.id, limit=50)
            return {
                **task.short(),
                "request": task.user_request,
                "result": task.result,
                "error": task.error,
                "failure_kind": task.failure_kind,
                "output_files": list(task.output_files or []),
                "retry_count": task.retry_count,
                "tool_calls": [
                    {
                        "step": c.step,
                        "tool": c.tool,
                        "status": c.status,
                        "attempts": c.attempts,
                        "duration_ms": round(c.duration_ms, 1),
                        "error": c.error[:500],
                    }
                    for c in calls
                ],
            }

    @app.post("/api/tasks/{task_id}/cancel", dependencies=[Depends(require_token)])
    async def cancel_task(task_id: str) -> dict[str, Any]:
        async with session_scope() as session:
            task = await repo.get_task(session, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            await repo.set_task_status(
                session, task.id, TaskStatus.CANCELLED, error="cancelled via API"
            )
            return {"cancelled": True, "task_id": task.id}

    # --- bridges: inbound webhooks ------------------------------------- #
    @app.post("/webhooks/{channel}")
    async def inbound_webhook(
        channel: str, request: Request, x_bridge_token: str = Header(default="")
    ) -> dict[str, Any]:
        settings = get_settings()
        if not settings.bridge_token:
            raise HTTPException(status_code=503, detail="BRIDGE_TOKEN is not configured")
        if x_bridge_token != settings.bridge_token:
            raise HTTPException(status_code=401, detail="invalid bridge token")

        from app.integrations import handle_inbound
        from app.telegram.notifier import Notifier

        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        return await handle_inbound(channel, payload, notifier=Notifier())

    @app.get("/api/inbound", dependencies=[Depends(require_token)])
    async def list_inbound(channel: str = "", limit: int = 20) -> dict[str, Any]:
        async with session_scope() as session:
            rows = await repo.list_inbound(
                session, channel=channel or None, limit=min(limit, 100)
            )
            return {
                "count": len(rows),
                "messages": [
                    {
                        "id": r.id, "channel": r.channel, "sender": r.sender,
                        "sender_name": r.sender_name, "text": r.text[:500],
                        "media_path": r.media_path, "handled": r.handled,
                        "task_id": r.task_id,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ],
            }

    # --- LLM provider control ------------------------------------------ #
    @app.get("/api/llm", dependencies=[Depends(require_token)])
    async def llm_state() -> dict[str, Any]:
        from app.llm import get_manager

        manager = get_manager()
        return {
            "active_provider": manager.active_key(),
            "active_model": manager.active_model(),
            "fallback": manager.fallback_order()[1:],
            "providers": [
                {"key": p.key, "label": p.label, "model": p.model, "configured": p.configured}
                for p in manager.configured_providers()
            ],
        }

    @app.post("/api/llm", dependencies=[Depends(require_token)])
    async def llm_switch(payload: dict = Body(...)) -> dict[str, Any]:
        from app.llm import LLMError, get_manager

        manager = get_manager()
        provider = str(payload.get("provider") or manager.active_key())
        model = payload.get("model")
        try:
            info = await manager.set_active(provider, str(model) if model else None)
        except LLMError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"active_provider": info.key, "active_model": manager.active_model()}

    @app.get("/api/llm/health", dependencies=[Depends(require_token)])
    async def llm_health() -> dict[str, Any]:
        from app.llm import get_manager

        return {"providers": await get_manager().health_all()}

    @app.get("/api/tools", dependencies=[Depends(require_token)])
    async def list_tools() -> dict[str, Any]:
        from app.tools import registry

        return {"count": len(registry.names()), "tools": registry.specs()}

    @app.get("/api/approvals", dependencies=[Depends(require_token)])
    async def approvals() -> dict[str, Any]:
        async with session_scope() as session:
            rows = await repo.list_pending_approvals(session)
            return {
                "count": len(rows),
                "approvals": [
                    {"id": a.id, "task_id": a.task_id, "tool": a.tool, "args": a.args}
                    for a in rows
                ],
            }

    @app.post("/api/approvals/{approval_id}", dependencies=[Depends(require_token)])
    async def decide(approval_id: str, approve: bool = True) -> dict[str, Any]:
        async with session_scope() as session:
            approval = await repo.decide_approval(
                session, approval_id, approved=approve, user_id=None
            )
            if approval is None:
                raise HTTPException(status_code=404, detail="approval not found")
            await repo.update_task(session, approval.task_id, status=TaskStatus.PENDING.value)
            tool, decided_args = approval.tool, dict(approval.args or {})
            payload = {"approval_id": approval.id, "status": approval.status}

        from app.security import autonomy

        await autonomy.remember_decision(tool, decided_args, approved=approve)
        return payload

    return app
