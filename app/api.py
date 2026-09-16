"""HTTP API.

Deliberately small: health endpoints for monitoring plus a control surface
(tasks, LLM, contacts, skills, scheduled jobs, autonomy, bridge status) so the
agent can be driven without Telegram if needed.  Bound to 127.0.0.1 by
docker-compose.

Two ways to authenticate against the control surface:
  1. The X-API-Token header (script/API clients) - see ``require_token``.
  2. A signed ``agent_session`` cookie obtained via POST /api/auth/login with
     the WEB_UI_PASSWORD (browser dashboard clients) - see the auth endpoints
     below and ``require_api_access``, which accepts either.

If WEB_UI_PASSWORD is unset, the web UI login surface is disabled (503) and no
valid session cookie can ever exist, so token-only deployments are unaffected.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
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
    """Original token-only dependency. Kept for backward compatibility with
    anything importing it directly; routes below use ``require_api_access``.
    """
    settings = get_settings()
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="API_TOKEN is not configured")
    if x_api_token != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid API token")


# --------------------------------------------------------------------------- #
# Web UI session cookies: stdlib-only signed tokens (hmac + base64 + json).
# --------------------------------------------------------------------------- #
SESSION_COOKIE_NAME = "agent_session"
SESSION_TTL_S = 30 * 24 * 3600  # 30 days
SESSION_PURPOSE = "agent_session_v1"


def _session_secret_file() -> Path:
    return get_settings().workspace / ".web_ui_session_secret"


def _get_session_secret() -> str:
    """Return the session-signing secret, creating + persisting one on first
    use if WEB_UI_SESSION_SECRET is unset. Mirrors
    app/security/vault.py's ``_load_or_create_key`` so a random secret
    survives process restarts (otherwise every login is invalidated on
    redeploy).
    """
    settings = get_settings()
    if settings.web_ui_session_secret:
        return settings.web_ui_session_secret

    path = _session_secret_file()
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            settings.web_ui_session_secret = raw
            return raw

    generated = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - windows / odd filesystems
        pass
    settings.web_ui_session_secret = generated
    return generated


def _sign(message: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _make_session_token() -> str:
    secret = _get_session_secret()
    payload = {"exp": int(time.time()) + SESSION_TTL_S, "purpose": SESSION_PURPOSE}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = _sign(body.encode("ascii"), secret)
    return f"{body}.{signature}"


def _verify_session_token(token: str) -> bool:
    """True only for an unexpired, correctly signed token. If no web UI
    password is configured, cookie auth never succeeds (no login is ever
    possible in that case, so this is mostly a defensive extra check).
    """
    settings = get_settings()
    if not settings.web_ui_password or not token or "." not in token:
        return False

    body, _, signature = token.partition(".")
    secret = _get_session_secret()
    expected = _sign(body.encode("ascii"), secret)
    if not hmac.compare_digest(signature, expected):
        return False

    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception:  # noqa: BLE001 - any decode failure means invalid token
        return False

    if not isinstance(payload, dict) or payload.get("purpose") != SESSION_PURPOSE:
        return False
    expires_at = payload.get("exp")
    if not isinstance(expires_at, (int, float)) or expires_at < time.time():
        return False
    return True


async def require_api_access(
    request: Request, x_api_token: str = Header(default="")
) -> None:
    """Succeeds if EITHER the X-API-Token header OR a valid ``agent_session``
    cookie is present. Token-only deployments (no WEB_UI_PASSWORD configured)
    behave exactly like the original ``require_token`` dependency, since no
    valid cookie can ever exist in that case.
    """
    cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
    if _verify_session_token(cookie):
        return

    settings = get_settings()
    if settings.api_token:
        if hmac.compare_digest(x_api_token, settings.api_token):
            return
        raise HTTPException(status_code=401, detail="invalid API token")

    if not settings.web_ui_password:
        raise HTTPException(status_code=503, detail="API_TOKEN is not configured")
    raise HTTPException(status_code=401, detail="invalid API token")


class LoginRequest(BaseModel):
    password: str


class AutonomyRequest(BaseModel):
    level: str


class SchedulerCreateRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=8000)
    kind: str = Field(default="once", pattern="^(once|interval|cron)$")
    when: str = ""
    every: str = ""
    cron: str = ""
    name: str = ""
    max_runs: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class OtpBotConfigRequest(BaseModel):
    enabled: bool | None = None
    target_bot: str | None = None
    add_command_template: str | None = None
    limit: int | None = None
    count: int | None = None
    quota_command: str | None = None
    quota_threshold: int | None = None
    cleanup_command: str | None = None
    interval_minutes: int | None = None


class OtpBotTagRequest(BaseModel):
    tag: str = Field(min_length=1, max_length=60)


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

    # --- web UI session auth -------------------------------------------- #
    @app.post("/api/auth/login")
    async def auth_login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        settings = get_settings()
        if not settings.web_ui_password:
            raise HTTPException(status_code=503, detail="web UI is not configured")

        given_hash = hashlib.sha256(payload.password.encode("utf-8")).digest()
        expected_hash = hashlib.sha256(settings.web_ui_password.encode("utf-8")).digest()
        if not hmac.compare_digest(given_hash, expected_hash):
            raise HTTPException(status_code=401, detail="incorrect password")

        token = _make_session_token()
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=SESSION_TTL_S,
            httponly=True,
            samesite="lax",
        )
        return {"ok": True}

    @app.post("/api/auth/logout")
    async def auth_logout(response: Response) -> dict[str, Any]:
        response.delete_cookie(SESSION_COOKIE_NAME)
        return {"ok": True}

    @app.get("/api/auth/me")
    async def auth_me(request: Request) -> dict[str, Any]:
        cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
        return {"authenticated": _verify_session_token(cookie)}

    @app.get("/api/tasks", dependencies=[Depends(require_api_access)])
    async def list_tasks(active_only: bool = False, limit: int = 20) -> dict[str, Any]:
        statuses = [s.value for s in ACTIVE_STATUSES] if active_only else None
        async with session_scope() as session:
            rows = await repo.list_tasks(session, statuses=statuses, limit=min(limit, 100))
            return {"count": len(rows), "tasks": [t.short() for t in rows]}

    @app.post("/api/tasks", dependencies=[Depends(require_api_access)], status_code=201)
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

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(require_api_access)])
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

    @app.post("/api/tasks/{task_id}/cancel", dependencies=[Depends(require_api_access)])
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

    @app.get("/api/inbound", dependencies=[Depends(require_api_access)])
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
    @app.get("/api/llm", dependencies=[Depends(require_api_access)])
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

    @app.post("/api/llm", dependencies=[Depends(require_api_access)])
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

    @app.get("/api/llm/health", dependencies=[Depends(require_api_access)])
    async def llm_health() -> dict[str, Any]:
        from app.llm import get_manager

        return {"providers": await get_manager().health_all()}

    @app.get("/api/tools", dependencies=[Depends(require_api_access)])
    async def list_tools() -> dict[str, Any]:
        from app.tools import registry

        return {"count": len(registry.names()), "tools": registry.specs()}

    @app.get("/api/approvals", dependencies=[Depends(require_api_access)])
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

    @app.post("/api/approvals/{approval_id}", dependencies=[Depends(require_api_access)])
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

    # --- contacts --------------------------------------------------------- #
    @app.get("/api/contacts", dependencies=[Depends(require_api_access)])
    async def list_contacts(
        channel: str = "", query: str = "", limit: int = 50
    ) -> dict[str, Any]:
        async with session_scope() as session:
            if query:
                rows = await repo.search_contacts(session, query, limit)
            else:
                rows = await repo.list_contacts(session, channel=channel or None, limit=limit)
            return {
                "count": len(rows),
                "contacts": [
                    {
                        "channel": c.channel,
                        "external_id": c.external_id,
                        "display_name": c.display_name,
                        "username_or_phone": c.username_or_phone or "",
                        "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
                    }
                    for c in rows
                ],
            }

    @app.post("/api/contacts/sync", dependencies=[Depends(require_api_access)])
    async def contacts_sync_endpoint() -> dict[str, Any]:
        from app.tools.contact_tools import contacts_sync

        return await contacts_sync()

    # --- skills / memory ---------------------------------------------------- #
    @app.get("/api/skills", dependencies=[Depends(require_api_access)])
    async def list_skills() -> dict[str, Any]:
        async with session_scope() as session:
            rows = await repo.memory_recent(session, limit=50, kinds=["skill"])
            return {
                "count": len(rows),
                "skills": [
                    {
                        "key": row.key.removeprefix("skill_"),
                        "value": row.value,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                    for row in rows
                ],
            }

    @app.get("/api/memory", dependencies=[Depends(require_api_access)])
    async def memory(query: str = "", limit: int = 20) -> dict[str, Any]:
        async with session_scope() as session:
            if query:
                rows = await repo.memory_search(session, query, limit)
            else:
                rows = await repo.memory_recent(session, limit)
            return {
                "count": len(rows),
                "memories": [
                    {
                        "key": row.key,
                        "value": row.value,
                        "kind": row.kind,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                    for row in rows
                ],
            }

    # --- scheduled jobs ------------------------------------------------- #
    @app.get("/api/jobs", dependencies=[Depends(require_api_access)])
    async def list_jobs() -> dict[str, Any]:
        async with session_scope() as session:
            jobs = await repo.list_jobs(session, limit=50)
            return {
                "count": len(jobs),
                "jobs": [
                    {
                        "job_id": job.id,
                        "name": job.name,
                        "kind": job.kind,
                        "enabled": job.enabled,
                        "next_run_at": job.next_run_at.isoformat() if job.next_run_at else None,
                        "runs": job.runs,
                    }
                    for job in jobs
                ],
            }

    @app.post("/api/jobs", dependencies=[Depends(require_api_access)])
    async def create_job(payload: SchedulerCreateRequest) -> dict[str, Any]:
        from app.db.models import JobKind, utcnow
        from app.scheduler.timeparse import TimeParseError, next_cron, parse_duration, parse_when
        from app.tools.base import InvalidInput

        kind = payload.kind.lower()
        interval_s: int | None = None
        max_runs = payload.max_runs
        try:
            if kind == "once":
                if not payload.when:
                    raise InvalidInput("kind=once requires 'when'")
                next_run = parse_when(payload.when)
                job_kind = JobKind.ONCE
                max_runs = 1
            elif kind == "interval":
                if not payload.every:
                    raise InvalidInput("kind=interval requires 'every'")
                interval_s = parse_duration(payload.every)
                if interval_s < 60:
                    raise InvalidInput("minimum interval is 60 seconds")
                next_run = utcnow() + timedelta(seconds=interval_s)
                job_kind = JobKind.INTERVAL
            elif kind == "cron":
                if not payload.cron:
                    raise InvalidInput("kind=cron requires 'cron'")
                next_run = next_cron(payload.cron)
                job_kind = JobKind.CRON
            else:
                raise InvalidInput("kind must be once, interval or cron")
        except (TimeParseError, InvalidInput) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with session_scope() as session:
            job = await repo.create_job(
                session,
                name=payload.name or payload.instruction[:60],
                kind=job_kind.value,
                instruction=payload.instruction,
                chat_id=None,
                user_id=None,
                next_run_at=next_run,
                cron_expr=payload.cron,
                interval_s=interval_s,
                max_runs=max_runs,
            )
            return {
                "created": True,
                "job_id": job.id,
                "kind": job.kind,
                "next_run_at": next_run.isoformat(),
            }

    @app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_api_access)])
    async def delete_job(job_id: str) -> dict[str, Any]:
        async with session_scope() as session:
            job = await repo.get_job(session, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            await repo.delete_job(session, job.id)
            return {"removed": True, "job_id": job.id}

    # --- autonomy --------------------------------------------------------- #
    @app.get("/api/autonomy", dependencies=[Depends(require_api_access)])
    async def autonomy_get() -> dict[str, Any]:
        return {"level": get_settings().autonomy_level}

    @app.post("/api/autonomy", dependencies=[Depends(require_api_access)])
    async def autonomy_set(payload: AutonomyRequest) -> dict[str, Any]:
        level = (payload.level or "").strip().lower()
        valid = {"balanced", "high", "paranoid"}
        if level not in valid:
            raise HTTPException(
                status_code=400, detail=f"level must be one of: {', '.join(sorted(valid))}"
            )
        async with session_scope() as session:
            await repo.set_setting(session, "autonomy_level", level)
        get_settings().autonomy_level = level
        return {"level": level}

    # --- bridge / integration status ---------------------------------- #
    @app.get("/api/bridges/status", dependencies=[Depends(require_api_access)])
    async def bridges_status() -> dict[str, Any]:
        from app.integrations.bridge_client import BridgeError, teams_bridge, whatsapp_bridge

        async def _status(factory) -> dict[str, Any]:
            try:
                return await factory().status()
            except BridgeError as exc:
                return {"ok": False, "error": str(exc)}

        return {
            "whatsapp": await _status(whatsapp_bridge),
            "teams": await _status(teams_bridge),
        }

    @app.get("/api/telegram_user/status", dependencies=[Depends(require_api_access)])
    async def telegram_user_status() -> dict[str, Any]:
        from app.integrations.telegram_user import get_userbot

        return await get_userbot().status()

    # --- chat (ChatGPT-style web conversation) --------------------------- #
    # Shares the exact same brain as the Telegram "just talk to me" flow
    # (app.agent.conversation.handle_message) and the same chat_id (the
    # owner's), so a message sent from the web dashboard and one sent from
    # Telegram land in one continuous conversation either surface can see.
    @app.get("/api/chat/history", dependencies=[Depends(require_api_access)])
    async def chat_history(limit: int = 50) -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            return {"chat_id": None, "thread_id": None, "messages": []}
        async with session_scope() as session:
            row = await repo.ensure_session(session, chat_id)
            turns = await repo.recent_messages(
                session, chat_id, limit=min(limit, 200), thread_id=row.current_thread_id
            )
            return {
                "chat_id": chat_id,
                "thread_id": row.current_thread_id,
                "messages": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "task_id": t.task_id,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in turns
                ],
            }

    @app.get("/api/chat/threads", dependencies=[Depends(require_api_access)])
    async def chat_threads() -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            return {"count": 0, "threads": []}
        async with session_scope() as session:
            threads = await repo.list_threads(session, chat_id, limit=50)
            return {
                "count": len(threads),
                "threads": [
                    {
                        "thread_id": t["thread_id"],
                        "title": t["title"],
                        "message_count": t["message_count"],
                        "last_active_at": t["last_active_at"].isoformat()
                        if t["last_active_at"] else None,
                        "is_current": t["is_current"],
                    }
                    for t in threads
                ],
            }

    @app.post("/api/chat/threads/new", dependencies=[Depends(require_api_access)])
    async def chat_new_thread() -> dict[str, Any]:
        """ChatGPT-style "New chat": same mechanism as Telegram's /new command.
        The previous thread is never deleted - it stays listed in
        GET /api/chat/threads and reachable via POST .../switch.
        """
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            raise HTTPException(
                status_code=503,
                detail="no owner chat configured (set TELEGRAM_ALLOWED_USER_IDS)",
            )
        async with session_scope() as session:
            new_thread_id = await repo.reset_session(session, chat_id)
        return {"thread_id": new_thread_id}

    @app.post("/api/chat/threads/{thread_id}/switch", dependencies=[Depends(require_api_access)])
    async def chat_switch_thread(thread_id: str) -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            raise HTTPException(
                status_code=503,
                detail="no owner chat configured (set TELEGRAM_ALLOWED_USER_IDS)",
            )
        async with session_scope() as session:
            threads = await repo.list_threads(session, chat_id, limit=200)
            if not any(t["thread_id"] == thread_id for t in threads):
                raise HTTPException(status_code=404, detail="thread not found")
            await repo.switch_thread(session, chat_id, thread_id)
        return {"thread_id": thread_id}

    @app.post("/api/chat", dependencies=[Depends(require_api_access)])
    async def chat_send(payload: ChatRequest) -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            raise HTTPException(
                status_code=503,
                detail="no owner chat configured (set TELEGRAM_ALLOWED_USER_IDS)",
            )

        from app.agent.conversation import handle_message

        reply = await handle_message(chat_id, chat_id, payload.message)
        return {
            "reply": reply.text,
            "intent": reply.intent.value,
            "task_id": reply.task_id,
            "created_task": reply.created_task,
        }

    @app.post("/api/chat/upload", dependencies=[Depends(require_api_access)])
    async def chat_upload(file: UploadFile) -> dict[str, Any]:
        """Save an uploaded file the exact same way Telegram's F.document
        handler does (app/telegram/bot.py): into uploads/, remembered as
        pending_upload on the owner's chat session so it auto-attaches to
        whatever instruction (web chat or Telegram) comes next.
        """
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            raise HTTPException(
                status_code=503,
                detail="no owner chat configured (set TELEGRAM_ALLOWED_USER_IDS)",
            )

        from app.security import rel_path, safe_path

        raw_name = file.filename or "upload"
        safe_name = "".join(c for c in raw_name if c not in '\\/:*?"<>|').strip() or "file"
        target = safe_path(f"uploads/{safe_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            stem, _, ext = safe_name.rpartition(".")
            stem = stem or safe_name
            target = safe_path(f"uploads/{stem}_{int(time.time())}{'.' + ext if ext else ''}")

        size = 0
        max_bytes = settings.max_file_bytes
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"file is over the {settings.max_file_mb} MB limit",
                    )
                out.write(chunk)

        rel = rel_path(target)
        async with session_scope() as session:
            row = await repo.ensure_session(session, chat_id)
            ctx = dict(row.context or {})
            ctx["pending_upload"] = {"path": rel, "name": safe_name}
            await repo.update_session(session, chat_id, context=ctx)
            # Record the upload as a visible chat message too - otherwise it
            # only lived in invisible session context and disappeared the
            # moment history reloaded, looking like the upload never happened.
            await repo.add_message(
                session, chat_id=chat_id, role="user",
                content=f"\U0001F4CE Uploaded: {safe_name}",
                thread_id=row.current_thread_id,
            )

        from app.automation import otp_bot

        await otp_bot.enqueue_file(rel, safe_name)

        return {"saved": True, "path": rel, "name": safe_name}

    @app.get("/api/chat/pending_upload", dependencies=[Depends(require_api_access)])
    async def chat_pending_upload() -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            return {"pending_upload": None}
        async with session_scope() as session:
            row = await repo.ensure_session(session, chat_id)
            return {"pending_upload": dict(row.context or {}).get("pending_upload")}

    @app.delete("/api/chat/pending_upload", dependencies=[Depends(require_api_access)])
    async def chat_clear_pending_upload() -> dict[str, Any]:
        settings = get_settings()
        chat_id = settings.owner_chat_id
        if chat_id is None:
            return {"cleared": True}
        async with session_scope() as session:
            row = await repo.ensure_session(session, chat_id)
            ctx = dict(row.context or {})
            ctx.pop("pending_upload", None)
            await repo.update_session(session, chat_id, context=ctx)
        return {"cleared": True}

    # --- OTP-number bot automation (deterministic, no LLM in the loop) -- #
    @app.get("/api/otpbot/config", dependencies=[Depends(require_api_access)])
    async def otpbot_get_config() -> dict[str, Any]:
        from app.automation import otp_bot

        return await otp_bot.get_config()

    @app.post("/api/otpbot/config", dependencies=[Depends(require_api_access)])
    async def otpbot_set_config(payload: OtpBotConfigRequest) -> dict[str, Any]:
        from app.automation import otp_bot

        patch = {k: v for k, v in payload.model_dump().items() if v is not None}
        return await otp_bot.save_config(patch)

    @app.get("/api/otpbot/status", dependencies=[Depends(require_api_access)])
    async def otpbot_status() -> dict[str, Any]:
        from app.automation import otp_bot

        config = await otp_bot.get_config()
        last_result = await otp_bot.get_last_result()
        last_start = await otp_bot.get_last_start_result()
        queue = await otp_bot.get_queue()
        active_files = await otp_bot.get_active_files()
        awaiting = await otp_bot.get_awaiting_tag_entry()
        return {
            "config": config,
            "last_result": last_result,
            "last_start": last_start,
            "queue": queue,
            "active_files": active_files,
            "awaiting_tag_for": awaiting,
        }

    @app.get("/api/otpbot/queue", dependencies=[Depends(require_api_access)])
    async def otpbot_get_queue() -> dict[str, Any]:
        from app.automation import otp_bot

        return {"queue": await otp_bot.get_queue(), "active_files": await otp_bot.get_active_files()}

    @app.post("/api/otpbot/queue/{entry_id}/tag", dependencies=[Depends(require_api_access)])
    async def otpbot_set_tag(entry_id: str, payload: OtpBotTagRequest) -> dict[str, Any]:
        from app.automation import otp_bot

        entry = await otp_bot.set_queue_tag(entry_id, payload.tag)
        if entry is None:
            raise HTTPException(status_code=404, detail="queue entry not found")
        return entry

    @app.delete("/api/otpbot/queue/{entry_id}", dependencies=[Depends(require_api_access)])
    async def otpbot_remove_from_queue(entry_id: str) -> dict[str, Any]:
        from app.automation import otp_bot

        removed = await otp_bot.remove_from_queue(entry_id)
        if not removed:
            raise HTTPException(status_code=404, detail="queue entry not found")
        return {"removed": True, "entry_id": entry_id}

    @app.post("/api/otpbot/start", dependencies=[Depends(require_api_access)])
    async def otpbot_start() -> dict[str, Any]:
        """Same deterministic entrypoint the "start"/"done" chat trigger uses -
        refuses (with a clear missing_tags list) if any queued file has no tag
        yet, so the dashboard button and the chat phrase never disagree.
        """
        from app.automation import otp_bot

        return await otp_bot.start_automation()

    @app.post("/api/otpbot/stop", dependencies=[Depends(require_api_access)])
    async def otpbot_stop() -> dict[str, Any]:
        from app.automation import otp_bot

        return await otp_bot.stop_automation()

    @app.post("/api/otpbot/run_now", dependencies=[Depends(require_api_access)])
    async def otpbot_run_now() -> dict[str, Any]:
        """Trigger one cycle immediately, without waiting for the scheduler's
        own interval - lets the dashboard's "Run now" button give instant
        feedback instead of the owner wondering if the setting even saved.
        """
        from dataclasses import asdict

        from app.automation import otp_bot

        config = await otp_bot.get_config()
        result = await otp_bot.run_cycle(config)
        return asdict(result)

    # --- static web dashboard ------------------------------------------- #
    # Mounted last so it never shadows an /api/* or /health route above.
    # No auth at the HTTP layer here: index.html itself is a login gate
    # (checks /api/auth/me on load), and every API call it makes is
    # separately protected by require_api_access.
    webui_dir = Path(__file__).resolve().parent / "webui"
    if webui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")

    return app
