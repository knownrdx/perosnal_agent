"""Tool executor.

This is the *only* path from an LLM decision to a real action.  It enforces,
in order:

1. tool exists and is enabled
2. permission level granted to the task
3. HIGH_RISK approval gate
4. idempotency guard for side-effecting tools
5. execution with timeout/retry (delegated to Tool.run)
6. persistent audit record of every attempt
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import get_settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import ApprovalStatus, FailureKind, ToolCallStatus
from app.logging_conf import get_logger
from app.security import ApprovalRequired, Permission, PermissionDenied, autonomy, check_permission
from app.tools import registry
from app.tools.base import ToolContext, ToolResult

log = get_logger(__name__)


def operation_key(task_id: str | None, tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{task_id or 'adhoc'}:{tool}:{digest}"


class Executor:
    """Executes a single validated tool call for a task."""

    def __init__(self, notifier: Any | None = None) -> None:
        self.notifier = notifier

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        settings = get_settings()
        tool = registry.get(tool_name)

        if tool is None:
            available = ", ".join(registry.names())
            result = ToolResult(
                ok=False,
                error=f"unknown tool '{tool_name}'. Available tools: {available}",
                kind=FailureKind.INVALID_INPUT,
            )
            await self._audit(ctx, tool_name, args, result, ToolCallStatus.DENIED)
            return result

        # --- permission ------------------------------------------------- #
        try:
            check_permission(tool.permission, ctx.permission)
        except PermissionDenied as exc:
            result = ToolResult(ok=False, error=str(exc), kind=FailureKind.PERMANENT)
            await self._audit(ctx, tool_name, args, result, ToolCallStatus.DENIED)
            return result

        # --- validation (early, so approval requests show clean args) ---- #
        try:
            cleaned = tool.validate(args)
        except Exception as exc:  # ToolError subclasses
            result = ToolResult(ok=False, error=str(exc), kind=FailureKind.INVALID_INPUT)
            await self._audit(ctx, tool_name, args, result, ToolCallStatus.ERROR)
            return result

        # --- approval gate ---------------------------------------------- #
        if tool.permission is Permission.HIGH_RISK and settings.require_approval_high_risk:
            # Routine, reversible, or explicitly-requested work should not
            # interrupt the owner. Only genuinely consequential actions do.
            verdict = await autonomy.evaluate(
                tool_name,
                cleaned,
                user_request=await self._task_request(ctx),
                side_effect=tool.side_effect,
            )
            if verdict.allow:
                log.info(
                    "autonomous_action",
                    extra={
                        "tool": tool_name,
                        "task_id": ctx.task_id,
                        "reason": verdict.reason,
                        "learned": verdict.learned,
                    },
                )
                async with session_scope() as session:
                    await repo.log_event(
                        session,
                        "acted_without_approval",
                        task_id=ctx.task_id,
                        data={
                            "tool": tool_name,
                            "action": autonomy.describe(tool_name, cleaned),
                            "reason": verdict.reason,
                            "learned": verdict.learned,
                        },
                    )
            else:
                decision = await self._approval_state(ctx, tool_name, cleaned, verdict.reason)
                if decision == "pending":
                    result = ToolResult(
                        ok=False,
                        error=f"approval required for {tool_name}",
                        kind=FailureKind.USER_ACTION_REQUIRED,
                    )
                    await self._audit(
                        ctx, tool_name, cleaned, result, ToolCallStatus.PENDING_APPROVAL
                    )
                    raise ApprovalRequired(tool_name, f"{tool_name} needs owner approval")
                if decision == "rejected":
                    result = ToolResult(
                        ok=False,
                        error=f"owner rejected the {tool_name} request",
                        kind=FailureKind.PERMANENT,
                    )
                    await self._audit(ctx, tool_name, cleaned, result, ToolCallStatus.DENIED)
                    return result

        # --- idempotency -------------------------------------------------#
        key: str | None = None
        if tool.side_effect:
            key = operation_key(ctx.task_id, tool_name, cleaned)
            async with session_scope() as session:
                existing = await repo.find_operation(session, key)
            if existing is not None:
                log.info(
                    "tool_idempotent_skip",
                    extra={"tool": tool_name, "task_id": ctx.task_id, "operation_key": key},
                )
                return ToolResult(
                    ok=True,
                    data={
                        **(existing.result or {}),
                        "idempotent_replay": True,
                        "note": "identical action already performed for this task; not repeated",
                    },
                )

        # --- run ---------------------------------------------------------#
        result = await tool.run(cleaned, ctx)

        if result.ok and key is not None:
            async with session_scope() as session:
                await repo.record_operation(
                    session, key=key, kind=tool_name, task_id=ctx.task_id, result=result.data
                )

        status = ToolCallStatus.OK if result.ok else _status_for(result.kind)
        await self._audit(ctx, tool_name, cleaned, result, status, operation_key=key)

        log.info(
            "tool_call",
            extra={
                "tool": tool_name,
                "task_id": ctx.task_id,
                "status": status.value,
                "duration": round(result.duration_ms, 1),
                "attempts": result.attempts,
            },
        )
        return result

    # ------------------------------------------------------------------ #
    async def _task_request(self, ctx: ToolContext) -> str:
        """The owner's own words for this task, used to judge intent."""
        if not ctx.task_id:
            return ""
        async with session_scope() as session:
            task = await repo.get_task(session, ctx.task_id)
        return task.user_request if task is not None else ""

    async def _approval_state(
        self, ctx: ToolContext, tool_name: str, args: dict[str, Any], why: str = ""
    ) -> str:
        """Return 'approved' | 'rejected' | 'pending', creating a request if needed."""
        async with session_scope() as session:
            approval = await repo.pending_approval_for_task(session, ctx.task_id or "")
            if approval is not None and approval.tool == tool_name:
                return "pending"

            # Look for a decided approval matching this exact request.
            from sqlalchemy import select  # local import keeps repo API surface small

            from app.db.models import Approval

            stmt = (
                select(Approval)
                .where(Approval.task_id == (ctx.task_id or ""), Approval.tool == tool_name)
                .order_by(Approval.created_at.desc())
                .limit(5)
            )
            for row in (await session.scalars(stmt)).all():
                if row.args == args:
                    if row.status == ApprovalStatus.APPROVED.value:
                        return "approved"
                    if row.status == ApprovalStatus.REJECTED.value:
                        return "rejected"

            created = await repo.create_approval(
                session,
                task_id=ctx.task_id or "",
                tool=tool_name,
                args=args,
                reason=why or f"{tool_name} is HIGH_RISK and requires explicit approval",
            )
            await repo.log_event(
                session,
                "approval_requested",
                task_id=ctx.task_id,
                data={"tool": tool_name, "approval_id": created.id},
            )
            approval_id = created.id

        if self.notifier is not None:
            await self.notifier.approval_request(
                chat_id=ctx.chat_id,
                approval_id=approval_id,
                task_id=ctx.task_id or "",
                tool=tool_name,
                args=args,
            )
        return "pending"

    async def _audit(
        self,
        ctx: ToolContext,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        status: ToolCallStatus,
        operation_key: str | None = None,
    ) -> None:
        if not ctx.task_id:
            return
        async with session_scope() as session:
            await repo.record_tool_call(
                session,
                task_id=ctx.task_id,
                step=ctx.step,
                tool=tool_name,
                args=args,
                status=status,
                result=result.data if result.ok else {},
                error=result.error,
                failure_kind=(result.kind or FailureKind.UNKNOWN).value if not result.ok else "",
                attempts=result.attempts,
                duration_ms=result.duration_ms,
                operation_key=operation_key,
            )


def _status_for(kind: FailureKind | None) -> ToolCallStatus:
    if kind is FailureKind.TEMPORARY:
        return ToolCallStatus.TIMEOUT
    if kind in {FailureKind.AUTH, FailureKind.PERMANENT}:
        return ToolCallStatus.ERROR
    return ToolCallStatus.ERROR
